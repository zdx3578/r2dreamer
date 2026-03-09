import math

import torch
import torch.nn.functional as F
from torch import nn

import tools


def _build_mlp(inp_dim, hidden_dim, layers, act_name, use_norm):
    act = getattr(torch.nn, act_name)
    modules = []
    dim = int(inp_dim)
    for _ in range(int(layers)):
        modules.append(nn.Linear(dim, hidden_dim, bias=True))
        if use_norm:
            modules.append(nn.RMSNorm(hidden_dim, eps=1e-04, dtype=torch.float32))
        modules.append(act())
        dim = int(hidden_dim)
    return nn.Sequential(*modules), dim


def _slot_grid(num_slots):
    rows = max(1, int(math.sqrt(num_slots)))
    cols = int(math.ceil(float(num_slots) / float(rows)))
    return rows, cols


def _pool_mask_to_slots(valid_mask, slots):
    batch_shape = valid_mask.shape[:-3]
    mask = valid_mask.reshape(-1, 1, *valid_mask.shape[-3:-1]).to(torch.float32)
    rows, cols = _slot_grid(slots)
    pooled = F.adaptive_avg_pool2d(mask, (rows, cols)).flatten(2)[..., :slots]
    if pooled.shape[-1] < slots:
        pooled = F.pad(pooled, (0, slots - pooled.shape[-1]))
    return pooled.reshape(*batch_shape, slots, 1)


class StructuredReadout(nn.Module):
    def __init__(self, config, feat_dim, act_name, use_norm):
        super().__init__()
        self.map_slots = int(config.map_slots)
        self.map_dim = int(config.map_dim)
        self.obj_slots = int(config.obj_slots)
        self.obj_dim = int(config.obj_dim)
        self.global_dim = int(config.global_dim)
        self.rule_dim = int(config.rule_dim)

        self.trunk, out_dim = _build_mlp(feat_dim, int(config.hidden), int(config.layers), act_name, use_norm)
        self.map_head = nn.Linear(out_dim, self.map_slots * self.map_dim, bias=True)
        self.obj_head = nn.Linear(out_dim, self.obj_slots * self.obj_dim, bias=True)
        self.global_head = nn.Linear(out_dim, self.global_dim, bias=True)
        self.rule_head = nn.Linear(out_dim, self.rule_dim, bias=True)

        self.map_recon = nn.Linear(self.map_slots * self.map_dim, feat_dim, bias=True)
        self.obj_recon = nn.Linear(self.obj_slots * self.obj_dim, feat_dim, bias=True)
        self.global_recon = nn.Linear(self.global_dim + self.rule_dim, feat_dim, bias=True)
        self.apply(tools.weight_init_)

    def forward(self, feat, valid_mask=None):
        hidden = self.trunk(feat)
        batch_shape = feat.shape[:-1]
        M_t = torch.tanh(self.map_head(hidden)).reshape(*batch_shape, self.map_slots, self.map_dim)
        O_t = torch.tanh(self.obj_head(hidden)).reshape(*batch_shape, self.obj_slots, self.obj_dim)
        g_t = torch.tanh(self.global_head(hidden))
        rho_t = torch.tanh(self.rule_head(hidden))

        if valid_mask is None:
            map_mask = torch.ones(*batch_shape, self.map_slots, 1, device=feat.device, dtype=feat.dtype)
            obj_mask = torch.ones(*batch_shape, self.obj_slots, 1, device=feat.device, dtype=feat.dtype)
            valid_ratio = torch.ones(*batch_shape, 1, device=feat.device, dtype=feat.dtype)
        else:
            valid_mask = valid_mask.to(feat.dtype)
            map_mask = _pool_mask_to_slots(valid_mask, self.map_slots).to(feat.dtype)
            flat_map_mask = map_mask.squeeze(-1).reshape(-1, 1, self.map_slots)
            pooled_obj_mask = F.adaptive_avg_pool1d(flat_map_mask, self.obj_slots)
            obj_mask = pooled_obj_mask.reshape(*batch_shape, self.obj_slots, 1).to(feat.dtype)
            valid_ratio = valid_mask.mean(dim=(-3, -2, -1), keepdim=False).unsqueeze(-1).to(feat.dtype)

        map_gate = 0.25 + 0.75 * map_mask
        obj_gate = 0.25 + 0.75 * obj_mask
        global_gate = 0.25 + 0.75 * valid_ratio
        M_t = M_t * map_gate
        O_t = O_t * obj_gate
        g_t = g_t * global_gate
        rho_t = rho_t * global_gate

        map_flat = M_t.reshape(*batch_shape, -1)
        obj_flat = O_t.reshape(*batch_shape, -1)
        global_flat = torch.cat([g_t, rho_t], dim=-1)
        return {
            "M_t": M_t,
            "O_t": O_t,
            "g_t": g_t,
            "rho_t": rho_t,
            "map_mask": map_mask,
            "obj_mask": obj_mask,
            "valid_ratio": valid_ratio,
            "M_recon": self.map_recon(map_flat),
            "O_recon": self.obj_recon(obj_flat),
            "g_rho_recon": self.global_recon(global_flat),
        }


class EffectModel(nn.Module):
    def __init__(self, config, feat_dim, act_dim, rule_dim, act_name, use_norm):
        super().__init__()
        inp_dim = int(feat_dim) + int(act_dim) + int(rule_dim)
        self.trunk, out_dim = _build_mlp(inp_dim, int(config.hidden), int(config.layers), act_name, use_norm)
        self.out = nn.Linear(out_dim, int(config.latent_dim), bias=True)
        self.apply(tools.weight_init_)

    def forward(self, feat, action, rho_t):
        hidden = self.trunk(torch.cat([feat, action, rho_t], dim=-1))
        return torch.tanh(self.out(hidden))


class EffectHeads(nn.Module):
    def __init__(self, config, latent_dim, map_slots, map_dim, obj_slots, obj_dim, global_dim, act_name, use_norm):
        super().__init__()
        self.map_slots = int(map_slots)
        self.map_dim = int(map_dim)
        self.obj_slots = int(obj_slots)
        self.obj_dim = int(obj_dim)
        self.global_dim = int(global_dim)
        self.trunk, out_dim = _build_mlp(latent_dim, int(config.hidden), int(config.layers), act_name, use_norm)
        self.delta_map = nn.Linear(out_dim, self.map_slots * self.map_dim, bias=True)
        self.delta_obj = nn.Linear(out_dim, self.obj_slots * self.obj_dim, bias=True)
        self.delta_global = nn.Linear(out_dim, self.global_dim, bias=True)
        self.event = nn.Linear(out_dim, 1, bias=True)
        self.apply(tools.weight_init_)

    def forward(self, z_eff):
        hidden = self.trunk(z_eff)
        batch_shape = z_eff.shape[:-1]
        return {
            "delta_map": self.delta_map(hidden).reshape(*batch_shape, self.map_slots, self.map_dim),
            "delta_obj": self.delta_obj(hidden).reshape(*batch_shape, self.obj_slots, self.obj_dim),
            "delta_global": self.delta_global(hidden),
            "event_logits": self.event(hidden),
        }


class ReachabilityHead(nn.Module):
    def __init__(self, config, feat_dim, map_slots, map_dim, act_name, use_norm):
        super().__init__()
        inp_dim = int(feat_dim) + int(map_slots) * int(map_dim)
        self.trunk, out_dim = _build_mlp(inp_dim, int(config.hidden), int(config.layers), act_name, use_norm)
        self.out = nn.Linear(out_dim, 1, bias=True)
        self.apply(tools.weight_init_)

    def forward(self, feat, M_t):
        x = torch.cat([feat, M_t.reshape(*M_t.shape[:-2], -1)], dim=-1)
        return self.out(self.trunk(x))


class GoalProgressHead(nn.Module):
    def __init__(self, config, feat_dim, global_dim, act_name, use_norm):
        super().__init__()
        inp_dim = int(feat_dim) + int(global_dim)
        self.trunk, out_dim = _build_mlp(inp_dim, int(config.hidden), int(config.layers), act_name, use_norm)
        self.out = nn.Linear(out_dim, 1, bias=True)
        self.apply(tools.weight_init_)

    def forward(self, feat, g_t):
        return self.out(self.trunk(torch.cat([feat, g_t], dim=-1)))
