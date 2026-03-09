import torch
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

    def forward(self, feat):
        hidden = self.trunk(feat)
        batch_shape = feat.shape[:-1]
        map_view = torch.tanh(self.map_head(hidden)).reshape(*batch_shape, self.map_slots, self.map_dim)
        obj_view = torch.tanh(self.obj_head(hidden)).reshape(*batch_shape, self.obj_slots, self.obj_dim)
        global_view = torch.tanh(self.global_head(hidden))
        rule_ctx = torch.tanh(self.rule_head(hidden))

        map_flat = map_view.reshape(*batch_shape, -1)
        obj_flat = obj_view.reshape(*batch_shape, -1)
        global_flat = torch.cat([global_view, rule_ctx], dim=-1)
        return {
            "map_view": map_view,
            "obj_view": obj_view,
            "global_view": global_view,
            "rule_ctx": rule_ctx,
            "map_recon": self.map_recon(map_flat),
            "obj_recon": self.obj_recon(obj_flat),
            "global_recon": self.global_recon(global_flat),
        }


class EffectModel(nn.Module):
    def __init__(self, config, feat_dim, act_dim, rule_dim, act_name, use_norm):
        super().__init__()
        inp_dim = int(feat_dim) + int(act_dim) + int(rule_dim)
        self.trunk, out_dim = _build_mlp(inp_dim, int(config.hidden), int(config.layers), act_name, use_norm)
        self.out = nn.Linear(out_dim, int(config.latent_dim), bias=True)
        self.apply(tools.weight_init_)

    def forward(self, feat, action, rule_ctx):
        hidden = self.trunk(torch.cat([feat, action, rule_ctx], dim=-1))
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

    def forward(self, feat, map_view):
        x = torch.cat([feat, map_view.reshape(*map_view.shape[:-2], -1)], dim=-1)
        return self.out(self.trunk(x))


class GoalProgressHead(nn.Module):
    def __init__(self, config, feat_dim, global_dim, act_name, use_norm):
        super().__init__()
        inp_dim = int(feat_dim) + int(global_dim)
        self.trunk, out_dim = _build_mlp(inp_dim, int(config.hidden), int(config.layers), act_name, use_norm)
        self.out = nn.Linear(out_dim, 1, bias=True)
        self.apply(tools.weight_init_)

    def forward(self, feat, global_view):
        return self.out(self.trunk(torch.cat([feat, global_view], dim=-1)))
