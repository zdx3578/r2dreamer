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


def _resize_mask_to_spatial(valid_mask, spatial_hw):
    batch_shape = valid_mask.shape[:-3]
    mask = valid_mask.reshape(-1, 1, *valid_mask.shape[-3:-1]).to(torch.float32)
    resized = F.adaptive_avg_pool2d(mask, spatial_hw)
    return resized.reshape(*batch_shape, spatial_hw[0], spatial_hw[1], 1)


def _pool_spatial_to_slots(spatial, slots):
    batch_shape = spatial.shape[:-3]
    h, w, ch = spatial.shape[-3:]
    x = spatial.reshape(-1, h, w, ch).permute(0, 3, 1, 2)
    rows, cols = _slot_grid(slots)
    pooled = F.adaptive_avg_pool2d(x, (rows, cols)).permute(0, 2, 3, 1).reshape(-1, rows * cols, ch)
    pooled = pooled[:, :slots]
    if pooled.shape[1] < slots:
        pooled = F.pad(pooled, (0, 0, 0, slots - pooled.shape[1]))
    return pooled.reshape(*batch_shape, slots, ch)


class StructuredReadout(nn.Module):
    def __init__(self, config, feat_dim, act_name, use_norm, spatial_shape=None):
        super().__init__()
        self.map_slots = int(config.map_slots)
        self.map_dim = int(config.map_dim)
        self.obj_slots = int(config.obj_slots)
        self.obj_dim = int(config.obj_dim)
        self.global_dim = int(config.global_dim)
        self.rule_dim = int(config.rule_dim)
        self.spatial_shape = tuple(spatial_shape) if spatial_shape is not None else None
        self.query_track_blend = float(getattr(config, "query_track_blend", 0.5))
        self.query_track_stopgrad = bool(getattr(config, "query_track_stopgrad", True))

        self.trunk, out_dim = _build_mlp(feat_dim, int(config.hidden), int(config.layers), act_name, use_norm)
        if self.spatial_shape is None:
            self.map_head = nn.Linear(out_dim, self.map_slots * self.map_dim, bias=True)
            self.obj_head = nn.Linear(out_dim, self.obj_slots * self.obj_dim, bias=True)
            self.global_head = nn.Linear(out_dim, self.global_dim, bias=True)
            self.rule_head = nn.Linear(out_dim, self.rule_dim, bias=True)
        else:
            spatial_dim = int(self.spatial_shape[-1])
            act = getattr(torch.nn, act_name)
            self.spatial_proj = nn.Linear(spatial_dim, int(config.hidden), bias=True)
            self.spatial_norm = (
                nn.RMSNorm(int(config.hidden), eps=1e-04, dtype=torch.float32) if use_norm else nn.Identity()
            )
            self.spatial_act = act()
            self.map_context = nn.Linear(out_dim, self.map_slots * int(config.hidden), bias=True)
            self.obj_context = nn.Linear(out_dim, self.obj_slots * int(config.hidden), bias=True)
            self.map_head = nn.Linear(int(config.hidden), self.map_dim, bias=True)
            self.obj_head = nn.Linear(int(config.hidden), self.obj_dim, bias=True)
            self.global_head = nn.Linear(out_dim + int(config.hidden), self.global_dim, bias=True)
            self.rule_head = nn.Linear(out_dim + int(config.hidden), self.rule_dim, bias=True)
            self.slot_queries = nn.Parameter(torch.empty(self.obj_slots, int(config.hidden)))
            self.prev_obj_proj = nn.Linear(self.obj_dim, int(config.hidden), bias=False)

        self.map_recon = nn.Linear(self.map_slots * self.map_dim, feat_dim, bias=True)
        self.obj_recon = nn.Linear(self.obj_slots * self.obj_dim, feat_dim, bias=True)
        self.global_recon = nn.Linear(self.global_dim + self.rule_dim, feat_dim, bias=True)
        self.apply(tools.weight_init_)
        if self.spatial_shape is not None:
            nn.init.normal_(self.slot_queries, std=0.02)

    def _dynamic_slot_queries(self, prev_slots, dtype):
        static_queries = self.slot_queries.to(dtype).unsqueeze(0)
        if prev_slots is None:
            return static_queries
        tracked_slots = prev_slots.detach() if self.query_track_stopgrad else prev_slots
        tracked_queries = self.prev_obj_proj(tracked_slots)
        blend = max(0.0, min(1.0, self.query_track_blend))
        return (1.0 - blend) * static_queries + blend * tracked_queries

    def _readout_from_spatial(self, hidden, spatial_hidden, spatial_mask):
        single_step = hidden.dim() == 2
        if single_step:
            hidden = hidden.unsqueeze(1)
            spatial_hidden = spatial_hidden.unsqueeze(1)
            spatial_mask = spatial_mask.unsqueeze(1)

        batch_shape = hidden.shape[:-2]
        steps = hidden.shape[-2]
        hidden_flat = hidden.reshape(-1, steps, hidden.shape[-1])
        spatial_flat = spatial_hidden.reshape(-1, steps, *spatial_hidden.shape[-3:])
        mask_flat = spatial_mask.reshape(-1, steps, *spatial_mask.shape[-3:])

        maps = []
        objects = []
        globals_ = []
        rules = []
        prev_slots = None

        for index in range(steps):
            hidden_t = hidden_flat[:, index]
            spatial_t = spatial_flat[:, index]
            mask_t = mask_flat[:, index]

            token_feat = spatial_t.reshape(spatial_t.shape[0], -1, spatial_t.shape[-1])
            token_weight = mask_t.reshape(mask_t.shape[0], -1, 1)
            norm_weight = token_weight / token_weight.sum(dim=-2, keepdim=True).clamp_min(1e-6)
            global_summary = (token_feat * norm_weight).sum(dim=-2)

            map_tokens = _pool_spatial_to_slots(spatial_t, self.map_slots)
            map_tokens = map_tokens + self.map_context(hidden_t).reshape(hidden_t.shape[0], self.map_slots, -1)
            map_out = torch.tanh(self.map_head(map_tokens))

            slot_queries = self._dynamic_slot_queries(prev_slots, token_feat.dtype)
            attn_logits = torch.einsum("bnd,bsd->bsn", token_feat, slot_queries)
            attn_logits = attn_logits / math.sqrt(float(token_feat.shape[-1]))
            attn = torch.softmax(attn_logits, dim=-1) * token_weight.squeeze(-1).unsqueeze(-2)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            pooled_obj_tokens = _pool_spatial_to_slots(spatial_t, self.obj_slots)
            obj_tokens = 0.5 * (torch.einsum("bsn,bnd->bsd", attn, token_feat) + pooled_obj_tokens)
            obj_tokens = obj_tokens + self.obj_context(hidden_t).reshape(hidden_t.shape[0], self.obj_slots, -1)
            obj_out = torch.tanh(self.obj_head(obj_tokens))

            global_input = torch.cat([hidden_t, global_summary], dim=-1)
            global_out = torch.tanh(self.global_head(global_input))
            rule_out = torch.tanh(self.rule_head(global_input))

            maps.append(map_out)
            objects.append(obj_out)
            globals_.append(global_out)
            rules.append(rule_out)
            prev_slots = obj_out

        M_t = torch.stack(maps, dim=1).reshape(*batch_shape, steps, self.map_slots, self.map_dim)
        O_t = torch.stack(objects, dim=1).reshape(*batch_shape, steps, self.obj_slots, self.obj_dim)
        g_t = torch.stack(globals_, dim=1).reshape(*batch_shape, steps, self.global_dim)
        rho_t = torch.stack(rules, dim=1).reshape(*batch_shape, steps, self.rule_dim)

        if single_step:
            M_t = M_t.squeeze(1)
            O_t = O_t.squeeze(1)
            g_t = g_t.squeeze(1)
            rho_t = rho_t.squeeze(1)
        return M_t, O_t, g_t, rho_t

    def forward(self, feat, valid_mask=None, spatial=None):
        hidden = self.trunk(feat)
        batch_shape = feat.shape[:-1]

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

        if spatial is not None and self.spatial_shape is not None:
            spatial = spatial.to(feat.dtype)
            spatial_hidden = self.spatial_act(self.spatial_norm(self.spatial_proj(spatial)))
            if valid_mask is None:
                spatial_mask = torch.ones(
                    *batch_shape,
                    spatial.shape[-3],
                    spatial.shape[-2],
                    1,
                    device=feat.device,
                    dtype=feat.dtype,
                )
            else:
                spatial_mask = _resize_mask_to_spatial(valid_mask, spatial.shape[-3:-1]).to(feat.dtype)
            M_t, O_t, g_t, rho_t = self._readout_from_spatial(hidden, spatial_hidden, spatial_mask)
        else:
            M_t = torch.tanh(self.map_head(hidden)).reshape(*batch_shape, self.map_slots, self.map_dim)
            O_t = torch.tanh(self.obj_head(hidden)).reshape(*batch_shape, self.obj_slots, self.obj_dim)
            g_t = torch.tanh(self.global_head(hidden))
            rho_t = torch.tanh(self.rule_head(hidden))

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
