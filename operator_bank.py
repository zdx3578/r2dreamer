import math

import torch
import torch.nn.functional as F
from torch import nn

import tools
from phase1a import _build_mlp


class OperatorBank(nn.Module):
    def __init__(
        self,
        config,
        feat_dim,
        act_dim,
        map_slots,
        map_dim,
        obj_slots,
        obj_dim,
        global_dim,
        rule_dim,
        effect_dim,
        act_name,
        use_norm,
    ):
        super().__init__()
        self.num_operators = int(config.num_operators)
        self.operator_dim = int(config.operator_dim)
        self.temperature = float(config.temperature)

        inp_dim = (
            int(feat_dim)
            + int(act_dim)
            + int(map_slots) * int(map_dim)
            + int(obj_slots) * int(obj_dim)
            + int(global_dim)
            + int(rule_dim)
        )
        self.trunk, out_dim = _build_mlp(inp_dim, int(config.hidden), int(config.layers), act_name, use_norm)
        self.context_proj = nn.Linear(out_dim, self.operator_dim, bias=True)
        self.effect_proj = nn.Linear(int(effect_dim), self.operator_dim, bias=True)
        self.prototypes = nn.Parameter(torch.randn(self.num_operators, self.operator_dim))
        self.apply(tools.weight_init_)

    def forward(self, feat, action, map_view, obj_view, global_view, rule_ctx, z_eff):
        context = torch.cat(
            [
                feat,
                action,
                map_view.reshape(*map_view.shape[:-2], -1),
                obj_view.reshape(*obj_view.shape[:-2], -1),
                global_view,
                rule_ctx,
            ],
            dim=-1,
        )
        hidden = self.trunk(context)
        context_embed = F.normalize(self.context_proj(hidden), dim=-1)
        effect_embed = F.normalize(self.effect_proj(z_eff), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)

        logits = torch.einsum("...d,kd->...k", context_embed, prototypes) / self.temperature
        q_u = torch.softmax(logits, dim=-1)
        target_logits = torch.einsum("...d,kd->...k", effect_embed, prototypes) / self.temperature
        target_q = torch.softmax(target_logits, dim=-1)
        operator_embed = torch.einsum("...k,kd->...d", q_u, self.prototypes)

        sample_entropy = -(q_u * torch.log(q_u + 1e-6)).sum(dim=-1).mean() / math.log(self.num_operators)
        avg_usage = q_u.reshape(-1, self.num_operators).mean(dim=0)
        usage_entropy = -(avg_usage * torch.log(avg_usage + 1e-6)).sum() / math.log(self.num_operators)
        return {
            "q_u_logits": logits,
            "q_u": q_u,
            "target_q": target_q,
            "operator_embed": operator_embed,
            "context_embed": context_embed,
            "effect_embed": effect_embed,
            "sample_entropy": sample_entropy,
            "usage_entropy": usage_entropy,
            "avg_usage": avg_usage,
        }
