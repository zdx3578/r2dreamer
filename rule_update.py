import torch
from torch import nn

import tools
from phase1a import _build_mlp


class RuleUpdateHead(nn.Module):
    def __init__(self, config, effect_dim, operator_dim, num_bindings, signature_dim, rule_dim, act_name, use_norm):
        super().__init__()
        inp_dim = int(effect_dim) + int(operator_dim) + int(num_bindings) + int(signature_dim)
        self.trunk, out_dim = _build_mlp(inp_dim, int(config.hidden), int(config.layers), act_name, use_norm)
        self.out = nn.Linear(out_dim, int(rule_dim), bias=True)
        self.apply(tools.weight_init_)

    def forward(self, z_eff, operator_embed, q_b, q_sigma):
        x = torch.cat([z_eff, operator_embed, q_b, q_sigma], dim=-1)
        return {"delta_rule": self.out(self.trunk(x))}
