import torch
from torch import nn

import tools
from phase1a import _build_mlp


class SignatureHead(nn.Module):
    def __init__(self, config, operator_dim, context_dim, act_name, use_norm):
        super().__init__()
        inp_dim = int(operator_dim) + int(context_dim)
        self.trunk, out_dim = _build_mlp(inp_dim, int(config.hidden), int(config.layers), act_name, use_norm)
        self.scope = nn.Linear(out_dim, 1, bias=True)
        self.duration = nn.Linear(out_dim, 1, bias=True)
        self.impact = nn.Linear(out_dim, 1, bias=True)
        self.apply(tools.weight_init_)

    def forward(self, operator_embed, context_embed):
        hidden = self.trunk(torch.cat([operator_embed, context_embed], dim=-1))
        scope = torch.sigmoid(self.scope(hidden))
        duration = torch.sigmoid(self.duration(hidden))
        impact = torch.tanh(self.impact(hidden))
        return {
            "scope": scope,
            "duration": duration,
            "impact": impact,
            "q_sigma": torch.cat([scope, duration, impact], dim=-1),
        }
