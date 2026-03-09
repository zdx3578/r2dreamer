import math

import torch
from torch import nn

import tools
from phase1a import _build_mlp


class BindingHead(nn.Module):
    binding_labels = ("instance", "type", "relation", "region", "backbone")

    def __init__(self, config, operator_dim, context_dim, act_name, use_norm):
        super().__init__()
        self.num_bindings = int(config.num_bindings)
        inp_dim = int(operator_dim) + int(context_dim)
        self.trunk, out_dim = _build_mlp(inp_dim, int(config.hidden), int(config.layers), act_name, use_norm)
        self.out = nn.Linear(out_dim, self.num_bindings, bias=True)
        self.apply(tools.weight_init_)

    def forward(self, operator_embed, context_embed):
        hidden = self.trunk(torch.cat([operator_embed, context_embed], dim=-1))
        logits = self.out(hidden)
        q_b = torch.softmax(logits, dim=-1)
        entropy = -(q_b * torch.log(q_b + 1e-6)).sum(dim=-1).mean() / math.log(self.num_bindings)
        return {
            "q_b_logits": logits,
            "q_b": q_b,
            "entropy": entropy,
        }
