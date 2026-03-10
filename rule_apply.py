import torch
from torch import nn


class RuleApply(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_memory_fusion = bool(getattr(config, "use_memory_fusion", True))

    def forward(self, rho_t, delta_rule_pred, memory_delta_rule, operator_conf, binding_conf, memory_conf, gate):
        gate = gate.to(delta_rule_pred.dtype)
        if not self.use_memory_fusion:
            alpha = torch.ones_like(memory_conf)
            blended_delta = delta_rule_pred
        else:
            head_conf = operator_conf * binding_conf
            memory_weight = memory_conf
            alpha = head_conf / (head_conf + memory_weight + 1e-6)
            blended_delta = alpha * delta_rule_pred + (1.0 - alpha) * memory_delta_rule
        delta_rule_fused = gate * blended_delta
        return {
            "alpha": alpha,
            "delta_rule_fused": delta_rule_fused,
            "rho_next_pred": rho_t + delta_rule_fused,
        }
