import math

import torch
import torch.nn.functional as F
from torch import nn


class RuleMemory(nn.Module):
    def __init__(self, config, num_operators, num_bindings, signature_dim, rule_dim):
        super().__init__()
        self.num_operators = int(num_operators)
        self.num_bindings = int(num_bindings)
        self.signature_dim = int(signature_dim)
        self.rule_dim = int(rule_dim)
        self.ema_decay = float(getattr(config, "memory_ema_decay", 0.99))
        self.retrieve_temperature = float(getattr(config, "memory_retrieve_temperature", 1.0))

        self.register_buffer(
            "delta_rule_proto", torch.zeros(self.num_operators, self.num_bindings, self.rule_dim), persistent=True
        )
        self.register_buffer(
            "signature_proto", torch.zeros(self.num_operators, self.num_bindings, self.signature_dim), persistent=True
        )
        self.register_buffer("usage_count", torch.zeros(self.num_operators, self.num_bindings), persistent=True)
        self.register_buffer("ema_conf", torch.zeros(self.num_operators, self.num_bindings), persistent=True)

    def retrieve(self, q_u, q_b, q_sigma=None):
        joint = q_u.unsqueeze(-1) * q_b.unsqueeze(-2)
        if self.retrieve_temperature != 1.0:
            joint = joint.clamp_min(1e-6).pow(1.0 / self.retrieve_temperature)
        valid_cells = (self.usage_count > 0).to(joint.dtype)
        weights = joint * valid_cells
        denom = weights.sum(dim=(-2, -1), keepdim=True)
        weights = torch.where(denom > 0, weights / denom.clamp_min(1e-6), torch.zeros_like(weights))

        memory_delta_rule = torch.einsum("...ub,ubr->...r", weights, self.delta_rule_proto)
        memory_signature = torch.einsum("...ub,ubs->...s", weights, self.signature_proto)
        base_conf = torch.einsum("...ub,ub->...", weights, self.ema_conf).unsqueeze(-1)

        if q_sigma is None:
            signature_agreement = base_conf.new_ones(base_conf.shape)
        else:
            signature_agreement = 0.5 * (
                1.0
                + F.cosine_similarity(
                    F.normalize(q_sigma, dim=-1),
                    F.normalize(memory_signature + 1e-6, dim=-1),
                    dim=-1,
                ).unsqueeze(-1)
            )
        memory_conf = torch.clamp(base_conf * signature_agreement, 0.0, 1.0)
        return {
            "memory_delta_rule": memory_delta_rule,
            "memory_signature_proto": memory_signature,
            "memory_conf": memory_conf,
            "memory_weights": weights,
        }

    def update(self, q_u, q_b, q_sigma, delta_rule_target, write_mask):
        flat_q_u = q_u.reshape(-1, self.num_operators)
        flat_q_b = q_b.reshape(-1, self.num_bindings)
        flat_q_sigma = q_sigma.reshape(-1, self.signature_dim)
        flat_delta = delta_rule_target.reshape(-1, self.rule_dim)
        flat_mask = write_mask.reshape(-1).bool()

        if flat_mask.any():
            selected_q_u = flat_q_u[flat_mask]
            selected_q_b = flat_q_b[flat_mask]
            selected_sigma = flat_q_sigma[flat_mask]
            selected_delta = flat_delta[flat_mask]

            operator_conf, operator_id = selected_q_u.max(dim=-1)
            binding_conf, binding_id = selected_q_b.max(dim=-1)
            write_conf = operator_conf * binding_conf

            cell_index = operator_id * self.num_bindings + binding_id
            for idx in cell_index.unique(sorted=True).tolist():
                member = cell_index == idx
                op_idx = idx // self.num_bindings
                bind_idx = idx % self.num_bindings
                conf = write_conf[member]
                conf_sum = conf.sum().clamp_min(1e-6)
                delta_mean = (selected_delta[member] * conf.unsqueeze(-1)).sum(dim=0) / conf_sum
                sigma_mean = (selected_sigma[member] * conf.unsqueeze(-1)).sum(dim=0) / conf_sum
                conf_mean = conf.mean()
                self.delta_rule_proto[op_idx, bind_idx].mul_(self.ema_decay).add_(delta_mean * (1.0 - self.ema_decay))
                self.signature_proto[op_idx, bind_idx].mul_(self.ema_decay).add_(sigma_mean * (1.0 - self.ema_decay))
                self.ema_conf[op_idx, bind_idx].mul_(self.ema_decay).add_(conf_mean * (1.0 - self.ema_decay))
                self.usage_count[op_idx, bind_idx].add_(float(member.sum()))

        total_cells = float(self.num_operators * self.num_bindings)
        occupied = (self.usage_count > 0).to(torch.float32)
        usage_fraction = occupied.mean()
        usage_dist = self.usage_count.reshape(-1)
        usage_dist = usage_dist / usage_dist.sum().clamp_min(1e-6)
        usage_entropy = -(usage_dist * torch.log(usage_dist + 1e-6)).sum()
        usage_entropy = usage_entropy / math.log(max(2, usage_dist.numel()))
        return {
            "write_rate": write_mask.to(torch.float32).mean(),
            "usage_fraction": usage_fraction,
            "usage_entropy": usage_entropy,
        }
