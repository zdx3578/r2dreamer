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
        self.usage_logit_scale = float(getattr(config, "memory_usage_logit_scale", 0.5))
        self.conf_logit_scale = float(getattr(config, "memory_conf_logit_scale", 0.5))
        self.signature_logit_scale = float(getattr(config, "memory_signature_logit_scale", 1.0))

        self.register_buffer(
            "delta_rule_proto", torch.zeros(self.num_operators, self.num_bindings, self.rule_dim), persistent=True
        )
        self.register_buffer(
            "signature_proto", torch.zeros(self.num_operators, self.num_bindings, self.signature_dim), persistent=True
        )
        self.register_buffer("usage_count", torch.zeros(self.num_operators, self.num_bindings), persistent=True)
        self.register_buffer("write_mass", torch.zeros(self.num_operators, self.num_bindings), persistent=True)
        self.register_buffer("ema_conf", torch.zeros(self.num_operators, self.num_bindings), persistent=True)

    def retrieve(self, q_u, q_b, q_sigma=None):
        joint = q_u.unsqueeze(-1) * q_b.unsqueeze(-2)
        valid_cells = (self.write_mass > 0).to(joint.dtype)
        joint_logit = torch.log(joint.clamp_min(1e-6))

        usage_prior = torch.log1p(self.write_mass)
        usage_prior = usage_prior / usage_prior.amax().clamp_min(1.0)
        conf_prior = self.ema_conf / self.ema_conf.amax().clamp_min(1e-6)

        if q_sigma is None:
            signature_score = joint.new_ones(joint.shape)
        else:
            signature_score = torch.einsum(
                "...s,ubs->...ub",
                F.normalize(q_sigma, dim=-1),
                F.normalize(self.signature_proto + 1e-6, dim=-1),
            )
            signature_score = 0.5 * (1.0 + signature_score)

        logits = joint_logit
        logits = logits + self.usage_logit_scale * usage_prior
        logits = logits + self.conf_logit_scale * conf_prior
        logits = logits + self.signature_logit_scale * signature_score

        flat_logits = logits.reshape(*joint.shape[:-2], -1)
        flat_valid = valid_cells.reshape(1, *([1] * (flat_logits.ndim - 2)), -1).expand_as(flat_logits)
        masked_logits = torch.where(flat_valid > 0, flat_logits, torch.full_like(flat_logits, -1e9))
        flat_weights = F.softmax(masked_logits / self.retrieve_temperature, dim=-1)
        flat_weights = flat_weights * flat_valid
        flat_weights = flat_weights / flat_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = flat_weights.reshape_as(joint)

        memory_delta_rule = torch.einsum("...ub,ubr->...r", weights, self.delta_rule_proto)
        memory_signature = torch.einsum("...ub,ubs->...s", weights, self.signature_proto)
        flat_signature = signature_score.reshape_as(flat_logits)
        flat_conf = conf_prior.reshape(1, *([1] * (flat_logits.ndim - 2)), -1).expand_as(flat_logits)
        top_weight, top_index = flat_weights.max(dim=-1, keepdim=True)
        top_conf = flat_conf.gather(-1, top_index)
        top_signature = flat_signature.gather(-1, top_index)
        memory_conf = torch.clamp(top_weight * top_conf * top_signature, 0.0, 1.0)
        return {
            "memory_delta_rule": memory_delta_rule,
            "memory_signature_proto": memory_signature,
            "memory_conf": memory_conf,
            "memory_weights": weights,
            "memory_top_weight": top_weight,
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
                prev_mass = self.write_mass[op_idx, bind_idx]
                new_mass = prev_mass + conf_sum
                blend = (conf_sum / new_mass.clamp_min(1e-6)).to(self.delta_rule_proto.dtype)
                delta_mean = (selected_delta[member] * conf.unsqueeze(-1)).sum(dim=0) / conf_sum
                sigma_mean = (selected_sigma[member] * conf.unsqueeze(-1)).sum(dim=0) / conf_sum
                conf_mean = conf.mean()
                self.delta_rule_proto[op_idx, bind_idx].lerp_(delta_mean, blend)
                self.signature_proto[op_idx, bind_idx].lerp_(sigma_mean, blend)
                self.ema_conf[op_idx, bind_idx].mul_(self.ema_decay).add_(conf_mean * (1.0 - self.ema_decay))
                self.write_mass[op_idx, bind_idx].copy_(new_mass)
                self.usage_count[op_idx, bind_idx].add_(float(member.sum()))

        occupied = (self.write_mass > 0).to(torch.float32)
        usage_fraction = occupied.mean()
        usage_dist = self.write_mass.reshape(-1)
        usage_dist = usage_dist / usage_dist.sum().clamp_min(1e-6)
        usage_entropy = -(usage_dist * torch.log(usage_dist + 1e-6)).sum()
        usage_entropy = usage_entropy / math.log(max(2, usage_dist.numel()))
        return {
            "write_rate": write_mask.to(torch.float32).mean(),
            "usage_fraction": usage_fraction,
            "usage_entropy": usage_entropy,
        }
