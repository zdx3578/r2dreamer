from dataclasses import dataclass

import torch


@dataclass
class Phase2Artifact:
    q_u: torch.Tensor
    target_q: torch.Tensor
    operator_id: torch.Tensor
    operator_conf: torch.Tensor
    operator_embed: torch.Tensor
    context_embed: torch.Tensor
    effect_embed: torch.Tensor
    operator_usage: torch.Tensor
    operator_sample_entropy: torch.Tensor
    operator_usage_entropy: torch.Tensor
    q_b: torch.Tensor
    binding_id: torch.Tensor
    binding_conf: torch.Tensor
    q_sigma: torch.Tensor
    scope: torch.Tensor
    duration: torch.Tensor
    impact: torch.Tensor
    delta_rule_pred: torch.Tensor
    memory_delta_rule: torch.Tensor
    memory_signature_proto: torch.Tensor
    memory_conf: torch.Tensor
    memory_weights: torch.Tensor
    delta_rule_fused: torch.Tensor
    rho_next_pred: torch.Tensor
    fusion_alpha: torch.Tensor
    gate: torch.Tensor
    object_gate: torch.Tensor
    match_gate: torch.Tensor
    warmup_gate: torch.Tensor
