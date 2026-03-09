import math

import torch
import torch.nn.functional as F
from torch import nn


class CounterfactualLocalityHead(nn.Module):
    def __init__(self, obj_slots):
        super().__init__()
        self.obj_slots = int(obj_slots)

    def forward(self, target_delta_obj, delta_obj_pred):
        eps = 1e-6
        pred_slot_mass = delta_obj_pred.abs().mean(dim=-1) + eps
        target_slot_mass = target_delta_obj.abs().mean(dim=-1) + eps
        pred_slot_prob = pred_slot_mass / pred_slot_mass.sum(dim=-1, keepdim=True)
        target_slot_prob = target_slot_mass / target_slot_mass.sum(dim=-1, keepdim=True)

        entropy = -(pred_slot_prob * torch.log(pred_slot_prob)).sum(dim=-1) / math.log(self.obj_slots)
        loss_sparse = entropy.mean()
        loss_conc = F.smooth_l1_loss(pred_slot_prob, target_slot_prob)
        loss_cf = 1.0 - F.cosine_similarity(pred_slot_prob, target_slot_prob, dim=-1).mean()

        return {
            "loss_sparse": loss_sparse,
            "loss_conc": loss_conc,
            "loss_cf": loss_cf,
            "slot_concentration": pred_slot_prob.max(dim=-1).values.mean(),
            "target_slot_prob": target_slot_prob,
        }
