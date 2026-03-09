import math

import torch
import torch.nn.functional as F
from torch import nn

from cf_locality import CounterfactualLocalityHead
import tools
from utils.slot_matching import align_slots, match_confidence, soft_slot_alignment


class ObjectificationModule(nn.Module):
    def __init__(self, config, obj_slots, obj_dim, effect_dim):
        super().__init__()
        self.obj_slots = int(obj_slots)
        self.obj_dim = int(obj_dim)
        self.effect_dim = int(effect_dim)
        self.temperature = float(config.temperature)
        self.identity_temperature = float(getattr(config, "identity_temperature", 0.25))
        self.num_motifs = int(config.num_motifs)
        self.ema_decay = float(config.ema_decay)
        self.sinkhorn_iters = int(getattr(config, "sinkhorn_iters", 10))

        self.w_match = float(config.w_match)
        self.w_temp = float(config.w_temp)
        self.w_smooth = float(config.w_smooth)
        self.w_cycle = float(getattr(config, "w_cycle", 0.5))
        self.w_contrast = float(getattr(config, "w_contrast", 0.5))
        self.w_sparse = float(config.w_sparse)
        self.w_conc = float(config.w_conc)
        self.w_cf = float(config.w_cf)
        self.w_pair = float(config.w_pair)
        self.w_motif = float(config.w_motif)
        self.w_reuse = float(config.w_reuse)

        match_dim = int(config.match_dim)
        rel_dim = int(config.relation_dim)
        self.slot_proj = nn.Linear(self.obj_dim, match_dim, bias=True)
        self.rel_proj = nn.Linear(self.obj_dim, rel_dim, bias=True)
        self.motif_head = nn.Linear(self.effect_dim, self.num_motifs, bias=True)
        self.motif_bank = nn.Parameter(torch.zeros(self.num_motifs, self.obj_slots))
        self.cf_locality = CounterfactualLocalityHead(self.obj_slots)
        self.register_buffer("running_obj_losses", torch.ones(3, dtype=torch.float32))
        self.apply(tools.weight_init_)

    def forward(self, O_t, next_O_t, z_eff, delta_O_pred, event_target, obj_mask=None):
        slot_weight = None if obj_mask is None else obj_mask.squeeze(-1).clamp_min(0.0)
        pred_next = O_t + delta_O_pred
        stable = self._stable_losses(O_t, next_O_t, pred_next, event_target, slot_weight)
        local = self._local_losses(next_O_t - O_t, delta_O_pred, slot_weight)
        rel = self._relational_losses(O_t, next_O_t, pred_next, z_eff, local["target_slot_prob"], slot_weight)

        loss_obj_stable = (
            self.w_match * stable["loss_match"]
            + self.w_temp * stable["loss_temp"]
            + self.w_smooth * stable["loss_smooth"]
            + self.w_cycle * stable["loss_cycle"]
            + self.w_contrast * stable["loss_contrast"]
        )
        loss_obj_local = self.w_sparse * local["loss_sparse"] + self.w_conc * local["loss_conc"] + self.w_cf * local["loss_cf"]
        loss_obj_rel = self.w_pair * rel["loss_pair"] + self.w_motif * rel["loss_motif"] + self.w_reuse * rel["loss_reuse"]

        m_obj = self._objectness_score(loss_obj_stable, loss_obj_local, loss_obj_rel)
        object_interface_score = torch.stack(
            [
                stable["match_margin_score"],
                stable["cycle_score"],
                stable["identity_score"],
                torch.clamp(local["slot_concentration"].detach(), 0.0, 1.0),
            ]
        ).mean()
        object_interface_score = torch.clamp(object_interface_score, 0.0, 1.0)
        return {
            "loss_obj_stable": loss_obj_stable,
            "loss_obj_local": loss_obj_local,
            "loss_obj_rel": loss_obj_rel,
            "objectness_score": m_obj,
            "slot_match_score": stable["match_score"],
            "slot_match_random": stable["match_random_score"],
            "slot_match_margin": stable["match_margin"],
            "slot_match_margin_score": stable["match_margin_score"],
            "slot_cycle_score": stable["cycle_score"],
            "slot_identity_score": stable["identity_score"],
            "slot_concentration": local["slot_concentration"],
            "motif_usage_entropy": rel["motif_usage_entropy"],
            "object_interface_score": object_interface_score,
        }

    def _weighted_mean(self, value, weight):
        if weight is None:
            return value.mean()
        weight = weight.to(value.dtype)
        while weight.dim() < value.dim():
            weight = weight.unsqueeze(-1)
        weight = torch.broadcast_to(weight, value.shape)
        numer = (value * weight).sum()
        denom = weight.sum().clamp_min(1e-6)
        return numer / denom

    def _stable_losses(self, O_t, next_O_t, pred_next, event_target, slot_weight):
        current = F.normalize(self.slot_proj(O_t), dim=-1)
        nxt = F.normalize(self.slot_proj(next_O_t), dim=-1)
        match = soft_slot_alignment(current, nxt, self.temperature, sinkhorn_iters=self.sinkhorn_iters)
        backward_match = soft_slot_alignment(nxt, current, self.temperature, sinkhorn_iters=self.sinkhorn_iters)
        aligned_next = align_slots(match, next_O_t)
        aligned_next_proj = align_slots(match, nxt)

        match_err = F.smooth_l1_loss(pred_next, aligned_next, reduction="none").mean(dim=-1)
        temp_err = 1.0 - F.cosine_similarity(pred_next, aligned_next, dim=-1)
        loss_match = self._weighted_mean(match_err, slot_weight)
        loss_temp = self._weighted_mean(temp_err, slot_weight)

        non_event = 1.0 - event_target.to(O_t.dtype).squeeze(-1)
        smooth = (O_t - aligned_next).pow(2).mean(dim=-1)
        smooth_weight = non_event.unsqueeze(-1)
        if slot_weight is not None:
            smooth_weight = smooth_weight * slot_weight
        loss_smooth = self._weighted_mean(smooth, smooth_weight)

        cycle = torch.matmul(match, backward_match)
        identity = torch.eye(self.obj_slots, device=cycle.device, dtype=cycle.dtype)
        cycle_err = (cycle - identity).pow(2)
        if slot_weight is None:
            loss_cycle = cycle_err.mean()
        else:
            pair_weight = slot_weight.unsqueeze(-1) * slot_weight.unsqueeze(-2)
            loss_cycle = self._weighted_mean(cycle_err, pair_weight)

        contrast_logits = torch.einsum("...id,...jd->...ij", current, aligned_next_proj)
        contrast_logits = contrast_logits / max(1e-6, self.identity_temperature)
        flat_logits = contrast_logits.reshape(-1, self.obj_slots, self.obj_slots)
        flat_targets = torch.arange(self.obj_slots, device=flat_logits.device).unsqueeze(0).expand(flat_logits.shape[0], -1)
        contrast_err = F.cross_entropy(flat_logits.reshape(-1, self.obj_slots), flat_targets.reshape(-1), reduction="none")
        if slot_weight is None:
            loss_contrast = contrast_err.mean()
        else:
            contrast_weight = slot_weight.reshape(-1, self.obj_slots).reshape(-1).to(contrast_err.dtype)
            loss_contrast = (contrast_err * contrast_weight).sum() / contrast_weight.sum().clamp_min(1e-6)

        match_score = match_confidence(match)
        match_random_score = self._shuffled_match_baseline(current, nxt)
        match_margin = match_score - match_random_score
        match_margin_score = torch.clamp(match_margin / (1.0 - match_random_score).clamp_min(1e-6), 0.0, 1.0)
        return {
            "loss_match": loss_match,
            "loss_temp": loss_temp,
            "loss_smooth": loss_smooth,
            "loss_cycle": loss_cycle,
            "loss_contrast": loss_contrast,
            "match_score": match_score,
            "match_random_score": match_random_score,
            "match_margin": match_margin,
            "match_margin_score": match_margin_score,
            "cycle_score": torch.clamp(1.0 - loss_cycle.detach(), 0.0, 1.0),
            "identity_score": torch.clamp(torch.exp(-loss_contrast.detach()), 0.0, 1.0),
        }

    def _shuffled_match_baseline(self, current, nxt):
        flat_current = current.reshape(-1, self.obj_slots, current.shape[-1])
        flat_nxt = nxt.reshape(-1, self.obj_slots, nxt.shape[-1])
        if flat_current.shape[0] <= 1:
            return current.new_tensor(1.0 / float(max(1, self.obj_slots)))
        shuffled_nxt = flat_nxt.roll(shifts=1, dims=0).reshape_as(nxt)
        shuffled_match = soft_slot_alignment(
            current,
            shuffled_nxt,
            self.temperature,
            sinkhorn_iters=self.sinkhorn_iters,
        )
        return match_confidence(shuffled_match)

    def _local_losses(self, target_delta_O, delta_O_pred, slot_weight):
        return self.cf_locality(target_delta_O, delta_O_pred, slot_weight)

    def _relational_losses(self, O_t, next_O_t, pred_next, z_eff, target_slot_prob, slot_weight):
        current_rel = self._relation_matrix(O_t)
        next_rel = self._relation_matrix(next_O_t)
        pred_rel = self._relation_matrix(pred_next)
        pair_err = F.smooth_l1_loss(pred_rel, next_rel, reduction="none")
        if slot_weight is None:
            loss_pair = pair_err.mean()
        else:
            pair_weight = slot_weight.unsqueeze(-1) * slot_weight.unsqueeze(-2)
            loss_pair = self._weighted_mean(pair_err, pair_weight)

        motif_logits = self.motif_head(z_eff)
        motif_weights = torch.softmax(motif_logits, dim=-1)
        motif_bank = torch.softmax(self.motif_bank, dim=-1)
        motif_mix = torch.einsum("...k,kn->...n", motif_weights, motif_bank)
        loss_motif = -(target_slot_prob * torch.log(motif_mix + 1e-6)).sum(dim=-1).mean()

        sample_entropy = -(motif_weights * torch.log(motif_weights + 1e-6)).sum(dim=-1).mean() / math.log(self.num_motifs)
        reduce_dims = tuple(range(motif_weights.dim() - 1))
        avg_usage = motif_weights.mean(dim=reduce_dims)
        uniform = torch.full_like(avg_usage, 1.0 / self.num_motifs)
        balance = (avg_usage * (torch.log(avg_usage + 1e-6) - torch.log(uniform))).sum()
        loss_reuse = 0.5 * sample_entropy + 0.5 * balance

        return {
            "loss_pair": loss_pair,
            "loss_motif": loss_motif,
            "loss_reuse": loss_reuse,
            "motif_usage_entropy": sample_entropy,
        }

    def _relation_matrix(self, O_t):
        rel = F.normalize(self.rel_proj(O_t), dim=-1)
        return torch.einsum("...id,...jd->...ij", rel, rel)

    def _objectness_score(self, stable_loss, local_loss, rel_loss):
        current = torch.stack([stable_loss.detach(), local_loss.detach(), rel_loss.detach()])
        with torch.no_grad():
            self.running_obj_losses.copy_(
                self.ema_decay * self.running_obj_losses + (1.0 - self.ema_decay) * current.to(self.running_obj_losses)
            )
        score = torch.exp(-current / (self.running_obj_losses + 1e-6)).mean()
        return torch.clamp(score, 0.0, 1.0)
