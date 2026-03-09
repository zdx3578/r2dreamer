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
        self.num_motifs = int(config.num_motifs)
        self.ema_decay = float(config.ema_decay)

        self.w_match = float(config.w_match)
        self.w_temp = float(config.w_temp)
        self.w_smooth = float(config.w_smooth)
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

    def forward(self, obj_view, next_obj_view, z_eff, delta_obj_pred, event_target):
        pred_next = obj_view + delta_obj_pred
        stable = self._stable_losses(obj_view, next_obj_view, pred_next, event_target)
        local = self._local_losses(next_obj_view - obj_view, delta_obj_pred)
        rel = self._relational_losses(obj_view, next_obj_view, pred_next, z_eff, local["target_slot_prob"])

        loss_obj_stable = self.w_match * stable["loss_match"] + self.w_temp * stable["loss_temp"] + self.w_smooth * stable["loss_smooth"]
        loss_obj_local = self.w_sparse * local["loss_sparse"] + self.w_conc * local["loss_conc"] + self.w_cf * local["loss_cf"]
        loss_obj_rel = self.w_pair * rel["loss_pair"] + self.w_motif * rel["loss_motif"] + self.w_reuse * rel["loss_reuse"]

        m_obj = self._objectness_score(loss_obj_stable, loss_obj_local, loss_obj_rel)
        return {
            "loss_obj_stable": loss_obj_stable,
            "loss_obj_local": loss_obj_local,
            "loss_obj_rel": loss_obj_rel,
            "objectness_score": m_obj,
            "slot_match_score": stable["match_score"],
            "slot_concentration": local["slot_concentration"],
            "motif_usage_entropy": rel["motif_usage_entropy"],
        }

    def _stable_losses(self, obj_view, next_obj_view, pred_next, event_target):
        current = F.normalize(self.slot_proj(obj_view), dim=-1)
        nxt = F.normalize(self.slot_proj(next_obj_view), dim=-1)
        match = soft_slot_alignment(current, nxt, self.temperature)
        aligned_next = align_slots(match, next_obj_view)

        loss_match = F.smooth_l1_loss(pred_next, aligned_next)
        loss_temp = 1.0 - F.cosine_similarity(pred_next, aligned_next, dim=-1).mean()

        non_event = 1.0 - event_target.to(obj_view.dtype).squeeze(-1)
        smooth = (obj_view - aligned_next).pow(2).mean(dim=-1)
        loss_smooth = (smooth * non_event.unsqueeze(-1)).sum() / (non_event.sum() * self.obj_slots + 1e-6)
        match_score = match_confidence(match)
        return {
            "loss_match": loss_match,
            "loss_temp": loss_temp,
            "loss_smooth": loss_smooth,
            "match_score": match_score,
        }

    def _local_losses(self, target_delta_obj, delta_obj_pred):
        return self.cf_locality(target_delta_obj, delta_obj_pred)

    def _relational_losses(self, obj_view, next_obj_view, pred_next, z_eff, target_slot_prob):
        current_rel = self._relation_matrix(obj_view)
        next_rel = self._relation_matrix(next_obj_view)
        pred_rel = self._relation_matrix(pred_next)
        loss_pair = F.smooth_l1_loss(pred_rel, next_rel)

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

    def _relation_matrix(self, obj_view):
        rel = F.normalize(self.rel_proj(obj_view), dim=-1)
        return torch.einsum("...id,...jd->...ij", rel, rel)

    def _objectness_score(self, stable_loss, local_loss, rel_loss):
        current = torch.stack([stable_loss.detach(), local_loss.detach(), rel_loss.detach()])
        with torch.no_grad():
            self.running_obj_losses.copy_(
                self.ema_decay * self.running_obj_losses + (1.0 - self.ema_decay) * current.to(self.running_obj_losses)
            )
        score = torch.exp(-current / (self.running_obj_losses + 1e-6)).mean()
        return torch.clamp(score, 0.0, 1.0)
