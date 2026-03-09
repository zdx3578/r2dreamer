import torch
import torch.nn.functional as F


def pairwise_slot_similarity(current, nxt):
    current = F.normalize(current, dim=-1)
    nxt = F.normalize(nxt, dim=-1)
    return torch.einsum("...id,...jd->...ij", current, nxt)


def sinkhorn_normalization(logits, iters=5):
    for _ in range(int(iters)):
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
    return torch.exp(logits)


def soft_slot_alignment(current, nxt, temperature, sinkhorn_iters=5):
    sim = pairwise_slot_similarity(current, nxt)
    logits = sim / float(temperature)
    return sinkhorn_normalization(logits, sinkhorn_iters)


def align_slots(match, nxt):
    return torch.einsum("...ij,...jd->...id", match, nxt)


def match_confidence(match):
    return match.max(dim=-1).values.mean()
