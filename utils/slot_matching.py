import torch
import torch.nn.functional as F


def pairwise_slot_similarity(current, nxt):
    current = F.normalize(current, dim=-1)
    nxt = F.normalize(nxt, dim=-1)
    return torch.einsum("...id,...jd->...ij", current, nxt)


def soft_slot_alignment(current, nxt, temperature):
    sim = pairwise_slot_similarity(current, nxt)
    return torch.softmax(sim / float(temperature), dim=-1)


def align_slots(match, nxt):
    return torch.einsum("...ij,...jd->...id", match, nxt)


def match_confidence(match):
    return match.max(dim=-1).values.mean()
