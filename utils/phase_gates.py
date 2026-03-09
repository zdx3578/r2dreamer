from __future__ import annotations

import json
import math
from pathlib import Path


def load_metrics_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _recent_values(records, key, window):
    values = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    if window:
        return values[-int(window) :]
    return values


def _mean(values):
    return sum(values) / len(values) if values else None


def _slope(values):
    if len(values) < 3:
        return None
    xs = list(range(len(values)))
    x_mean = _mean(xs)
    y_mean = _mean(values)
    numer = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return None
    return numer / denom


def _finite_required(records, keys):
    for key in keys:
        values = _recent_values(records, key, window=0)
        if not values:
            return False
    return True


def evaluate_phase1a_gate(records, window=5):
    required = [
        "train/loss/delta_map",
        "train/loss/delta_obj",
        "train/loss/delta_global",
        "train/loss/event",
        "train/phase1a/map_std",
        "train/phase1a/obj_std",
        "train/phase1a/delta_map_abs",
        "train/phase1a/delta_obj_abs",
    ]
    has_required = _finite_required(records, required)
    map_std = _recent_values(records, "train/phase1a/map_std", window)
    obj_std = _recent_values(records, "train/phase1a/obj_std", window)
    delta_map_abs = _recent_values(records, "train/phase1a/delta_map_abs", window)
    delta_obj_abs = _recent_values(records, "train/phase1a/delta_obj_abs", window)
    opt_loss = _recent_values(records, "train/opt/loss", window)

    dual_view_active = bool(map_std and obj_std and _mean(map_std) > 1e-4 and _mean(obj_std) > 1e-4)
    effect_nontrivial = bool(
        delta_map_abs and delta_obj_abs and _mean(delta_map_abs) > 1e-4 and _mean(delta_obj_abs) > 1e-4
    )
    stable_optimization = bool(opt_loss and max(opt_loss) < 1e6)

    checks = {
        "has_required_metrics": has_required,
        "dual_view_active": dual_view_active,
        "effect_nontrivial": effect_nontrivial,
        "stable_optimization": stable_optimization,
    }
    return {
        "phase": "phase1a",
        "ready": all(checks.values()),
        "checks": checks,
        "summary": {
            "map_std_mean": _mean(map_std),
            "obj_std_mean": _mean(obj_std),
            "delta_map_abs_mean": _mean(delta_map_abs),
            "delta_obj_abs_mean": _mean(delta_obj_abs),
            "opt_loss_mean": _mean(opt_loss),
        },
    }


def evaluate_phase1b_gate(records, window=5, slot_count=8):
    phase1a = evaluate_phase1a_gate(records, window=window)
    required = [
        "train/loss/obj_stable",
        "train/loss/obj_local",
        "train/loss/obj_rel",
        "train/phase1b/m_obj",
        "train/phase1b/slot_match",
        "train/phase1b/slot_concentration",
        "train/phase1b/motif_entropy",
    ]
    has_required = _finite_required(records, required)

    m_obj = _recent_values(records, "train/phase1b/m_obj", window)
    slot_match = _recent_values(records, "train/phase1b/slot_match", window)
    slot_concentration = _recent_values(records, "train/phase1b/slot_concentration", window)
    obj_stable = _recent_values(records, "train/loss/obj_stable", window)
    obj_local = _recent_values(records, "train/loss/obj_local", window)
    obj_rel = _recent_values(records, "train/loss/obj_rel", window)
    random_slot_baseline = 1.0 / float(max(1, slot_count))

    slot_matching_better_than_random = bool(slot_match and _mean(slot_match) > random_slot_baseline)
    locality_better_than_random = bool(slot_concentration and _mean(slot_concentration) > random_slot_baseline)
    object_losses_finite = bool(obj_stable and obj_local and obj_rel)
    m_obj_nontrivial = bool(m_obj and 0.05 < _mean(m_obj) < 0.95)
    m_obj_slope = _slope(m_obj)
    trend_available = m_obj_slope is not None
    m_obj_rising = bool(trend_available and m_obj_slope > 0.0)

    checks = {
        "phase1a_ready": phase1a["ready"],
        "has_required_metrics": has_required,
        "slot_matching_better_than_random": slot_matching_better_than_random,
        "locality_better_than_random": locality_better_than_random,
        "object_losses_finite": object_losses_finite,
        "m_obj_nontrivial": m_obj_nontrivial,
        "m_obj_rising": m_obj_rising,
    }
    return {
        "phase": "phase1b",
        "ready": all(checks.values()),
        "checks": checks,
        "summary": {
            "m_obj_mean": _mean(m_obj),
            "m_obj_slope": m_obj_slope,
            "slot_match_mean": _mean(slot_match),
            "slot_concentration_mean": _mean(slot_concentration),
            "random_slot_baseline": random_slot_baseline,
            "trend_available": trend_available,
        },
        "phase1a": phase1a,
    }


def evaluate_phase2_gate(records, window=5, slot_count=8):
    phase1b = evaluate_phase1b_gate(records, window=window, slot_count=slot_count)
    required = [
        "train/loss/op_assign",
        "train/loss/op_proto",
        "train/loss/op_reuse",
        "train/loss/bind_ce",
        "train/loss/bind_consistency",
        "train/loss/sig_scope",
        "train/loss/sig_duration",
        "train/loss/sig_impact",
        "train/loss/rule_update",
        "train/phase2/operator_entropy",
        "train/phase2/operator_usage_entropy",
        "train/phase2/binding_entropy",
        "train/phase2/signature_std",
    ]
    has_required = _finite_required(records, required)
    operator_entropy = _recent_values(records, "train/phase2/operator_entropy", window)
    usage_entropy = _recent_values(records, "train/phase2/operator_usage_entropy", window)
    binding_entropy = _recent_values(records, "train/phase2/binding_entropy", window)
    signature_std = _recent_values(records, "train/phase2/signature_std", window)
    rule_delta_abs = _recent_values(records, "train/phase2/rule_delta_abs", window)
    gate_scale = _recent_values(records, "train/phase2/gate_scale", window)

    operator_not_collapsed = bool(operator_entropy and usage_entropy and _mean(operator_entropy) > 0.1 and _mean(usage_entropy) > 0.1)
    binding_not_noise = bool(binding_entropy and 0.1 < _mean(binding_entropy) < 0.98)
    signature_nontrivial = bool(signature_std and _mean(signature_std) > 1e-3)
    rule_update_active = bool(rule_delta_abs and _mean(rule_delta_abs) > 1e-4)
    gate_open = bool(gate_scale and _mean(gate_scale) > 0.0)

    checks = {
        "phase1b_ready": phase1b["ready"],
        "has_required_metrics": has_required,
        "operator_not_collapsed": operator_not_collapsed,
        "binding_not_noise": binding_not_noise,
        "signature_nontrivial": signature_nontrivial,
        "rule_update_active": rule_update_active,
        "gate_open": gate_open,
    }
    return {
        "phase": "phase2",
        "ready": all(checks.values()),
        "checks": checks,
        "summary": {
            "operator_entropy_mean": _mean(operator_entropy),
            "operator_usage_entropy_mean": _mean(usage_entropy),
            "binding_entropy_mean": _mean(binding_entropy),
            "signature_std_mean": _mean(signature_std),
            "rule_delta_abs_mean": _mean(rule_delta_abs),
            "gate_scale_mean": _mean(gate_scale),
        },
        "phase1b": phase1b,
    }


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Phase 1A/1B gate metrics from metrics.jsonl.")
    parser.add_argument("metrics_path", type=Path)
    parser.add_argument("--phase", choices=["phase1a", "phase1b", "phase2"], default="phase1b")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--slot-count", type=int, default=8)
    args = parser.parse_args()

    records = load_metrics_records(args.metrics_path)
    if args.phase == "phase1a":
        result = evaluate_phase1a_gate(records, window=args.window)
    elif args.phase == "phase2":
        result = evaluate_phase2_gate(records, window=args.window, slot_count=args.slot_count)
    else:
        result = evaluate_phase1b_gate(records, window=args.window, slot_count=args.slot_count)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
