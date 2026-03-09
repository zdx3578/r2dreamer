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


def _last(values):
    return values[-1] if values else None


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
        "train/phase1b/slot_cycle",
        "train/phase1b/slot_identity",
        "train/phase1b/slot_concentration",
        "train/phase1b/motif_entropy",
    ]
    has_required = _finite_required(records, required)

    m_obj = _recent_values(records, "train/phase1b/m_obj", window)
    slot_match = _recent_values(records, "train/phase1b/slot_match", window)
    slot_match_random = _recent_values(records, "train/phase1b/slot_match_random", window)
    slot_match_margin = _recent_values(records, "train/phase1b/slot_match_margin", window)
    slot_cycle = _recent_values(records, "train/phase1b/slot_cycle", window)
    slot_identity = _recent_values(records, "train/phase1b/slot_identity", window)
    slot_concentration = _recent_values(records, "train/phase1b/slot_concentration", window)
    object_interface = _recent_values(records, "train/phase1b/object_interface", window)
    obj_stable = _recent_values(records, "train/loss/obj_stable", window)
    obj_local = _recent_values(records, "train/loss/obj_local", window)
    obj_rel = _recent_values(records, "train/loss/obj_rel", window)
    uniform_slot_baseline = 1.0 / float(max(1, slot_count))
    random_slot_baseline = _mean(slot_match_random) if slot_match_random else uniform_slot_baseline
    if not slot_match_margin and slot_match:
        slot_match_margin = [value - random_slot_baseline for value in slot_match]
    if not object_interface and slot_match_margin and slot_cycle and slot_identity and slot_concentration:
        denom = max(1e-6, 1.0 - random_slot_baseline)
        object_interface = [
            max(0.0, margin / denom) + cycle + identity + concentration
            for margin, cycle, identity, concentration in zip(
                slot_match_margin, slot_cycle, slot_identity, slot_concentration
            )
        ]
        object_interface = [value / 4.0 for value in object_interface]

    slot_match_threshold = max(0.25, uniform_slot_baseline + 0.05)
    slot_cycle_threshold = 0.5
    slot_identity_threshold = 0.2
    object_interface_threshold = 0.25

    # The shuffled-match baseline is informative for monitoring, but in Atari-style batches it
    # stays high even on healthy runs. Readiness should track absolute structure quality instead.
    slot_matching_nontrivial = bool(slot_match and _mean(slot_match) > slot_match_threshold)
    locality_better_than_uniform = bool(slot_concentration and _mean(slot_concentration) > uniform_slot_baseline)
    slot_cycle_healthy = bool(slot_cycle and _mean(slot_cycle) > slot_cycle_threshold)
    slot_identity_healthy = bool(slot_identity and _mean(slot_identity) > slot_identity_threshold)
    object_interface_healthy = bool(object_interface and _mean(object_interface) > object_interface_threshold)
    object_losses_finite = bool(obj_stable and obj_local and obj_rel)
    m_obj_nontrivial = bool(m_obj and 0.05 < _mean(m_obj) < 0.95)

    checks = {
        "phase1a_ready": phase1a["ready"],
        "has_required_metrics": has_required,
        "slot_matching_nontrivial": slot_matching_nontrivial,
        "locality_better_than_uniform": locality_better_than_uniform,
        "slot_cycle_healthy": slot_cycle_healthy,
        "slot_identity_healthy": slot_identity_healthy,
        "object_interface_healthy": object_interface_healthy,
        "object_losses_finite": object_losses_finite,
        "m_obj_nontrivial": m_obj_nontrivial,
    }
    return {
        "phase": "phase1b",
        "ready": all(checks.values()),
        "checks": checks,
        "summary": {
            "m_obj_mean": _mean(m_obj),
            "slot_match_mean": _mean(slot_match),
            "slot_match_random_mean": _mean(slot_match_random) if slot_match_random else None,
            "slot_match_margin_mean": _mean(slot_match_margin),
            "slot_cycle_mean": _mean(slot_cycle),
            "slot_identity_mean": _mean(slot_identity),
            "slot_concentration_mean": _mean(slot_concentration),
            "object_interface_mean": _mean(object_interface),
            "uniform_slot_baseline": uniform_slot_baseline,
            "random_slot_baseline": random_slot_baseline,
            "slot_match_threshold": slot_match_threshold,
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
    match_gate_scale = _recent_values(records, "train/phase2/match_gate_scale", window)

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
            "match_gate_scale_mean": _mean(match_gate_scale),
        },
        "phase1b": phase1b,
    }


def evaluate_phase2_executable_gate(records, window=5, slot_count=8):
    phase2 = evaluate_phase2_gate(records, window=window, slot_count=slot_count)
    required = [
        "train/loss/rule_apply",
        "train/phase2/operator_top1_conf",
        "train/phase2/binding_top1_conf",
        "train/phase2/memory_conf",
        "train/phase2/retrieval_agreement",
        "train/phase2/rule_apply_error",
        "train/phase2/rule_memory_usage",
        "train/phase2/rule_memory_write_rate",
    ]
    has_required = _finite_required(records, required)
    operator_conf = _recent_values(records, "train/phase2/operator_top1_conf", window)
    binding_conf = _recent_values(records, "train/phase2/binding_top1_conf", window)
    memory_conf = _recent_values(records, "train/phase2/memory_conf", window)
    retrieval_agreement = _recent_values(records, "train/phase2/retrieval_agreement", window)
    rule_apply_error = _recent_values(records, "train/phase2/rule_apply_error", window)
    rule_memory_usage = _recent_values(records, "train/phase2/rule_memory_usage", window)
    rule_memory_entropy = _recent_values(records, "train/phase2/rule_memory_entropy", window)
    rule_memory_write_rate = _recent_values(records, "train/phase2/rule_memory_write_rate", window)
    fused_delta_rule_abs = _recent_values(records, "train/phase2/fused_delta_rule_abs", window)

    operator_confident = bool(operator_conf and _mean(operator_conf) > 0.25)
    binding_confident = bool(binding_conf and _mean(binding_conf) > 0.22)
    memory_populated = bool(rule_memory_usage and _mean(rule_memory_usage) > 0.0)
    memory_writing = bool(rule_memory_write_rate and _mean(rule_memory_write_rate) > 0.0)
    retrieval_confident = bool(memory_conf and _mean(memory_conf) > 0.05)
    retrieval_agreement_nontrivial = bool(retrieval_agreement and _mean(retrieval_agreement) > 0.55)
    rule_apply_stable = bool(rule_apply_error and _mean(rule_apply_error) < 0.25)
    fused_rule_nontrivial = bool(fused_delta_rule_abs and _mean(fused_delta_rule_abs) > 1e-4)

    checks = {
        "phase2_ready": phase2["ready"],
        "has_required_metrics": has_required,
        "operator_confident": operator_confident,
        "binding_confident": binding_confident,
        "memory_populated": memory_populated,
        "memory_writing": memory_writing or memory_populated,
        "retrieval_confident": retrieval_confident,
        "retrieval_agreement_nontrivial": retrieval_agreement_nontrivial,
        "rule_apply_stable": rule_apply_stable,
        "fused_rule_nontrivial": fused_rule_nontrivial,
    }
    return {
        "phase": "phase2_executable",
        "ready": all(checks.values()),
        "checks": checks,
        "summary": {
            "operator_top1_conf_mean": _mean(operator_conf),
            "binding_top1_conf_mean": _mean(binding_conf),
            "memory_conf_mean": _mean(memory_conf),
            "retrieval_agreement_mean": _mean(retrieval_agreement),
            "rule_apply_error_mean": _mean(rule_apply_error),
            "rule_memory_usage_mean": _mean(rule_memory_usage),
            "rule_memory_entropy_mean": _mean(rule_memory_entropy),
            "rule_memory_write_rate_mean": _mean(rule_memory_write_rate),
            "fused_delta_rule_abs_mean": _mean(fused_delta_rule_abs),
        },
        "phase2": phase2,
    }


def evaluate_atari_task_gate(records, window=10, score_window=20):
    required = ["train/ret"]
    has_required = _finite_required(records, required)
    train_ret = _recent_values(records, "train/ret", window)
    opt_loss = _recent_values(records, "train/opt/loss", window)
    scores = _recent_values(records, "episode/score", score_window)
    lengths = _recent_values(records, "episode/length", score_window)

    ret_mean = _mean(train_ret)
    ret_slope = _slope(train_ret)
    score_mean = _mean(scores)
    score_max = max(scores) if scores else None

    returns_nontrivial = bool(train_ret and ret_mean is not None and ret_mean > 0.1)
    returns_stable = bool(opt_loss and max(opt_loss) < 1e6)
    learning_progress = bool(
        train_ret
        and (
            (ret_slope is not None and ret_slope > -0.02)
            or (ret_mean is not None and ret_mean > 0.3)
        )
    )
    score_nontrivial = bool(scores and score_max is not None and score_max > 0.0)
    episode_coverage = bool(scores and lengths)

    checks = {
        "has_required_metrics": has_required,
        "returns_nontrivial": returns_nontrivial,
        "returns_stable": returns_stable,
        "learning_progress": learning_progress,
        "score_nontrivial": score_nontrivial,
        "episode_coverage": episode_coverage,
    }
    return {
        "phase": "atari_task",
        "ready": all(checks.values()),
        "checks": checks,
        "summary": {
            "ret_mean": ret_mean,
            "ret_last": _last(train_ret),
            "ret_slope": ret_slope,
            "score_mean": score_mean,
            "score_last": _last(scores),
            "score_max": score_max,
            "score_count": len(scores),
            "episode_length_mean": _mean(lengths),
        },
    }


def evaluate_atari_closed_loop(records, window=5, task_window=10, score_window=20, slot_count=8):
    phase2 = evaluate_phase2_gate(records, window=window, slot_count=slot_count)
    task = evaluate_atari_task_gate(records, window=task_window, score_window=score_window)
    checks = {
        "phase2_ready": phase2["ready"],
        "task_ready": task["ready"],
    }
    return {
        "phase": "atari_closed_loop",
        "ready": all(checks.values()),
        "checks": checks,
        "summary": {
            "phase2_gate_scale_mean": phase2["summary"]["gate_scale_mean"],
            "ret_mean": task["summary"]["ret_mean"],
            "score_mean": task["summary"]["score_mean"],
            "score_max": task["summary"]["score_max"],
        },
        "phase2": phase2,
        "task": task,
    }


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate structured and task gates from metrics.jsonl.")
    parser.add_argument("metrics_path", type=Path)
    parser.add_argument(
        "--phase",
        choices=["phase1a", "phase1b", "phase2", "phase2_executable", "atari_task", "atari_closed_loop"],
        default="phase1b",
    )
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--task-window", type=int, default=10)
    parser.add_argument("--score-window", type=int, default=20)
    parser.add_argument("--slot-count", type=int, default=8)
    args = parser.parse_args()

    records = load_metrics_records(args.metrics_path)
    if args.phase == "phase1a":
        result = evaluate_phase1a_gate(records, window=args.window)
    elif args.phase == "phase2":
        result = evaluate_phase2_gate(records, window=args.window, slot_count=args.slot_count)
    elif args.phase == "phase2_executable":
        result = evaluate_phase2_executable_gate(records, window=args.window, slot_count=args.slot_count)
    elif args.phase == "atari_task":
        result = evaluate_atari_task_gate(records, window=args.task_window, score_window=args.score_window)
    elif args.phase == "atari_closed_loop":
        result = evaluate_atari_closed_loop(
            records,
            window=args.window,
            task_window=args.task_window,
            score_window=args.score_window,
            slot_count=args.slot_count,
        )
    else:
        result = evaluate_phase1b_gate(records, window=args.window, slot_count=args.slot_count)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
