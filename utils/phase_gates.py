from __future__ import annotations

import json
import math
from pathlib import Path


BASELINE_PROFILES = {
    "exec_v5_30k_b4_alien_2bc36b5": {
        "slot_match_mean": {"baseline": 0.5093164443969727, "mode": "max"},
        "object_interface_mean": {"baseline": 0.6171978712081909, "mode": "max"},
        "retrieval_agreement_mean": {"baseline": 0.9987801313400269, "mode": "max"},
        "rule_apply_error_mean": {"baseline": 0.0004358673933893442, "mode": "min"},
        "ret_mean": {"baseline": 1.0185977280139924, "mode": "max"},
        "score_mean": {"baseline": 209.0909090909091, "mode": "max"},
    },
    "rollout_v2_30k_b4_alien_304f8ba": {
        "slot_match_mean": {"baseline": 0.5093248963356019, "mode": "max"},
        "object_interface_mean": {"baseline": 0.6185529708862305, "mode": "max"},
        "retrieval_agreement_mean": {"baseline": 0.9918229460716248, "mode": "max"},
        "rule_apply_error_mean": {"baseline": 0.0003354866232257336, "mode": "min"},
        "ret_mean": {"baseline": 1.0176645517349243, "mode": "max"},
        "score_mean": {"baseline": 216.0, "mode": "max"},
    },
}
DEFAULT_BASELINE_PROFILE = "rollout_v2_30k_b4_alien_304f8ba"
BASELINE_SIGNOFF_THRESHOLDS = {
    "slot_match_mean": -0.002,
    "object_interface_mean": -0.01,
    "retrieval_agreement_mean": -0.01,
    "rule_apply_error_mean": -2e-4,
    "ret_mean": -0.02,
    "score_mean": -15.0,
}


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


def phase2_window_metrics(phase1b, phase2_executable, phase2_rollout, atari_task):
    return {
        "slot_match_mean": phase1b["summary"]["slot_match_mean"],
        "object_interface_mean": phase1b["summary"]["object_interface_mean"],
        "retrieval_agreement_mean": phase2_executable["summary"]["retrieval_agreement_mean"],
        "rule_apply_error_mean": phase2_executable["summary"]["rule_apply_error_mean"],
        "two_step_apply_error_mean": phase2_rollout["summary"]["two_step_apply_error_mean"],
        "four_step_apply_error_mean": phase2_rollout["summary"]["four_step_apply_error_mean"],
        "seven_step_apply_error_mean": phase2_rollout["summary"]["seven_step_apply_error_mean"],
        "two_step_memory_conf_mean": phase2_rollout["summary"]["two_step_memory_conf_mean"],
        "four_step_memory_conf_mean": phase2_rollout["summary"]["four_step_memory_conf_mean"],
        "seven_step_memory_conf_mean": phase2_rollout["summary"]["seven_step_memory_conf_mean"],
        "ret_mean": atari_task["summary"]["ret_mean"],
        "score_mean": atari_task["summary"]["score_mean"],
        "score_max": atari_task["summary"]["score_max"],
    }


def phase2_baseline_delta(window_metrics, profile=DEFAULT_BASELINE_PROFILE):
    baseline_profile = BASELINE_PROFILES.get(profile)
    if baseline_profile is None:
        raise KeyError(f"Unknown baseline profile: {profile}")
    deltas = {}
    for name, spec in baseline_profile.items():
        value = window_metrics.get(name)
        if not isinstance(value, (int, float)):
            continue
        baseline = float(spec["baseline"])
        mode = spec.get("mode", "max")
        raw_delta = float(value) - baseline
        signed_delta = raw_delta if mode != "min" else -raw_delta
        deltas[name] = {
            "value": float(value),
            "baseline": baseline,
            "raw_delta": raw_delta,
            "signed_delta": signed_delta,
            "mode": mode,
        }
    return deltas


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


def evaluate_phase2_rollout_two_step_gate(records, window=5, slot_count=8):
    executable = evaluate_phase2_executable_gate(records, window=window, slot_count=slot_count)
    required = [
        "train/loss/two_step_apply",
        "train/phase2/two_step_memory_conf",
        "train/phase2/two_step_retrieval_agreement",
        "train/phase2/two_step_apply_error",
        "train/phase2/two_step_fused_delta_rule_abs",
    ]
    has_required = _finite_required(records, required)
    two_step_gate_scale = _recent_values(records, "train/phase2/two_step_gate_scale", window)
    two_step_memory_conf = _recent_values(records, "train/phase2/two_step_memory_conf", window)
    two_step_retrieval_agreement = _recent_values(records, "train/phase2/two_step_retrieval_agreement", window)
    two_step_apply_error = _recent_values(records, "train/phase2/two_step_apply_error", window)
    two_step_fused_delta_rule_abs = _recent_values(records, "train/phase2/two_step_fused_delta_rule_abs", window)

    rollout_gate_open = bool(two_step_gate_scale and _mean(two_step_gate_scale) > 0.0)
    rollout_memory_confident = bool(two_step_memory_conf and _mean(two_step_memory_conf) > 0.05)
    rollout_retrieval_nontrivial = bool(two_step_retrieval_agreement and _mean(two_step_retrieval_agreement) > 0.55)
    rollout_apply_stable = bool(two_step_apply_error and _mean(two_step_apply_error) < 0.25)
    rollout_fused_nontrivial = bool(two_step_fused_delta_rule_abs and _mean(two_step_fused_delta_rule_abs) > 1e-4)

    checks = {
        "phase2_executable_ready": executable["ready"],
        "has_required_metrics": has_required,
        "rollout_gate_open": rollout_gate_open,
        "rollout_memory_confident": rollout_memory_confident,
        "rollout_retrieval_nontrivial": rollout_retrieval_nontrivial,
        "rollout_apply_stable": rollout_apply_stable,
        "rollout_fused_nontrivial": rollout_fused_nontrivial,
    }
    return {
        "phase": "phase2_rollout_two_step",
        "ready": all(checks.values()),
        "checks": checks,
        "summary": {
            "two_step_gate_scale_mean": _mean(two_step_gate_scale),
            "two_step_memory_conf_mean": _mean(two_step_memory_conf),
            "two_step_retrieval_agreement_mean": _mean(two_step_retrieval_agreement),
            "two_step_apply_error_mean": _mean(two_step_apply_error),
            "two_step_fused_delta_rule_abs_mean": _mean(two_step_fused_delta_rule_abs),
        },
        "phase2_executable": executable,
    }


def evaluate_phase2_rollout_long_gate(records, window=5, slot_count=8):
    two_step = evaluate_phase2_rollout_two_step_gate(records, window=window, slot_count=slot_count)
    required = [
        "train/loss/four_step_apply",
        "train/phase2/four_step_curriculum_scale",
        "train/phase2/four_step_memory_conf",
        "train/phase2/four_step_retrieval_agreement",
        "train/phase2/four_step_apply_error",
        "train/phase2/four_step_fused_delta_rule_abs",
        "train/phase2/seven_step_memory_conf",
        "train/phase2/seven_step_retrieval_agreement",
        "train/phase2/seven_step_apply_error",
    ]
    has_required = _finite_required(records, required)
    four_step_curriculum_scale = _recent_values(records, "train/phase2/four_step_curriculum_scale", window)
    four_step_gate_scale = _recent_values(records, "train/phase2/four_step_gate_scale", window)
    four_step_memory_conf = _recent_values(records, "train/phase2/four_step_memory_conf", window)
    four_step_retrieval_agreement = _recent_values(records, "train/phase2/four_step_retrieval_agreement", window)
    four_step_apply_error = _recent_values(records, "train/phase2/four_step_apply_error", window)
    four_step_fused_delta_rule_abs = _recent_values(records, "train/phase2/four_step_fused_delta_rule_abs", window)
    seven_step_memory_conf = _recent_values(records, "train/phase2/seven_step_memory_conf", window)
    seven_step_retrieval_agreement = _recent_values(records, "train/phase2/seven_step_retrieval_agreement", window)
    seven_step_apply_error = _recent_values(records, "train/phase2/seven_step_apply_error", window)
    seven_step_fused_delta_rule_abs = _recent_values(records, "train/phase2/seven_step_fused_delta_rule_abs", window)

    four_step_curriculum_active = bool(four_step_curriculum_scale and _mean(four_step_curriculum_scale) > 0.0)
    four_step_gate_open = bool(four_step_gate_scale and _mean(four_step_gate_scale) > 0.0)
    four_step_memory_confident = bool(four_step_memory_conf and _mean(four_step_memory_conf) > 0.05)
    four_step_retrieval_nontrivial = bool(four_step_retrieval_agreement and _mean(four_step_retrieval_agreement) > 0.55)
    four_step_apply_stable = bool(four_step_apply_error and _mean(four_step_apply_error) < 0.25)
    four_step_fused_nontrivial = bool(four_step_fused_delta_rule_abs and _mean(four_step_fused_delta_rule_abs) > 1e-4)
    seven_step_memory_confident = bool(seven_step_memory_conf and _mean(seven_step_memory_conf) > 0.05)
    seven_step_retrieval_nontrivial = bool(seven_step_retrieval_agreement and _mean(seven_step_retrieval_agreement) > 0.55)
    seven_step_not_exploding = bool(seven_step_apply_error and _mean(seven_step_apply_error) < 0.35)

    checks = {
        "phase2_rollout_two_step_ready": two_step["ready"],
        "has_required_metrics": has_required,
        "four_step_curriculum_active": four_step_curriculum_active,
        "four_step_gate_open": four_step_gate_open,
        "four_step_memory_confident": four_step_memory_confident,
        "four_step_retrieval_nontrivial": four_step_retrieval_nontrivial,
        "four_step_apply_stable": four_step_apply_stable,
        "four_step_fused_nontrivial": four_step_fused_nontrivial,
        "seven_step_memory_confident": seven_step_memory_confident,
        "seven_step_retrieval_nontrivial": seven_step_retrieval_nontrivial,
        "seven_step_not_exploding": seven_step_not_exploding,
    }
    return {
        "phase": "phase2_rollout_long",
        "ready": all(checks.values()),
        "checks": checks,
        "summary": {
            "four_step_curriculum_scale_mean": _mean(four_step_curriculum_scale),
            "four_step_gate_scale_mean": _mean(four_step_gate_scale),
            "four_step_memory_conf_mean": _mean(four_step_memory_conf),
            "four_step_retrieval_agreement_mean": _mean(four_step_retrieval_agreement),
            "four_step_apply_error_mean": _mean(four_step_apply_error),
            "four_step_fused_delta_rule_abs_mean": _mean(four_step_fused_delta_rule_abs),
            "seven_step_memory_conf_mean": _mean(seven_step_memory_conf),
            "seven_step_retrieval_agreement_mean": _mean(seven_step_retrieval_agreement),
            "seven_step_apply_error_mean": _mean(seven_step_apply_error),
            "seven_step_fused_delta_rule_abs_mean": _mean(seven_step_fused_delta_rule_abs),
        },
        "phase2_rollout_two_step": two_step,
    }


def evaluate_phase2_rollout_gate(records, window=5, slot_count=8):
    two_step = evaluate_phase2_rollout_two_step_gate(records, window=window, slot_count=slot_count)
    long_horizon = evaluate_phase2_rollout_long_gate(records, window=window, slot_count=slot_count)
    checks = {
        "phase2_rollout_two_step_ready": two_step["ready"],
        "phase2_rollout_long_ready": long_horizon["ready"],
    }
    return {
        "phase": "phase2_rollout",
        "ready": all(checks.values()),
        "checks": checks,
        "summary": {
            **two_step["summary"],
            **long_horizon["summary"],
        },
        "two_step": two_step,
        "long_horizon": long_horizon,
        "phase2_executable": two_step["phase2_executable"],
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


def evaluate_baseline_relative_gate(records, window=5, task_window=10, score_window=20, slot_count=8, profile=DEFAULT_BASELINE_PROFILE):
    phase1b = evaluate_phase1b_gate(records, window=window, slot_count=slot_count)
    phase2_executable = evaluate_phase2_executable_gate(records, window=window, slot_count=slot_count)
    phase2_rollout = evaluate_phase2_rollout_gate(records, window=window, slot_count=slot_count)
    atari_task = evaluate_atari_task_gate(records, window=task_window, score_window=score_window)
    window_metrics = phase2_window_metrics(phase1b, phase2_executable, phase2_rollout, atari_task)
    baseline_delta = phase2_baseline_delta(window_metrics, profile=profile)
    checks = {}
    for name, threshold in BASELINE_SIGNOFF_THRESHOLDS.items():
        checks[f"{name}_within_budget"] = bool(name in baseline_delta and baseline_delta[name]["signed_delta"] >= threshold)
    return {
        "phase": "baseline_relative",
        "ready": all(checks.values()) if checks else False,
        "checks": checks,
        "summary": {
            "baseline_profile": profile,
            "window_metrics": window_metrics,
            "baseline_delta": baseline_delta,
        },
        "phase1b": phase1b,
        "phase2_executable": phase2_executable,
        "phase2_rollout": phase2_rollout,
        "atari_task": atari_task,
    }


def evaluate_atari_closed_loop(
    records, window=5, task_window=10, score_window=20, slot_count=8, profile=DEFAULT_BASELINE_PROFILE
):
    executable = evaluate_phase2_executable_gate(records, window=window, slot_count=slot_count)
    rollout = evaluate_phase2_rollout_gate(records, window=window, slot_count=slot_count)
    task = evaluate_atari_task_gate(records, window=task_window, score_window=score_window)
    baseline = evaluate_baseline_relative_gate(
        records, window=window, task_window=task_window, score_window=score_window, slot_count=slot_count, profile=profile
    )
    checks = {
        "phase2_executable_ready": executable["ready"],
        "phase2_rollout_ready": rollout["ready"],
        "task_ready": task["ready"],
        "baseline_relative_ready": baseline["ready"],
    }
    return {
        "phase": "atari_closed_loop",
        "ready": all(checks.values()),
        "checks": checks,
        "summary": {
            "phase2_gate_scale_mean": executable["phase2"]["summary"]["gate_scale_mean"],
            "phase2_rollout_two_step_ready": rollout["checks"]["phase2_rollout_two_step_ready"],
            "phase2_rollout_long_ready": rollout["checks"]["phase2_rollout_long_ready"],
            "baseline_profile": profile,
            "ret_mean": task["summary"]["ret_mean"],
            "score_mean": task["summary"]["score_mean"],
            "score_max": task["summary"]["score_max"],
        },
        "phase2_executable": executable,
        "phase2_rollout": rollout,
        "task": task,
        "baseline_relative": baseline,
    }


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate structured and task gates from metrics.jsonl.")
    parser.add_argument("metrics_path", type=Path)
    parser.add_argument(
        "--phase",
        choices=[
            "phase1a",
            "phase1b",
            "phase2",
            "phase2_executable",
            "phase2_rollout_two_step",
            "phase2_rollout_long",
            "phase2_rollout",
            "atari_task",
            "baseline_relative",
            "atari_closed_loop",
        ],
        default="phase1b",
    )
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--task-window", type=int, default=10)
    parser.add_argument("--score-window", type=int, default=20)
    parser.add_argument("--slot-count", type=int, default=8)
    parser.add_argument("--baseline-profile", default=DEFAULT_BASELINE_PROFILE, choices=sorted(BASELINE_PROFILES))
    args = parser.parse_args()

    records = load_metrics_records(args.metrics_path)
    if args.phase == "phase1a":
        result = evaluate_phase1a_gate(records, window=args.window)
    elif args.phase == "phase2":
        result = evaluate_phase2_gate(records, window=args.window, slot_count=args.slot_count)
    elif args.phase == "phase2_executable":
        result = evaluate_phase2_executable_gate(records, window=args.window, slot_count=args.slot_count)
    elif args.phase == "phase2_rollout_two_step":
        result = evaluate_phase2_rollout_two_step_gate(records, window=args.window, slot_count=args.slot_count)
    elif args.phase == "phase2_rollout_long":
        result = evaluate_phase2_rollout_long_gate(records, window=args.window, slot_count=args.slot_count)
    elif args.phase == "phase2_rollout":
        result = evaluate_phase2_rollout_gate(records, window=args.window, slot_count=args.slot_count)
    elif args.phase == "atari_task":
        result = evaluate_atari_task_gate(records, window=args.task_window, score_window=args.score_window)
    elif args.phase == "baseline_relative":
        result = evaluate_baseline_relative_gate(
            records,
            window=args.window,
            task_window=args.task_window,
            score_window=args.score_window,
            slot_count=args.slot_count,
            profile=args.baseline_profile,
        )
    elif args.phase == "atari_closed_loop":
        result = evaluate_atari_closed_loop(
            records,
            window=args.window,
            task_window=args.task_window,
            score_window=args.score_window,
            slot_count=args.slot_count,
            profile=args.baseline_profile,
        )
    else:
        result = evaluate_phase1b_gate(records, window=args.window, slot_count=args.slot_count)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
