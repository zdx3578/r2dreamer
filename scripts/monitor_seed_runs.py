import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.phase_gates import (
    evaluate_atari_closed_loop,
    evaluate_atari_task_gate,
    evaluate_phase1b_gate,
    evaluate_phase2_executable_gate,
    evaluate_phase2_gate,
    evaluate_phase2_rollout_gate,
    load_metrics_records,
)

PEAK_METRICS = {
    "slot_match": {"key": "train/phase1b/slot_match", "mode": "max"},
    "slot_match_random": {"key": "train/phase1b/slot_match_random", "mode": "max"},
    "slot_match_margin": {"key": "train/phase1b/slot_match_margin", "mode": "max"},
    "slot_match_margin_score": {"key": "train/phase1b/slot_match_margin_score", "mode": "max"},
    "slot_cycle": {"key": "train/phase1b/slot_cycle", "mode": "max"},
    "slot_identity": {"key": "train/phase1b/slot_identity", "mode": "max"},
    "slot_concentration": {"key": "train/phase1b/slot_concentration", "mode": "max"},
    "object_interface": {"key": "train/phase1b/object_interface", "mode": "max"},
    "m_obj": {"key": "train/phase1b/m_obj", "mode": "max"},
    "phase2_gate_scale": {"key": "train/phase2/gate_scale", "mode": "max"},
    "phase2_match_gate_scale": {"key": "train/phase2/match_gate_scale", "mode": "max"},
    "phase2_operator_top1_conf": {"key": "train/phase2/operator_top1_conf", "mode": "max"},
    "phase2_operator_margin": {"key": "train/phase2/operator_margin", "mode": "max"},
    "phase2_binding_top1_conf": {"key": "train/phase2/binding_top1_conf", "mode": "max"},
    "phase2_memory_conf": {"key": "train/phase2/memory_conf", "mode": "max"},
    "phase2_retrieval_peak": {"key": "train/phase2/retrieval_peak", "mode": "max"},
    "phase2_retrieval_agreement": {"key": "train/phase2/retrieval_agreement", "mode": "max"},
    "phase2_rule_memory_usage": {"key": "train/phase2/rule_memory_usage", "mode": "max"},
    "phase2_rule_memory_entropy": {"key": "train/phase2/rule_memory_entropy", "mode": "max"},
    "phase2_rule_memory_write_rate": {"key": "train/phase2/rule_memory_write_rate", "mode": "max"},
    "phase2_memory_read_error": {"key": "train/phase2/memory_read_error", "mode": "min"},
    "phase2_memory_agreement_error": {"key": "train/phase2/memory_agreement_error", "mode": "min"},
    "phase2_memory_agreement_coverage": {"key": "train/phase2/memory_agreement_coverage", "mode": "max"},
    "phase2_rule_apply_error": {"key": "train/phase2/rule_apply_error", "mode": "min"},
    "phase2_two_step_memory_conf": {"key": "train/phase2/two_step_memory_conf", "mode": "max"},
    "phase2_two_step_retrieval_agreement": {"key": "train/phase2/two_step_retrieval_agreement", "mode": "max"},
    "phase2_two_step_apply_error": {"key": "train/phase2/two_step_apply_error", "mode": "min"},
    "ret": {"key": "train/ret", "mode": "max"},
}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _seed_status(seed_dir: Path) -> str:
    metrics = seed_dir / "metrics.jsonl"
    run_out = seed_dir / "run.out"
    if metrics.exists() and metrics.stat().st_size > 0:
        rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
        last = rows[-1]
        step = last.get("step", "na")
        updates = last.get("train/opt/updates", "na")
        slot_match = last.get("train/phase1b/slot_match", "na")
        ret = last.get("train/ret", "na")
        return f"step={step} updates={updates} slot_match={slot_match} ret={ret}"
    if run_out.exists():
        return "started, waiting for metrics"
    return "not started"


def _safe_mean(values):
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def _peak_metric(records, key, mode="max"):
    best_value = None
    best_step = None
    for record in records:
        value = record.get(key)
        if not isinstance(value, (int, float)):
            continue
        value = float(value)
        if best_value is None:
            best_value = value
            best_step = record.get("step")
            continue
        if mode == "min" and value < best_value:
            best_value = value
            best_step = record.get("step")
        elif mode != "min" and value > best_value:
            best_value = float(value)
            best_step = record.get("step")
    if best_value is None:
        return None
    return {"value": best_value, "step": best_step, "mode": mode}


def _metric_peaks(records):
    peaks = {}
    for name, spec in PEAK_METRICS.items():
        peak = _peak_metric(records, spec["key"], mode=spec.get("mode", "max"))
        if peak is not None:
            peaks[name] = peak
    return peaks


def _write_final_summary(root: Path):
    aggregate = {
        "phase1b_ready": [],
        "phase2_ready": [],
        "phase2_executable_ready": [],
        "phase2_rollout_ready": [],
        "atari_task_ready": [],
        "atari_closed_loop_ready": [],
        "slot_match_mean": [],
        "m_obj_mean": [],
        "ret_last": [],
    }
    peak_values = {name: [] for name in PEAK_METRICS}
    best_peaks = {}
    for seed_dir in sorted(root.glob("seed_*")):
        metrics = seed_dir / "metrics.jsonl"
        if not metrics.exists() or metrics.stat().st_size == 0:
            continue
        records = load_metrics_records(metrics)
        phase1b = evaluate_phase1b_gate(records)
        phase2 = evaluate_phase2_gate(records)
        phase2_executable = evaluate_phase2_executable_gate(records)
        phase2_rollout = evaluate_phase2_rollout_gate(records)
        atari_task = evaluate_atari_task_gate(records)
        atari_closed_loop = evaluate_atari_closed_loop(records)
        rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
        ret_last = None
        for row in reversed(rows):
            if "train/ret" in row:
                ret_last = row["train/ret"]
                break
        peaks = _metric_peaks(rows)
        seed_summary = {
            "seed": seed_dir.name,
            "phase1b": phase1b,
            "phase2": phase2,
            "phase2_executable": phase2_executable,
            "phase2_rollout": phase2_rollout,
            "atari_task": atari_task,
            "atari_closed_loop": atari_closed_loop,
            "ret_last": ret_last,
            "peaks": peaks,
        }
        (seed_dir / "final_summary.json").write_text(json.dumps(seed_summary, indent=2, ensure_ascii=False) + "\n")
        aggregate["phase1b_ready"].append(bool(phase1b["ready"]))
        aggregate["phase2_ready"].append(bool(phase2["ready"]))
        aggregate["phase2_executable_ready"].append(bool(phase2_executable["ready"]))
        aggregate["phase2_rollout_ready"].append(bool(phase2_rollout["ready"]))
        aggregate["atari_task_ready"].append(bool(atari_task["ready"]))
        aggregate["atari_closed_loop_ready"].append(bool(atari_closed_loop["ready"]))
        aggregate["slot_match_mean"].append(phase1b["summary"]["slot_match_mean"])
        aggregate["m_obj_mean"].append(phase1b["summary"]["m_obj_mean"])
        aggregate["ret_last"].append(ret_last)
        for name, peak in peaks.items():
            peak_values[name].append(peak["value"])
            current_best = best_peaks.get(name)
            if current_best is None:
                best_peaks[name] = {"seed": seed_dir.name, **peak}
                continue
            mode = peak.get("mode", "max")
            better = peak["value"] < current_best["value"] if mode == "min" else peak["value"] > current_best["value"]
            if better:
                best_peaks[name] = {"seed": seed_dir.name, **peak}

    overall = {
        "all_phase1b_ready": all(aggregate["phase1b_ready"]) if aggregate["phase1b_ready"] else False,
        "all_phase2_ready": all(aggregate["phase2_ready"]) if aggregate["phase2_ready"] else False,
        "all_phase2_executable_ready": all(aggregate["phase2_executable_ready"]) if aggregate["phase2_executable_ready"] else False,
        "all_phase2_rollout_ready": all(aggregate["phase2_rollout_ready"]) if aggregate["phase2_rollout_ready"] else False,
        "all_atari_task_ready": all(aggregate["atari_task_ready"]) if aggregate["atari_task_ready"] else False,
        "all_atari_closed_loop_ready": all(aggregate["atari_closed_loop_ready"]) if aggregate["atari_closed_loop_ready"] else False,
        "slot_match_mean_avg": _safe_mean(aggregate["slot_match_mean"]),
        "m_obj_mean_avg": _safe_mean(aggregate["m_obj_mean"]),
        "ret_last_avg": _safe_mean(aggregate["ret_last"]),
        "peak_value_avg": {name: _safe_mean(values) for name, values in peak_values.items() if values},
        "best_peaks": best_peaks,
    }
    (root / "final_summary.json").write_text(json.dumps(overall, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--pids", nargs="*", type=int, default=[])
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    root = Path(args.root)
    status_path = root / "monitor_status.txt"
    log_path = root / "monitor.log"
    done_path = root / "monitor_done.txt"

    if done_path.exists():
        done_path.unlink()

    while True:
        lines = [time.strftime("%F %T")]
        for seed_dir in sorted(root.glob("seed_*")):
            lines.append(f"{seed_dir.name}: {_seed_status(seed_dir)}")
        text = "\n".join(lines) + "\n"
        status_path.write_text(text)
        log_path.write_text(text)

        if args.pids and not any(_pid_alive(pid) for pid in args.pids):
            _write_final_summary(root)
            done_path.write_text(f"all seeds finished at {time.strftime('%F %T')}\n")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
