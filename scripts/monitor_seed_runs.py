import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.phase_gates import evaluate_phase1b_gate, evaluate_phase2_gate, load_metrics_records

PEAK_METRICS = {
    "slot_match": "train/phase1b/slot_match",
    "slot_match_margin": "train/phase1b/slot_match_margin",
    "slot_cycle": "train/phase1b/slot_cycle",
    "slot_identity": "train/phase1b/slot_identity",
    "slot_concentration": "train/phase1b/slot_concentration",
    "object_interface": "train/phase1b/object_interface",
    "m_obj": "train/phase1b/m_obj",
    "phase2_gate_scale": "train/phase2/gate_scale",
    "phase2_match_gate_scale": "train/phase2/match_gate_scale",
    "ret": "train/ret",
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


def _peak_metric(records, key):
    best_value = None
    best_step = None
    for record in records:
        value = record.get(key)
        if not isinstance(value, (int, float)):
            continue
        if best_value is None or float(value) > best_value:
            best_value = float(value)
            best_step = record.get("step")
    if best_value is None:
        return None
    return {"value": best_value, "step": best_step}


def _metric_peaks(records):
    peaks = {}
    for name, key in PEAK_METRICS.items():
        peak = _peak_metric(records, key)
        if peak is not None:
            peaks[name] = peak
    return peaks


def _write_final_summary(root: Path):
    aggregate = {"phase1b_ready": [], "phase2_ready": [], "slot_match_mean": [], "m_obj_mean": [], "ret_last": []}
    peak_values = {name: [] for name in PEAK_METRICS}
    best_peaks = {}
    for seed_dir in sorted(root.glob("seed_*")):
        metrics = seed_dir / "metrics.jsonl"
        if not metrics.exists() or metrics.stat().st_size == 0:
            continue
        records = load_metrics_records(metrics)
        phase1b = evaluate_phase1b_gate(records)
        phase2 = evaluate_phase2_gate(records)
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
            "ret_last": ret_last,
            "peaks": peaks,
        }
        (seed_dir / "final_summary.json").write_text(json.dumps(seed_summary, indent=2, ensure_ascii=False) + "\n")
        aggregate["phase1b_ready"].append(bool(phase1b["ready"]))
        aggregate["phase2_ready"].append(bool(phase2["ready"]))
        aggregate["slot_match_mean"].append(phase1b["summary"]["slot_match_mean"])
        aggregate["m_obj_mean"].append(phase1b["summary"]["m_obj_mean"])
        aggregate["ret_last"].append(ret_last)
        for name, peak in peaks.items():
            peak_values[name].append(peak["value"])
            current_best = best_peaks.get(name)
            if current_best is None or peak["value"] > current_best["value"]:
                best_peaks[name] = {"seed": seed_dir.name, **peak}

    overall = {
        "all_phase1b_ready": all(aggregate["phase1b_ready"]) if aggregate["phase1b_ready"] else False,
        "all_phase2_ready": all(aggregate["phase2_ready"]) if aggregate["phase2_ready"] else False,
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
