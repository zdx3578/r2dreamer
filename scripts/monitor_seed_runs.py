import argparse
import json
import os
import signal
import time
from pathlib import Path


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
            done_path.write_text(f"all seeds finished at {time.strftime('%F %T')}\n")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
