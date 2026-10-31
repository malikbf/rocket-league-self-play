"""Run bounded PPO iterations with checkpoints, evaluation and a hard deadline."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
from uuid import uuid4

from rocket_league_ai.training.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def stop_process_tree(process: subprocess.Popen) -> None:
    """Stop the isolated run and descendants, including simulator workers."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       check=False, timeout=10, capture_output=True)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.toml")
    parser.add_argument("--resume", type=Path, help="Resume a project checkpoint directory")
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.worker_output is not None:
        from rocket_league_ai.training.ppo import run_smoke
        run_smoke(config, args.worker_output, ROOT, args.resume)
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "logs" / "ppo" / f"{config.experiment.name}-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True)
    # Run the exact config copied here, so edits cannot change a running experiment.
    snapshot = output / "config.toml"
    snapshot.write_bytes(args.config.read_bytes())
    print(f"Artifacts: {output}", flush=True)
    with (output / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--config", str(snapshot),
             "--worker-output", str(output)]
            + (["--resume", str(args.resume.resolve())] if args.resume else []),
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
        )
        try:
            code = process.wait(timeout=config.training.timeout_seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            stop_process_tree(process)
            print("FAIL: interrupted or exceeded deadline; process tree stopped", file=sys.stderr)
            return 1
        finally:
            if os.name != "nt" and process.returncode != 0:
                # Clean up descendants after failures; successful workers already joined.
                stop_process_tree(process)
    print((output / "console.log").read_text(encoding="utf-8"))
    if code == 0 and not (output / "result.json").exists():
        print("FAIL: worker exited without a verified result", file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
