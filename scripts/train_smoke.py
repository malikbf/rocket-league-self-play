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
    parser.add_argument("--live", action="store_true", help="Show real-time training in RLViser on macOS")
    parser.add_argument("--resume", type=Path, help="Resume a project checkpoint directory")
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.worker_output is not None:
        from rocket_league_ai.training.ppo import run_smoke
        run_smoke(config, args.worker_output, ROOT, args.resume, args.live)
        return 0

    if args.live:
        if sys.platform != "darwin" or not (ROOT / ".venv-viewer/bin/python").exists():
            parser.error("Live viewing requires macOS viewer setup: python scripts/setup_viewer.py")
        # Two agents contribute transitions at 120/action_repeat decisions per second.
        estimated = config.training.timesteps * config.training.iterations * config.environment.action_repeat / 240
        if estimated + 30 > config.training.timeout_seconds:
            parser.error("Live run needs a longer deadline or fewer transitions (allow 30s overhead)")
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
            + (["--resume", str(args.resume.resolve())] if args.resume else [])
            + (["--live"] if args.live else []),
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
        )
        viewer_process = None
        viewer_log = None
        try:
            if args.live:
                viewer_log = (output / "live-viewer.log").open("w", encoding="utf-8")
                viewer_process = subprocess.Popen(
                    [str(ROOT / ".venv-viewer/bin/python"), str(ROOT / "scripts/play_viewer.py"),
                     str(output / "live.json"), "--live", "--seconds", str(config.training.timeout_seconds)],
                    stdout=viewer_log, stderr=subprocess.STDOUT, start_new_session=True,
                )
                print("LIVE: both cars use the learning policy; training is paced to real time.", flush=True)
            code = process.wait(timeout=config.training.timeout_seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            stop_process_tree(process)
            print("FAIL: interrupted or exceeded deadline; process tree stopped", file=sys.stderr)
            return 1
        finally:
            if viewer_process is not None:
                if viewer_process.poll() is None:
                    viewer_process.terminate()
                    try:
                        viewer_process.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        stop_process_tree(viewer_process)
                elif viewer_process.returncode != 0:
                    print("Viewer failed; inspect live-viewer.log. Training results are separate.", file=sys.stderr)
            if viewer_log is not None:
                viewer_log.close()
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
