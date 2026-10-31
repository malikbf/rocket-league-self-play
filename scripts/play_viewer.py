"""Play exported states using the isolated RLViser environment (no Torch/RLGym)."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import signal
import time

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--live", action="store_true", help="Read a changing latest-state file")
    parser.add_argument("--seconds", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 600:
        parser.error("seconds must be 1–600")
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    ready_deadline = time.monotonic() + 30
    while args.live and not args.replay.exists():
        if time.monotonic() >= ready_deadline:
            raise RuntimeError("No live simulator state received within 30 seconds")
        time.sleep(0.1)
    data = json.loads(args.replay.read_text(encoding="utf-8"))
    directory = ROOT / "artifacts/rlviser"
    binary = directory / "rlviser"
    if not binary.exists():
        parser.error("Missing artifacts/rlviser/rlviser; see README")
    import RocketSim as rs
    import rlviser_py as viewer

    # The official bridge looks for ./rlviser and the app writes its local settings here.
    os.chdir(directory)
    from arena_overlay import ArenaOverlay
    overlay = ArenaOverlay()
    process = None
    try:
        with (directory / "viewer.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([str(directory / "RLViser.app/Contents/MacOS/rlviser")
                                        if (directory / "RLViser.app").exists() else str(binary)], cwd=directory, stdout=log, stderr=log)
        time.sleep(2)
        if process.poll() is not None:
            raise RuntimeError(f"RLViser exited ({process.returncode}); see {directory / 'viewer.log'}")
        viewer.set_boost_pad_locations(data["boost_locations"])

        def set_physics(target, values):
            target.pos = rs.Vec(*values["position"])
            target.vel = rs.Vec(*values["velocity"])
            target.ang_vel = rs.Vec(*values["angular_velocity"])
            target.rot_mat = rs.RotMat(*values["rotation"])

        deadline = time.monotonic() + args.seconds
        frame_index = 0
        packets = 0
        while time.monotonic() < deadline and process.poll() is None:
            if args.live:
                data = json.loads(args.replay.read_text(encoding="utf-8"))
                frame = data["frame"]
            else:
                frame = data["frames"][frame_index % len(data["frames"])]
            ball = rs.BallState()
            set_physics(ball, frame["ball"])
            cars = []
            for car in frame["cars"]:
                state = rs.CarState()
                set_physics(state, car["physics"])
                state.boost = car["boost"]
                cars.append((car["id"], car["team"], rs.CarConfig(car["hitbox"]), state))
            viewer.render(packets, data["fps"], rs.GameMode.SOCCAR, frame["pads"], ball, cars)
            if packets % 15 == 0:
                overlay.send()
            packets += 1
            speed = viewer.get_game_speed()
            speed = max(0.1, min(8.0, speed)) if math.isfinite(speed) else 1.0
            if not viewer.get_game_paused():
                frame_index += 1
            time.sleep(1 / (data["fps"] * speed))
        if process.poll() not in (None, 0):
            raise RuntimeError("Viewer crashed; inspect artifacts/rlviser/viewer.log")
        print(f"Playback finished: {packets} packets sent")
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        overlay.close()
        if process is not None:
            if process.poll() is None:
                viewer.quit()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)


if __name__ == "__main__":
    raise SystemExit(main())
