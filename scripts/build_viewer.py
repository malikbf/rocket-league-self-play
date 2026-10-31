"""Build RLViser 0.9.3 with the missing Bevy gizmo-renderer feature on macOS.

All tools/source/build products live under ignored artifacts/viewer-build.
Run setup_viewer.py first. This initial Rust build can take several minutes.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a4e524085cdb1601cf9c0af2f815591fb696db54"


def main() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("This build helper supports Apple Silicon macOS only")
    directory = ROOT / "artifacts/viewer-build"
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CARGO_HOME=str(directory / "cargo"), RUSTUP_HOME=str(directory / "rustup"))
    env["PATH"] = str(directory / "cargo/bin") + os.pathsep + env["PATH"]
    cargo = directory / "cargo/bin/cargo"
    if not cargo.exists():
        installer = directory / "rustup-init"
        with urllib.request.urlopen("https://static.rust-lang.org/rustup/dist/aarch64-apple-darwin/rustup-init", timeout=60) as response:
            installer.write_bytes(response.read())
        installer.chmod(0o755)
        subprocess.run([str(installer), "-y", "--no-modify-path", "--profile", "minimal",
                        "--default-toolchain", "1.99.1"], env=env, check=True)
    subprocess.run([str(directory / "cargo/bin/rustup"), "component", "add", "rustfmt"], env=env, check=True)
    source = directory / "source"
    if not source.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", "v0.9.3",
                        "https://github.com/VirxEC/rlviser.git", str(source)], check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != COMMIT:
        raise RuntimeError("Unexpected viewer source revision")
    manifest = source / "Cargo.toml"
    text = manifest.read_text()
    if '"bevy_gizmos_render"' not in text:
        text = text.replace('  "bevy_gizmos",', '  "bevy_gizmos",\n  "bevy_gizmos_render",')
        manifest.write_text(text)
    subprocess.run([str(cargo), "build", "--release", "--no-default-features", "--jobs", "2"], cwd=source, env=env, check=True)
    binary = source / "target/release/rlviser"
    output = ROOT / "artifacts/rlviser"
    target = output / "rlviser-enclosure"
    shutil.copy2(binary, target)
    app_binary = output / "RLViser.app/Contents/MacOS/rlviser"
    shutil.copy2(target, app_binary)
    (output / "enclosure-build.json").write_text(json.dumps({
        "source_commit": COMMIT, "change": "Enable bevy_gizmos_render feature",
        "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    print("Enclosure-capable viewer installed")


if __name__ == "__main__":
    main()
