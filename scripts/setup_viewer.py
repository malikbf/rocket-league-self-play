"""Install pinned RLViser for Apple Silicon without changing training packages."""
import hashlib
from pathlib import Path
import platform
import plistlib
import shutil
import subprocess
import sys
import urllib.request
import venv

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/VirxEC/rlviser/releases/download/v0.9.3/rlviser-aarch64-apple-darwin"
SHA256 = "821bfb775b14b42fdf5b8da0a7459d3aa2f225ec5b58f2358d90396439adf76e"


def main() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("This installer supports Apple Silicon macOS only")
    directory = ROOT / "artifacts/rlviser"
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / "rlviser"
    if not binary.exists() or hashlib.sha256(binary.read_bytes()).hexdigest() != SHA256:
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != SHA256:
            raise RuntimeError("Official viewer checksum mismatch")
        binary.write_bytes(payload)
    binary.chmod(0o755)
    contents = directory / "RLViser.app/Contents"
    (contents / "MacOS").mkdir(parents=True, exist_ok=True)
    patched = directory / "rlviser-enclosure"
    shutil.copy2(patched if patched.exists() else binary, contents / "MacOS/rlviser")
    with (contents / "Info.plist").open("wb") as stream:
        plistlib.dump({"CFBundleName": "RLViser", "CFBundleDisplayName": "RLViser",
                      "CFBundleIdentifier": "org.rlviser.viewer", "CFBundleExecutable": "rlviser",
                      "CFBundlePackageType": "APPL", "CFBundleVersion": "0.9.3",
                      "NSHighResolutionCapable": True}, stream)
    environment = ROOT / ".venv-viewer"
    if not (environment / "bin/python").exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    subprocess.run([str(environment / "bin/python"), "-m", "pip", "install",
                    "rlviser-py==0.6.14", "rocketsim==2.2.1", "numpy==2.4.6", "flatbuffers==25.12.19"], check=True)
    print("Viewer installed. Run: python scripts/watch_policy.py")


if __name__ == "__main__":
    main()
