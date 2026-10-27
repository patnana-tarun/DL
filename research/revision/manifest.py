"""Reproducibility manifest (review item 8): SHA-256 of every input and artifact, plus the software environment.

    python research/revision/manifest.py        -> MANIFEST.sha256 (sha256sum format) + results/environment.json
    sha256sum -c MANIFEST.sha256                 # verify a copy of the project
"""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402,F401

PATTERNS = ["videos/*.mp4", "models/pretrained/*.pt", "runs/yolo26n/weights/best.pt", "runs/rfdetr_nano_v2/*.pth",
            "runs/revision/*/weights/best.pt", "edge_deploy/models/*.onnx", "edge_deploy/models/vlm/*.gguf",
            "datasets/**/_annotations.coco.json", "datasets/**/teachers.json", "datasets/coco2017_hf/*.parquet",
            "research/annotation/labels/*/*.json"]


def sha(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def main():
    files = sorted({p for pat in PATTERNS for p in Path(".").glob(pat) if p.is_file() and not p.is_symlink()})
    lines = [f"{sha(p)}  {p.as_posix()}" for p in files]
    Path("MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    env = {"python": sys.version, "platform": platform.platform(), "gpu": sh("nvidia-smi --query-gpu=name,driver_version"
           " --format=csv,noheader"), "cuda_torch": sh(f"{sys.executable} -c 'import torch;print(torch.version.cuda)'"),
           "pip_freeze": sh(f"{sys.executable} -m pip freeze").splitlines(),
           "llama_server": sh("edge_deploy/third_party/llama.cpp-vulkan-x64/llama-server --version 2>&1 | tail -2"),
           "seeds": {"training": 0, "bootstrap": 0, "sampling": 0},
           "determinism": "Ultralytics deterministic=False (cuDNN autotune); bootstrap and sampling are seeded; "
                          "VLM at temperature 0 with llama.cpp (greedy) is repeatable on the same build and GPU."}
    Path("results").mkdir(exist_ok=True)
    json.dump(env, open("results/environment.json", "w"), indent=1)
    print(len(files), "files hashed -> MANIFEST.sha256; environment -> results/environment.json")


if __name__ == "__main__":
    main()
