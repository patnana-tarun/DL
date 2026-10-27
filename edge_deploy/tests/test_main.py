"""`python -m edgevision` entry point: --max-frames, --report, --dump-tracks on a synthetic file."""
import json
import signal

import numpy as np
import pytest
from conftest import ONNX_384, needs_ffmpeg

from edgevision.__main__ import main

pytestmark = [needs_ffmpeg, pytest.mark.skipif(not ONNX_384.exists(), reason="ONNX model missing")]


@pytest.fixture
def keep_signal_handlers():
    old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    yield
    for s, h in old.items():
        signal.signal(s, h)


def test_cli_run_report_and_dump(videos, tmp_path, keep_signal_handlers):
    rep, npz = tmp_path / "out" / "report.json", tmp_path / "out" / "tracks.npz"
    rc = main(["--set", f"source.uri={videos['std']}", "--set", "source.width=640", "--set", "source.height=480",
               "--set", "detector.weights=models/yolo26n_person_384x640.onnx", "--set", "detector.providers=[cpu]",
               "--set", "vlm.enabled=false", "--set", "server.enabled=false", "--set", f"store.path={tmp_path / 'e.db'}",
               "--set", "log_level=WARNING", "--max-frames", "12", "--report", str(rep), "--dump-tracks", str(npz)])
    assert rc == 0
    r = json.loads(rep.read_text())
    assert r["frames"] == 12 and r["source"]["kind"] == "file" and r["source"]["fps"] == pytest.approx(30.0)
    assert set(r["latency_ms"]) == {"detect_ms", "track_ms", "analytics_ms", "loop_ms"}   # no UI -> no annotation
    assert r["analytics"]["t"] == pytest.approx(11 / 30, abs=0.01)
    z = np.load(npz)
    assert float(z["fps"]) == pytest.approx(30.0)
    assert z["ts"].tolist() == pytest.approx([i / 30 for i in range(12)], abs=1e-5)   # media timestamps (µs)
    for i in range(12):
        assert z[f"trk/{i}"].shape[1] == 5 and z[f"det/{i}"].shape[1] == 5


def test_cli_rejects_bad_override(keep_signal_handlers):
    with pytest.raises(ValueError):
        main(["--set", "source.fps=10", "--set", "source.stride=2"])
