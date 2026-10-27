"""python -m edgevision --config configs/desktop.yaml [--set source.uri=... --set vlm.enabled=false ...]

Options for benchmarks / evaluation:
  --max-frames N        stop after N frames
  --report out.json     write a run report (latency per stage, throughput, final analytics)
  --dump-tracks out.npz save per-frame tracks and raw detections (for offline scoring)
"""
import argparse
import json
import logging
import signal
import sys
from pathlib import Path

import numpy as np

from . import config
from .pipeline import Pipeline


def main(argv=None):
    ap = argparse.ArgumentParser(prog="edgevision", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(config.BASE / "configs" / "desktop.yaml"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--report", default="")
    ap.add_argument("--dump-tracks", default="")
    a = ap.parse_args(argv)
    cfg = config.load(a.config, a.set)
    logging.basicConfig(level=cfg["log_level"], format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    p = Pipeline(cfg, record_tracks=bool(a.dump_tracks))

    def _sigint(*_):
        logging.getLogger("edgevision").info("stopping...")
        p.running = False
        p.source.stop()
    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)
    p.run(max_frames=a.max_frames)
    rep = p.report()
    logging.getLogger("edgevision").info("run report: %s", json.dumps(rep))
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(rep, indent=1))
    if a.dump_tracks:
        Path(a.dump_tracks).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.dump_tracks, fps=p.source.fps,
                            **{f"trk/{i}": np.c_[tb, ids] for i, _, tb, ids, _, _ in p.record},
                            **{f"det/{i}": np.c_[b, s] for i, _, _, _, b, s in p.record},
                            ts=np.array([r[1] for r in p.record]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
