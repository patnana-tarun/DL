import copy
from pathlib import Path

import pytest

from edgevision import config


def test_defaults_load_and_are_not_shared():
    cfg = config.load()
    assert cfg == config.DEFAULTS
    assert cfg is not config.DEFAULTS
    cfg["source"]["width"] = 1
    cfg["alerts"].append({"name": "x"})
    assert config.DEFAULTS["source"]["width"] == 1280            # deep copy: defaults untouched
    assert len(config.DEFAULTS["alerts"]) == 1
    assert config.load()["source"]["width"] == 1280


def test_desktop_yaml_merges_with_defaults():
    cfg = config.load(config.BASE / "configs" / "desktop.yaml")
    assert cfg["source"]["uri"] == "rtsp://127.0.0.1:8554/cam1"
    assert cfg["vlm"]["observe_every_s"] == 60                    # set in the YAML
    assert cfg["vlm"]["max_tokens"] == config.DEFAULTS["vlm"]["max_tokens"]   # not in the YAML -> default
    assert cfg["tracker"] == config.DEFAULTS["tracker"]
    assert cfg["detector"]["providers"] == ["tensorrt", "cuda", "cpu"]


def test_yaml_deep_merge(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("source:\n  uri: videos/x.mp4\n  fps: 10\nanalytics:\n  zones:\n    door: [[0,0],[1,0],[1,1]]\n"
                 "alerts: []\nserver:\n  port: 9001\n")
    cfg = config.load(p)
    assert cfg["source"]["uri"] == "videos/x.mp4" and cfg["source"]["fps"] == 10
    assert cfg["source"]["width"] == 1280 and cfg["source"]["rtsp_transport"] == "tcp"   # siblings kept
    assert cfg["analytics"]["zones"] == {"door": [[0, 0], [1, 0], [1, 1]]}
    assert cfg["analytics"]["line"] == [[0.5, 0.0], [0.5, 1.0]]
    assert cfg["alerts"] == []                                    # lists are replaced, not merged
    assert cfg["server"] == {**config.DEFAULTS["server"], "port": 9001}


def test_empty_yaml(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert config.load(p) == config.DEFAULTS


@pytest.mark.parametrize("override, path, expected", [
    ("source.fps=10", ("source", "fps"), 10),
    ("detector.conf=0.3", ("detector", "conf"), 0.3),
    ("vlm.enabled=false", ("vlm", "enabled"), False),
    ("vlm.enabled=true", ("vlm", "enabled"), True),
    ("source.realtime=null", ("source", "realtime"), None),
    ("source.uri=videos/aa1.mp4", ("source", "uri"), "videos/aa1.mp4"),
    ("source.uri=rtsp://10.0.0.5:8554/cam1?x=1", ("source", "uri"), "rtsp://10.0.0.5:8554/cam1?x=1"),
    ("server.host=0.0.0.0", ("server", "host"), "0.0.0.0"),
    ("analytics.line=[[0, 0.5], [1, 0.5]]", ("analytics", "line"), [[0, 0.5], [1, 0.5]]),
    ("detector.providers=[cpu]", ("detector", "providers"), ["cpu"]),
    ("analytics.zones={door: [[0,0],[1,0],[1,1]]}", ("analytics", "zones"), {"door": [[0, 0], [1, 0], [1, 1]]}),
    ("output.uri=", ("output", "uri"), None),
    ("brand.new.key=7", ("brand", "new", "key"), 7),
])
def test_overrides_parse_types(override, path, expected):
    cfg = config.load(overrides=[override])
    node = cfg
    for k in path:
        node = node[k]
    assert node == expected
    assert type(node) is type(expected)


def test_overrides_apply_after_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("source: {fps: 10}\n")
    cfg = config.load(p, ["source.fps=0", "source.stride=3"])
    assert cfg["source"]["fps"] == 0 and cfg["source"]["stride"] == 3


def test_override_without_equals_rejected():
    with pytest.raises(ValueError, match="key=value"):
        config.load(overrides=["source.fps"])


@pytest.mark.parametrize("overrides, match", [
    (["source.fps=10", "source.stride=2"], "either source.fps or source.stride"),
    (["detector.backend=tensorflow"], "unknown detector.backend"),
    (["source.width=0"], "positive"),
    (["source.height=-5"], "positive"),
    (["analytics.line=[[0.3, 0.3], [0.3, 0.3]]"], "two different points"),
    (["alerts=[{name: a, metric: occupancy, op: '>'}]"], "misses"),
    (["alerts=[{metric: occupancy, op: '>', value: 3}]"], "misses"),
])
def test_validation_errors(overrides, match):
    with pytest.raises(ValueError, match=match):
        config.load(overrides=overrides)


def test_validate_accepts_defaults_and_stride_alone():
    config.validate(copy.deepcopy(config.DEFAULTS))
    cfg = config.load(overrides=["source.stride=2"])
    assert cfg["source"]["stride"] == 2


def test_resolve_relative_absolute_url(tmp_path):
    assert config.resolve("models/a.onnx") == str(config.BASE / "models" / "a.onnx")
    assert Path(config.resolve("models/a.onnx")).is_absolute()
    absolute = str(tmp_path / "x.db")
    assert config.resolve(absolute) == absolute
    for url in ("rtsp://127.0.0.1:8554/cam1", "udp://127.0.0.1:5000", "http://h/v1", "csi://0"):
        assert config.resolve(url) == url
    assert config.resolve("") == "" and config.resolve(None) is None
    assert config.resolve("~/x.onnx") == str(Path.home() / "x.onnx")
    assert config.resolve(Path("data/e.db")) == str(config.BASE / "data" / "e.db")


def test_base_is_edge_deploy():
    assert (config.BASE / "edgevision" / "config.py").exists()
