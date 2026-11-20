import json

from whispertype.config import Config


def test_defaults():
    c = Config()
    assert c.model == "small"
    assert c.hotkey == "f8"
    assert c.mode == "hold"


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "config.json"
    c = Config(model="base", hotkey="f9", mode="toggle", device=2)
    c.save(p)
    d = Config.load(p)
    assert d.model == "base"
    assert d.hotkey == "f9"
    assert d.mode == "toggle"
    assert d.device == 2


def test_unknown_keys_ignored(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"model": "tiny", "nonsense": 1}), encoding="utf-8")
    c = Config.load(p)
    assert c.model == "tiny"
    assert not hasattr(c, "nonsense")


def test_merged_ignores_none():
    c = Config()
    d = c.merged(model="base", device=None, hotkey=None)
    assert d.model == "base"
    assert d.device is None


def test_bad_json_falls_back(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{not json", encoding="utf-8")
    assert Config.load(p).model == "small"