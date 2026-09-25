import json


def test_runlog_lifecycle(tmp_path):
    from brain.runlog import RunLog

    log = RunLog("unit-test", config={"lr": 1e-4, "data": "x"}, root=tmp_path)
    assert log.dir.is_dir()
    cfg = json.loads((log.dir / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["lr"] == 1e-4
    assert "git=" in (log.dir / "env.txt").read_text(encoding="utf-8")

    log.metric(10, loss=0.5)
    log.metric(20, loss=0.3, kl=0.1)
    log.finish({"status": "ok", "final_loss": 0.3})

    lines = (log.dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(l)["step"] for l in lines] == [10, 20]
    assert json.loads(lines[1])["kl"] == 0.1

    idx = (tmp_path / "RUNS.md").read_text(encoding="utf-8")
    assert log.run_id in idx
    assert "final_loss=0.3" in idx


def test_runlog_rejects_path_unsafe_tag(tmp_path):
    import pytest

    from brain.runlog import RunLog

    for bad in ("a/b", "a\\b", "..", "x..y", ""):
        with pytest.raises(ValueError, match="tag"):
            RunLog(bad, config={}, root=tmp_path)
    # 正常 tag 不误伤
    RunLog("rlcd-ok_1", config={}, root=tmp_path)
