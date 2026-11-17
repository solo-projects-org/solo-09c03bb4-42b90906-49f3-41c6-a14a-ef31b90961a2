"""关键流程自测：上传、锚点、校时、跨日原子性、并发交错、重启恢复、失败不留半版。"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

CSV_HEADER = "segment,local_seconds,temperature\n"


def make_client(tmp_path, monkeypatch, worker="0", delay="0", max_points=None) -> TestClient:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("WORKER_ENABLED", worker)
    monkeypatch.setenv("RECALC_DELAY", delay)
    if max_points is not None:
        monkeypatch.setenv("MAX_POINTS", str(max_points))
    from app.main import create_app
    return TestClient(create_app())


@pytest.fixture()
def client(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as c:
        yield c


def upload(client: TestClient, rows, header: str | None = CSV_HEADER, filename="data.csv"):
    content = (header or "") + "".join(f"{s},{sec},{t}\n" for s, sec, t in rows)
    return client.post("/api/uploads", files={"file": (filename, content, "text/csv")})


def set_anchors(client: TestClient, seg, l1, u1, l2, u2, version=None):
    payload = {"segment": seg, "local1": l1, "utc1": u1, "local2": l2, "utc2": u2}
    if version is not None:
        payload["version"] = version
    return client.post("/api/anchors", json=payload)


def wait_idle(client: TestClient, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        calc = client.get("/api/status").json()["calc"]
        if calc["status"] != "running" and not calc["dirty"]:
            return calc
        time.sleep(0.05)
    raise AssertionError("重算未在超时内收敛")


# ---------------------------------------------------------------- 上传

def test_upload_happy(client):
    r = upload(client, [(1, 100, 4.5), (1, 101, 4.6), (2, 5, -2.0)])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted"] == 3 and body["batch_id"] == 1
    st = client.get("/api/status").json()
    assert st["points"] == {"accepted": 3, "max": 2000}
    assert st["batches"][0]["rows_accepted"] == 3


def test_duplicate_within_file_rejected(client):
    r = upload(client, [(1, 100, 1.0), (1, 100, 2.0)])
    assert r.status_code == 400
    assert r.json()["code"] == "CSV_INVALID"
    assert any("重复" in e["message"] for e in r.json()["errors"])
    assert client.get("/api/status").json()["points"]["accepted"] == 0


def test_duplicate_against_db_rejected(client):
    assert upload(client, [(1, 100, 1.0)]).status_code == 200
    r = upload(client, [(1, 100, 9.9)])
    assert r.status_code == 400 and r.json()["code"] == "DUPLICATE_SECONDS"
    # 不同段同秒数允许
    assert upload(client, [(2, 100, 3.0)]).status_code == 200
    assert client.get("/api/status").json()["points"]["accepted"] == 2


def test_point_cap_enforced(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch, max_points="5") as c:
        assert upload(c, [(1, i, 1.0) for i in range(4)]).status_code == 200
        r = upload(c, [(1, 10, 1.0), (1, 11, 1.0)])
        assert r.status_code == 400 and r.json()["code"] == "POINT_LIMIT_EXCEEDED"
        assert c.get("/api/status").json()["points"]["accepted"] == 4
        assert upload(c, [(1, 10, 1.0)]).status_code == 200
        assert c.get("/api/status").json()["points"]["accepted"] == 5


def test_bad_rows_reported_with_row_numbers(client):
    content = "segment,local_seconds,temperature\n1,100,abc\n2,x,1.0\n"
    r = client.post("/api/uploads", files={"file": ("b.csv", content, "text/csv")})
    assert r.status_code == 400
    assert {e["row"] for e in r.json()["errors"]} == {1, 2}
    assert client.get("/api/status").json()["points"]["accepted"] == 0


def test_csv_chinese_header_bom_and_no_header(client):
    content = "﻿开机段号,本地秒数,温度\n1,100,4.5\n"
    r = client.post("/api/uploads", files={"file": ("c.csv", content.encode("utf-8"), "text/csv")})
    assert r.status_code == 200, r.text
    r = client.post("/api/uploads",
                    files={"file": ("d.csv", "2,100,4.5\n2,101,4.6\n", "text/csv")})
    assert r.status_code == 200 and r.json()["accepted"] == 2


# ---------------------------------------------------------------- 锚点

def test_anchor_create_and_validation(client):
    r = set_anchors(client, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
    assert r.status_code == 200 and r.json()["version"] == 1
    # UTC 不递增
    r = set_anchors(client, 2, 0, "2026-01-02T00:00:00Z", 10, "2026-01-01T00:00:00Z")
    assert r.status_code == 400 and r.json()["code"] == "UTC_NOT_INCREASING"
    # 本地秒数不递增
    r = set_anchors(client, 2, 10, "2026-01-01T00:00:00Z", 10, "2026-01-02T00:00:00Z")
    assert r.status_code == 400 and r.json()["code"] == "BAD_LOCAL_RANGE"


def test_anchor_correction_requires_seen_version(client):
    assert set_anchors(client, 1, 100, "2026-01-01T00:00:00Z",
                     200, "2026-01-01T00:10:00Z").json()["version"] == 1
    # 未带版本
    r = set_anchors(client, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:20:00Z")
    assert r.status_code == 409 and r.json()["current_version"] == 1
    # 版本不符
    r = set_anchors(client, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:20:00Z",
                    version=3)
    assert r.status_code == 409 and r.json()["current_version"] == 1
    # 版本正确
    r = set_anchors(client, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:20:00Z",
                    version=1)
    assert r.status_code == 200 and r.json()["version"] == 2
    # 对不存在的段按版本更正
    r = set_anchors(client, 9, 0, "2026-01-01T00:00:00Z", 10, "2026-01-01T00:10:00Z",
                    version=1)
    assert r.status_code == 409 and r.json()["current_version"] is None


# ---------------------------------------------------------------- 校时

def test_calibration_math_and_pending(client):
    upload(client, [(1, 99, 1.0), (1, 100, 2.0), (1, 150, 3.0),
                    (1, 200, 4.0), (1, 201, 5.0), (2, 100, 6.0)])
    set_anchors(client, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
    body = client.get("/api/readings", params={"date": "2026-01-01"}).json()
    assert body["count"] == 3
    times = {row["local_seconds"]: row["utc_time"] for row in body["readings"]}
    assert times[100] == "2026-01-01T00:00:00.000Z"
    assert times[150] == "2026-01-01T00:05:00.000Z"   # 线性中点
    assert times[200] == "2026-01-01T00:10:00.000Z"
    # 查询结果须含批次号、原行号、锚点版本
    for row in body["readings"]:
        assert row["batch_id"] == 1
        assert row["anchor_version"] == 1
        assert row["row_no"] >= 1
    rows_by_sec = {row["local_seconds"]: row["row_no"] for row in body["readings"]}
    assert rows_by_sec == {100: 2, 150: 3, 200: 4}
    # 范围外与无锚点的读数留存待校时
    pend = client.get("/api/pending").json()
    got = sorted((p["segment"], p["local_seconds"], p["reason"]) for p in pend["pending"])
    assert [(s, sec) for s, sec, _ in got] == [(1, 99), (1, 201), (2, 100)]
    assert "未设置" in got[2][2] and "超出" in got[0][2]


def test_day_crossing_moves_atomically(client):
    upload(client, [(1, 100, 1.0), (1, 150, 2.0)])
    # 100 → 1日23:00，150 → 2日00:00（恰跨日）
    set_anchors(client, 1, 100, "2026-01-01T23:00:00Z", 200, "2026-01-02T01:00:00Z")
    d1 = client.get("/api/readings", params={"date": "2026-01-01"}).json()
    d2 = client.get("/api/readings", params={"date": "2026-01-02"}).json()
    assert d1["count"] == 1 and d2["count"] == 1
    # 更正锚点，整体后移一天：旧日移除与新日加入一起生效
    set_anchors(client, 1, 100, "2026-01-02T23:00:00Z", 200, "2026-01-03T01:00:00Z",
                version=1)
    d1 = client.get("/api/readings", params={"date": "2026-01-01"}).json()
    d2 = client.get("/api/readings", params={"date": "2026-01-02"}).json()
    d3 = client.get("/api/readings", params={"date": "2026-01-03"}).json()
    assert d1["count"] == 0 and d2["count"] == 1 and d3["count"] == 1
    assert d2["readings"][0]["anchor_version"] == 2
    assert d2["calc_version"] == d3["calc_version"]  # 同一次换版


# ---------------------------------------------------------------- 并发 / 恢复

def test_interleaved_upload_during_recalc(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch, worker="1", delay="0.6") as c:
        upload(c, [(1, 100, 1.0)])
        set_anchors(c, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
        wait_idle(c)
        v1 = c.get("/api/status").json()["calc"]["version"]

        upload(c, [(1, 150, 2.0)])  # 触发一次带延迟的重算
        t0 = time.time()
        while time.time() - t0 < 5:  # 等它进入重算中
            if c.get("/api/status").json()["calc"]["status"] == "running":
                break
            time.sleep(0.02)
        r = upload(c, [(1, 160, 3.0)])  # 重算期间上传新点，须被接纳
        assert r.status_code == 200

        st = wait_idle(c)
        assert st["version"] > v1  # 版本单调递增，不回退
        body = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert sorted(row["local_seconds"] for row in body["readings"]) == [100, 150, 160]


def test_old_version_visible_during_recalc(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch, worker="1", delay="0.6") as c:
        upload(c, [(1, 100, 1.0)])
        set_anchors(c, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
        wait_idle(c)
        # 更正锚点使读数跨到次日；重算带延迟，期间旧版仍完整可见
        set_anchors(c, 1, 100, "2026-01-02T00:00:00Z", 200, "2026-01-02T00:10:00Z",
                    version=1)
        d1 = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        d2 = c.get("/api/readings", params={"date": "2026-01-02"}).json()
        assert d1["count"] == 1 and d2["count"] == 0
        wait_idle(c)
        d1 = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        d2 = c.get("/api/readings", params={"date": "2026-01-02"}).json()
        assert d1["count"] == 0 and d2["count"] == 1
        assert d2["readings"][0]["anchor_version"] == 2


def test_restart_recovers_pending_recalc(tmp_path, monkeypatch):
    from app import calibration
    from app.main import create_app

    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("WORKER_ENABLED", "0")
    # 模拟“崩溃前没来得及重算”：屏蔽同步重算，只落库
    orig = calibration.RecalcWorker.run_until_quiet
    monkeypatch.setattr(calibration.RecalcWorker, "run_until_quiet", lambda self: None)
    with TestClient(create_app()) as c1:
        upload(c1, [(1, 100, 1.0)])
        set_anchors(c1, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
        assert c1.get("/api/status").json()["calc"]["dirty"] is True
    monkeypatch.setattr(calibration.RecalcWorker, "run_until_quiet", orig)
    # 重启后：后台线程自动补算，已接纳且在锚点范围内的点最终进入当前版
    monkeypatch.setenv("WORKER_ENABLED", "1")
    with TestClient(create_app()) as c2:
        wait_idle(c2)
        body = c2.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert body["count"] == 1
        assert body["readings"][0]["local_seconds"] == 100


def test_failed_recalc_keeps_old_version(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as c:  # 同步重算模式
        upload(c, [(1, 100, 1.0)])
        set_anchors(c, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
        before = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert before["count"] == 1
        v_before = c.get("/api/status").json()["calc"]["version"]

        from app import calibration

        def boom(readings, anchors):
            raise RuntimeError("模拟计算失败")

        monkeypatch.setattr(calibration, "compute_calibration", boom)
        r = upload(c, [(1, 150, 2.0)])  # 新点仍被接纳，但重算失败
        assert r.status_code == 200
        calc = c.get("/api/status").json()["calc"]
        assert calc["status"] == "failed"
        assert calc["version"] == v_before          # 版本没有推进
        after = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert after["count"] == 1                   # 旧版完整，无半版
        assert [row["local_seconds"] for row in after["readings"]] == [100]

        monkeypatch.undo()  # 恢复 compute_calibration
        c.post("/api/recalc")
        calc = c.get("/api/status").json()["calc"]
        assert calc["status"] == "idle" and not calc["dirty"]
        body = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert sorted(row["local_seconds"] for row in body["readings"]) == [100, 150]


# ---------------------------------------------------------------- 页面

def test_html_pages_render(client):
    upload(client, [(1, 100, 1.0)])
    set_anchors(client, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
    for path in ["/", "/upload", "/anchors", "/query?date=2026-01-01"]:
        r = client.get(path)
        assert r.status_code == 200 and "text/html" in r.headers["content-type"], path
    html = client.get("/query?date=2026-01-01").text
    assert "批次号" in html and "原行号" in html and "锚点版本" in html


# ---------------------------------------------------------------- 隔离

def set_isolation(client: TestClient, seg, start, end, reason="探头故障", version=None):
    payload = {"segment": seg, "local_start": start, "local_end": end, "reason": reason}
    if version is not None:
        payload["version"] = version
    return client.post("/api/isolations", json=payload)


def clear_isolation(client: TestClient, seg, version=None):
    payload = {"segment": seg}
    if version is not None:
        payload["version"] = version
    return client.post("/api/isolations/clear", json=payload)


def secs(body):
    return [r["local_seconds"] for r in body["readings"]]


def test_isolation_hides_closed_interval_and_keeps_trace(client):
    upload(client, [(1, 100, 1.0), (1, 150, 2.0), (1, 200, 3.0), (1, 250, 4.0)])
    set_anchors(client, 1, 100, "2026-01-01T00:00:00Z", 300, "2026-01-01T00:30:00Z")
    r = set_isolation(client, 1, 150, 200, "探头断线，读数恒定")
    assert r.status_code == 200 and r.json()["version"] == 1

    body = client.get("/api/readings", params={"date": "2026-01-01"}).json()
    assert secs(body) == [100, 250]                 # 闭区间两端 150、200 都被隔离
    assert body["count"] == 2
    # 隔离点保留批次号、原行号、原因，且与可交付读数同版本
    iso = body["isolated"]
    assert [(p["local_seconds"], p["row_no"]) for p in iso] == [(150, 2), (200, 3)]
    assert all(p["batch_id"] == 1 and "断线" in p["reason"] for p in iso)
    assert all(p["calc_version"] == body["calc_version"] for p in iso)
    # 隔离点不计入待校时
    pend = client.get("/api/pending").json()["pending"]
    assert pend == []


def test_isolation_validation_and_version_conflict(client):
    upload(client, [(1, 100, 1.0)])
    assert set_isolation(client, 1, 100, 100, "").json()["code"] == "REASON_REQUIRED"
    assert set_isolation(client, 1, 200, 100).json()["code"] == "BAD_LOCAL_RANGE"
    assert set_isolation(client, 1, 100, 100).status_code == 200

    # 更正未带版本 / 版本不符 → 409，原决定不变
    r = set_isolation(client, 1, 100, 150, "更正")
    assert r.status_code == 409 and r.json()["current_version"] == 1
    r = set_isolation(client, 1, 100, 150, "更正", version=5)
    assert r.status_code == 409 and r.json()["current_version"] == 1
    rec = client.get("/api/isolations").json()["isolations"][0]
    assert (rec["local_start"], rec["local_end"], rec["reason"]) == (100, 100, "探头故障")
    # 原记录无版本字段时不能按更正提交
    r = set_isolation(client, 2, 10, 20, version=1)
    assert r.status_code == 409 and r.json()["current_version"] is None

    r = set_isolation(client, 1, 100, 150, "更正", version=1)
    assert r.status_code == 200 and r.json()["version"] == 2


def test_clear_isolation_restores_points(client):
    upload(client, [(1, 100, 1.0), (1, 150, 2.0)])
    set_anchors(client, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
    set_isolation(client, 1, 150, 150)
    body = client.get("/api/readings", params={"date": "2026-01-01"}).json()
    assert secs(body) == [100]

    # 解除须带所见版本
    r = clear_isolation(client, 1)
    assert r.status_code == 409 and r.json()["current_version"] == 1
    r = clear_isolation(client, 1, version=1)
    assert r.status_code == 200 and r.json()["version"] == 2
    body = client.get("/api/readings", params={"date": "2026-01-01"}).json()
    assert secs(body) == [100, 150]                 # 直接从原始点恢复供数
    assert body["isolated"] == []
    # 重复解除被拒绝
    assert clear_isolation(client, 1, version=2).json()["code"] == "ALREADY_CLEARED"
    # 记录保留供追溯（active=False）
    rec = client.get("/api/isolations").json()["isolations"][0]
    assert rec["active"] is False and rec["reason"] == "探头故障"


def test_isolation_decision_applies_to_later_calibrated_points(client):
    """隔离决定先于锚点：尚未校时的点日后进入锚点范围也沿用隔离。"""
    upload(client, [(1, 100, 1.0), (1, 150, 2.0)])
    set_isolation(client, 1, 140, 160, "该时段探头故障")
    pend = client.get("/api/pending").json()["pending"]
    # 隔离点不在待校时中，普通无锚点点仍在
    assert [p["local_seconds"] for p in pend] == [100]

    set_anchors(client, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
    body = client.get("/api/readings", params={"date": "2026-01-01"}).json()
    assert secs(body) == [100]
    assert [p["local_seconds"] for p in body["isolated"]] == [150]
    assert body["isolated"][0]["reason"] == "该时段探头故障"


def test_widened_isolation_removes_already_delivered_point(client):
    upload(client, [(1, 100, 1.0), (1, 150, 2.0), (1, 200, 3.0)])
    set_anchors(client, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
    assert secs(client.get("/api/readings", params={"date": "2026-01-01"}).json()) == [100, 150, 200]
    set_isolation(client, 1, 100, 150)
    body = client.get("/api/readings", params={"date": "2026-01-01"}).json()
    assert secs(body) == [200]                       # 已交付点随换版一起撤下
    assert sorted(p["local_seconds"] for p in body["isolated"]) == [100, 150]


# ---------------------------------------------------------------- 隔离与校时交错

def test_old_version_marked_undeliverable_while_isolation_pending(tmp_path, monkeypatch):
    """隔离变更触发重算期间，旧版仍可回看但标为不可交付；换版后隔离与读数一同生效。"""
    with make_client(tmp_path, monkeypatch, worker="1", delay="0.6") as c:
        upload(c, [(1, 100, 1.0), (1, 150, 2.0)])
        set_anchors(c, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
        wait_idle(c)
        v1 = c.get("/api/readings", params={"date": "2026-01-01"}).json()["calc_version"]

        set_isolation(c, 1, 150, 150)
        t0 = time.time()
        while time.time() - t0 < 5:
            if c.get("/api/status").json()["calc"]["status"] == "running":
                break
            time.sleep(0.02)
        mid = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert mid["dirty"] is True and mid["deliverable"] is False  # 旧版可回看但标不可交付
        assert mid["calc_version"] == v1
        assert secs(mid) == [100, 150]                            # 但尚未应用新隔离
        assert "不可交付" in c.get("/query?date=2026-01-01").text

        wait_idle(c)
        after = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert after["dirty"] is False and after["deliverable"] is True
        assert after["calc_version"] == v1 + 1
        assert "不可交付" not in c.get("/query?date=2026-01-01").text
        assert secs(after) == [100]
        assert [p["local_seconds"] for p in after["isolated"]] == [150]
        assert all(p["calc_version"] == after["calc_version"] for p in after["isolated"])


def test_late_recalc_cannot_overwrite_isolation_decision(tmp_path, monkeypatch):
    """迟到的重算不得覆盖新隔离决定：代次复核使过期结果丢弃。"""
    with make_client(tmp_path, monkeypatch, worker="1", delay="0.6") as c:
        upload(c, [(1, 100, 1.0)])
        set_anchors(c, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
        wait_idle(c)

        # 上传触发一次带延迟的重算，在其计算窗口内登记隔离
        upload(c, [(1, 150, 2.0)])
        t0 = time.time()
        while time.time() - t0 < 5:
            if c.get("/api/status").json()["calc"]["status"] == "running":
                break
            time.sleep(0.02)
        assert set_isolation(c, 1, 150, 150, "重算窗口内登记").status_code == 200
        wait_idle(c)
        body = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert secs(body) == [100]                 # 新隔离决定生效，迟到结果未覆盖
        assert [p["local_seconds"] for p in body["isolated"]] == [150]
        assert body["isolated"][0]["reason"] == "重算窗口内登记"


def test_clear_during_recalc_keeps_new_decision(tmp_path, monkeypatch):
    """解除隔离与新上传交错：最终版本必须反映解除，且版本单调。"""
    with make_client(tmp_path, monkeypatch, worker="1", delay="0.6") as c:
        upload(c, [(1, 100, 1.0), (1, 150, 2.0)])
        set_anchors(c, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
        set_isolation(c, 1, 150, 150)
        wait_idle(c)
        v1 = c.get("/api/status").json()["calc"]["version"]
        assert secs(c.get("/api/readings", params={"date": "2026-01-01"}).json()) == [100]

        upload(c, [(1, 175, 3.0)])  # 触发带延迟重算
        t0 = time.time()
        while time.time() - t0 < 5:
            if c.get("/api/status").json()["calc"]["status"] == "running":
                break
            time.sleep(0.02)
        assert clear_isolation(c, 1, version=1).status_code == 200
        st = wait_idle(c)
        assert st["version"] > v1
        body = c.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert secs(body) == [100, 150, 175]
        assert body["isolated"] == []


def test_failed_recalc_keeps_old_version_and_isolation(client):
    upload(client, [(1, 100, 1.0)])
    set_anchors(client, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
    before = client.get("/api/readings", params={"date": "2026-01-01"}).json()
    v_before = before["calc_version"]

    from app import calibration

    def boom(readings, anchors, isolations=None):
        raise RuntimeError("模拟计算失败")

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(calibration, "compute_calibration", boom)
        assert set_isolation(client, 1, 100, 100).status_code == 200
        calc = client.get("/api/status").json()["calc"]
        assert calc["status"] == "failed" and calc["version"] == v_before
        after = client.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert after["calc_version"] == v_before
        assert secs(after) == [100] and after["isolated"] == []  # 旧版原样保留，无半版
    finally:
        monkeypatch.undo()
    client.post("/api/recalc")
    body = client.get("/api/readings", params={"date": "2026-01-01"}).json()
    assert body["count"] == 0 and [p["local_seconds"] for p in body["isolated"]] == [100]
    assert body["calc_version"] == v_before + 1


def test_restart_recovers_isolation_change(tmp_path, monkeypatch):
    from app import calibration
    from app.main import create_app

    monkeypatch.setenv("DB_PATH", str(tmp_path / "iso.db"))
    monkeypatch.setenv("WORKER_ENABLED", "0")
    orig = calibration.RecalcWorker.run_until_quiet
    monkeypatch.setattr(calibration.RecalcWorker, "run_until_quiet", lambda self: None)
    with TestClient(create_app()) as c1:
        upload(c1, [(1, 100, 1.0), (1, 150, 2.0)])
        set_anchors(c1, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
        assert set_isolation(c1, 1, 150, 150).status_code == 200
        assert c1.get("/api/status").json()["calc"]["dirty"] is True
    monkeypatch.setattr(calibration.RecalcWorker, "run_until_quiet", orig)
    monkeypatch.setenv("WORKER_ENABLED", "1")
    with TestClient(create_app()) as c2:
        wait_idle(c2)
        body = c2.get("/api/readings", params={"date": "2026-01-01"}).json()
        assert secs(body) == [100]
        assert [p["local_seconds"] for p in body["isolated"]] == [150]


def test_isolation_html_flow(client):
    upload(client, [(1, 100, 1.0), (1, 150, 2.0)])
    set_anchors(client, 1, 100, "2026-01-01T00:00:00Z", 200, "2026-01-01T00:10:00Z")
    r = client.post("/isolations", data={
        "segment": "1", "local_start": "150", "local_end": "150",
        "reason": "探头故障", "version": "",
    }, follow_redirects=False)
    assert r.status_code == 303
    html = client.get("/isolations").text
    assert "隔离中" in html and "探头故障" in html and "150" in html
    # 版本冲突时页面返回提示且原决定保留
    r = client.post("/isolations", data={
        "segment": "1", "local_start": "100", "local_end": "100",
        "reason": "改区间", "version": "9",
    })
    assert r.status_code == 200 and "版本不一致" in r.text
    assert "探头故障" in r.text
    # 页面解除
    r = client.post("/isolations/clear", data={"segment": "1", "version": "1"},
                    follow_redirects=False)
    assert r.status_code == 303
    body = client.get("/api/readings", params={"date": "2026-01-01"}).json()
    assert secs(body) == [100, 150]
    assert "已解除" in client.get("/isolations").text
