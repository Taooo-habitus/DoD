"""Tests for server-level document concurrency limiting."""

from __future__ import annotations

import threading
import time

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import DoD.server.app as server_app  # noqa: E402


def _wait_for_success(client: TestClient, job_id: str, timeout_s: float = 8.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        response = client.get(f"/v1/jobs/{job_id}/result")
        if response.status_code == 200:
            return
        if response.status_code == 409:
            time.sleep(0.05)
            continue
        raise AssertionError(f"Unexpected status while waiting: {response.status_code}")
    raise AssertionError("Timed out waiting for job completion")


def test_server_respects_max_concurrent_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    """When max concurrent docs is 1, jobs should execute one at a time."""
    monkeypatch.setenv("DOD_SERVER_MAX_CONCURRENT_DOCS", "1")

    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()

    def fake_digest_document(_cfg):
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        try:
            time.sleep(0.15)
            return {
                "manifest": "manifest.json",
                "page_table": "page_table.jsonl",
                "toc_tree": "toc_tree.json",
            }
        finally:
            with lock:
                state["active"] -= 1

    monkeypatch.setattr(server_app, "digest_document", fake_digest_document)
    monkeypatch.setattr(
        server_app,
        "_build_result_payload",
        lambda _artifacts: {"toc_tree": {}, "page_table": [], "image_page_table": []},
    )

    with TestClient(server_app.app) as client:
        submit_a = client.post(
            "/v1/digest?wait=false",
            files={"file": ("a.pdf", b"%PDF-1.4\nA", "application/pdf")},
        )
        submit_b = client.post(
            "/v1/digest?wait=false",
            files={"file": ("b.pdf", b"%PDF-1.4\nB", "application/pdf")},
        )
        assert submit_a.status_code == 200
        assert submit_b.status_code == 200

        job_a = submit_a.json()["job_id"]
        job_b = submit_b.json()["job_id"]

        _wait_for_success(client, job_a)
        _wait_for_success(client, job_b)

    assert state["max_active"] == 1
