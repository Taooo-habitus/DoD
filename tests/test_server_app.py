"""Tests for the FastAPI server API lifecycle."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Generator

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import DoD.server.app as server_app  # noqa: E402


def _wait_for_result(client: TestClient, job_id: str, timeout_s: float = 5.0) -> dict:
    """Poll result endpoint until the job finishes."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        response = client.get(f"/v1/jobs/{job_id}/result")
        if response.status_code == 200:
            return response.json()
        if response.status_code in {409, 500}:
            time.sleep(0.05)
            continue
        raise AssertionError(f"Unexpected status code: {response.status_code}")
    raise AssertionError("Timed out waiting for job result")


@pytest.fixture
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient, None, None]:
    """Create a fresh server client with isolated working directory."""
    monkeypatch.setenv("DOD_SERVER_WORK_DIR", str(tmp_path / "server_jobs"))
    monkeypatch.setenv("DOD_SERVER_MAX_CONCURRENT_DOCS", "2")
    with TestClient(server_app.app) as test_client:
        yield test_client


def test_healthz(client: TestClient) -> None:
    """Health endpoint should return liveness metadata."""
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["max_concurrent_docs"] == 2


def test_digest_sync_success(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synchronous digest should return full result payload."""

    def fake_digest_document(_cfg):
        return {
            "manifest": "manifest.json",
            "page_table": "page_table.jsonl",
            "toc_tree": "toc_tree.json",
        }

    def fake_build_result_payload(_artifacts):
        return {
            "toc_tree": {"structure": []},
            "page_table": [{"page_id": 1, "text": "x"}],
            "image_page_table": [{"page_id": 1, "image_b64": "abc"}],
        }

    monkeypatch.setattr(server_app, "digest_document", fake_digest_document)
    monkeypatch.setattr(server_app, "_build_result_payload", fake_build_result_payload)

    response = client.post(
        "/v1/digest", files={"file": ("doc.pdf", b"%PDF-1.4\nfake", "application/pdf")}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "succeeded"
    assert "result" in payload
    assert "toc_tree" in payload["result"]
    assert "page_table" in payload["result"]
    assert "image_page_table" in payload["result"]


def test_digest_async_flow(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Async digest should queue first, then expose result via polling."""

    def fake_digest_document(_cfg):
        time.sleep(0.05)
        return {
            "manifest": "manifest.json",
            "page_table": "page_table.jsonl",
            "toc_tree": "toc_tree.json",
        }

    monkeypatch.setattr(server_app, "digest_document", fake_digest_document)
    monkeypatch.setattr(
        server_app,
        "_build_result_payload",
        lambda _artifacts: {"toc_tree": {}, "page_table": [], "image_page_table": []},
    )

    submit = client.post(
        "/v1/digest?wait=false",
        files={"file": ("doc.pdf", b"%PDF-1.4\nfake", "application/pdf")},
    )
    assert submit.status_code == 200
    job_id = submit.json()["job_id"]

    result = _wait_for_result(client, job_id)
    assert result["status"] == "succeeded"
    assert "result" in result


def test_digest_rejects_non_pdf(client: TestClient) -> None:
    """Server should reject non-PDF uploads."""
    response = client.post(
        "/v1/digest", files={"file": ("doc.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400
    assert "Only PDF input is supported" in response.json()["detail"]


def test_digest_rejects_empty_pdf(client: TestClient) -> None:
    """Server should reject empty PDF payloads."""
    response = client.post(
        "/v1/digest", files={"file": ("doc.pdf", b"", "application/pdf")}
    )
    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["detail"]


def test_unknown_job_returns_404(client: TestClient) -> None:
    """Unknown job ids should return not found."""
    response = client.get("/v1/jobs/does-not-exist")
    assert response.status_code == 404


def test_digest_sync_failure_returns_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synchronous wait mode should surface job failures immediately."""

    def fake_digest_document(_cfg):
        raise RuntimeError("boom")

    monkeypatch.setattr(server_app, "digest_document", fake_digest_document)

    response = client.post(
        "/v1/digest", files={"file": ("doc.pdf", b"%PDF-1.4\nfake", "application/pdf")}
    )
    assert response.status_code == 500
    assert "boom" in response.json()["detail"]


def test_build_result_payload_reads_artifacts(tmp_path: Path) -> None:
    """Result payload builder should load JSON and JSONL artifacts."""
    page_table_path = tmp_path / "page_table.jsonl"
    image_page_table_path = tmp_path / "image_page_table.jsonl"
    toc_tree_path = tmp_path / "toc_tree.json"
    manifest_path = tmp_path / "manifest.json"

    page_table_path.write_text(
        "\n".join(
            [
                json.dumps({"page_id": 1, "text": "A"}),
                json.dumps({"page_id": 2, "text": "B"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    image_page_table_path.write_text(
        json.dumps({"page_id": 1, "image_path": "x", "image_b64": "abc"}) + "\n",
        encoding="utf-8",
    )
    toc_tree_path.write_text(
        json.dumps({"doc_name": "doc", "structure": []}), encoding="utf-8"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "artifacts": {
                    "page_table": str(page_table_path),
                    "image_page_table": str(image_page_table_path),
                    "toc_tree": str(toc_tree_path),
                }
            }
        ),
        encoding="utf-8",
    )

    payload = server_app._build_result_payload(
        {
            "manifest": str(manifest_path),
            "page_table": str(page_table_path),
            "toc_tree": str(toc_tree_path),
        }
    )

    assert payload["toc_tree"]["doc_name"] == "doc"
    assert len(payload["page_table"]) == 2
    assert len(payload["image_page_table"]) == 1
