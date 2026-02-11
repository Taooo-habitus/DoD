"""FastAPI server for concurrent document digestion jobs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile

from DoD.config import PipelineConfig
from DoD.io.artifacts import read_json
from DoD.pipeline import digest_document

logger = logging.getLogger(__name__)


@dataclass
class JobRecord:
    """In-memory job state for a single request."""

    job_id: str
    status: str
    created_at: str
    updated_at: str
    input_path: str
    output_dir: str
    artifacts: Dict[str, str] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    task: Optional[asyncio.Task[None]] = None


def _now_utc_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file as a list of dictionaries."""
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_result_payload(artifacts: Dict[str, str]) -> Dict[str, Any]:
    """Load generated artifacts into API-friendly response payload."""
    manifest_path = Path(artifacts["manifest"])
    manifest = read_json(manifest_path)

    page_table_path = Path(artifacts["page_table"])
    toc_tree_path = Path(artifacts["toc_tree"])
    image_page_table_path = Path(manifest["artifacts"]["image_page_table"])

    return {
        "toc_tree": read_json(toc_tree_path),
        "page_table": _read_jsonl(page_table_path),
        "image_page_table": _read_jsonl(image_page_table_path),
        "artifact_paths": manifest["artifacts"],
        "profiling": manifest.get("profiling", {}),
        "profiling_summary": _format_profiling_summary(manifest.get("profiling", {})),
        "manifest": manifest,
    }


def _format_profiling_summary(profiling: Dict[str, Any]) -> str:
    """Return a concise profiling summary string."""
    stages = profiling.get("stages", {})
    stage_parts: List[str] = []
    if isinstance(stages, dict):
        for name, payload in stages.items():
            if not isinstance(payload, dict):
                continue
            seconds = payload.get("seconds")
            skipped = payload.get("skipped")
            if skipped:
                stage_parts.append(f"{name}=skipped")
            elif isinstance(seconds, (int, float)):
                stage_parts.append(f"{name}={seconds:.2f}s")
    total_seconds = profiling.get("total_seconds")
    page_count = profiling.get("page_count")
    suffix_parts: List[str] = []
    if isinstance(total_seconds, (int, float)):
        suffix_parts.append(f"total={total_seconds:.2f}s")
    if isinstance(page_count, int):
        suffix_parts.append(f"pages={page_count}")
    summary = ", ".join(stage_parts)
    if suffix_parts:
        summary = (
            f"{summary} ({', '.join(suffix_parts)})"
            if summary
            else ", ".join(suffix_parts)
        )
    return summary or "profiling unavailable"


def _build_pipeline_config(
    *,
    input_path: Path,
    output_dir: Path,
    text_extractor_backend: Optional[str],
    normalize_max_pages: Optional[int],
    toc_backend: Optional[str],
    toc_model: Optional[str],
    toc_concurrent_requests: Optional[int],
    toc_check_page_num: Optional[int],
    toc_api_key: Optional[str],
    toc_api_base_url: Optional[str],
) -> PipelineConfig:
    """Build a request-scoped pipeline config."""
    cfg = _load_default_config()
    cfg.input_path = str(input_path)
    cfg.artifacts.output_dir = str(output_dir)

    if text_extractor_backend:
        cfg.text_extractor.backend = text_extractor_backend
    if normalize_max_pages is not None:
        cfg.normalize.max_pages = normalize_max_pages
    if toc_backend:
        cfg.toc.backend = toc_backend
    if toc_model:
        cfg.toc.model = toc_model
    if toc_concurrent_requests is not None:
        cfg.toc.concurrent_requests = max(1, int(toc_concurrent_requests))
    if toc_check_page_num is not None:
        cfg.toc.toc_check_page_num = max(1, int(toc_check_page_num))
    if toc_api_key is not None:
        cfg.toc.api_key = toc_api_key
    if toc_api_base_url is not None:
        cfg.toc.api_base_url = toc_api_base_url
    return cfg


def _load_default_config() -> PipelineConfig:
    """Load config defaults from conf/config.yaml for server jobs."""
    try:
        import yaml
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError(
            "pyyaml and omegaconf are required to load server config defaults."
        ) from exc

    config_path = Path(__file__).resolve().parents[3] / "conf" / "config.yaml"
    cfg = PipelineConfig()
    if not config_path.exists():
        return cfg
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "hydra" in raw:
        raw.pop("hydra", None)
    data = OmegaConf.to_container(OmegaConf.create(raw or {}), resolve=True)
    if isinstance(data, dict):
        merged = OmegaConf.merge(OmegaConf.structured(PipelineConfig()), data)
        cfg_obj = OmegaConf.to_object(merged)
        if isinstance(cfg_obj, PipelineConfig):
            cfg = cfg_obj
    return cfg


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Initialize app-scoped state."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )
    logging.getLogger("DoD").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    work_dir = Path(os.getenv("DOD_SERVER_WORK_DIR", "outputs/server_jobs")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    _app.state.work_dir = work_dir
    _app.state.max_concurrent_docs = max(
        1, _env_int("DOD_SERVER_MAX_CONCURRENT_DOCS", 2)
    )
    _app.state.doc_semaphore = asyncio.Semaphore(_app.state.max_concurrent_docs)
    _app.state.jobs: Dict[str, JobRecord] = {}
    _app.state.jobs_lock = asyncio.Lock()
    yield


app = FastAPI(title="DoD Server", version="0.1.0", lifespan=_lifespan)


async def _get_job_or_404(job_id: str) -> JobRecord:
    """Return a job record or raise 404."""
    async with app.state.jobs_lock:
        job = app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    return job


async def _run_job(job_id: str, cfg: PipelineConfig) -> None:
    """Execute a job under document-level concurrency control."""
    async with app.state.doc_semaphore:
        async with app.state.jobs_lock:
            job = app.state.jobs[job_id]
            job.status = "running"
            job.updated_at = _now_utc_iso()

        try:
            artifacts = await asyncio.to_thread(digest_document, cfg)
            result = await asyncio.to_thread(_build_result_payload, artifacts)
            summary = result.get("profiling_summary")
            if summary:
                logger.info("Pipeline finished: %s", summary)
            config_payload = result.get("manifest", {}).get("config")
            if config_payload is not None:
                logger.info("Pipeline config: %s", json.dumps(config_payload))
            async with app.state.jobs_lock:
                job = app.state.jobs[job_id]
                job.status = "succeeded"
                job.updated_at = _now_utc_iso()
                job.artifacts = artifacts
                job.result = result
        except Exception as exc:  # noqa: BLE001 - server should return job failure
            async with app.state.jobs_lock:
                job = app.state.jobs[job_id]
                job.status = "failed"
                job.updated_at = _now_utc_iso()
                job.error = str(exc)


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    """Liveness probe."""
    return {
        "status": "ok",
        "max_concurrent_docs": app.state.max_concurrent_docs,
        "work_dir": str(app.state.work_dir),
    }


@app.post("/v1/digest")
async def digest(
    file: UploadFile = File(...),
    text_extractor_backend: Optional[str] = Form(default=None),
    normalize_max_pages: Optional[int] = Form(default=None),
    toc_backend: Optional[str] = Form(default=None),
    toc_model: Optional[str] = Form(default=None),
    toc_concurrent_requests: Optional[int] = Form(default=None),
    toc_check_page_num: Optional[int] = Form(default=None),
    toc_api_key: Optional[str] = Form(default=None),
    toc_api_base_url: Optional[str] = Form(default=None),
    wait: bool = Query(default=True),
) -> Dict[str, Any]:
    """Submit a PDF digestion job; optionally wait for completion."""
    filename = file.filename or "input.pdf"
    suffix = Path(filename).suffix.lower() or ".pdf"
    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF input is supported.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    job_id = uuid.uuid4().hex
    now = _now_utc_iso()
    job_dir = app.state.work_dir / job_id
    input_path = job_dir / f"input{suffix}"
    output_dir = job_dir / "artifacts"
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(payload)

    cfg = _build_pipeline_config(
        input_path=input_path,
        output_dir=output_dir,
        text_extractor_backend=text_extractor_backend,
        normalize_max_pages=normalize_max_pages,
        toc_backend=toc_backend,
        toc_model=toc_model,
        toc_concurrent_requests=toc_concurrent_requests,
        toc_check_page_num=toc_check_page_num,
        toc_api_key=toc_api_key,
        toc_api_base_url=toc_api_base_url,
    )

    task = asyncio.create_task(_run_job(job_id, cfg))
    async with app.state.jobs_lock:
        app.state.jobs[job_id] = JobRecord(
            job_id=job_id,
            status="queued",
            created_at=now,
            updated_at=now,
            input_path=str(input_path),
            output_dir=str(output_dir),
            task=task,
        )

    if not wait:
        return {
            "job_id": job_id,
            "status": "queued",
            "status_url": f"/v1/jobs/{job_id}",
            "result_url": f"/v1/jobs/{job_id}/result",
        }

    await task
    job = await _get_job_or_404(job_id)
    if job.status == "failed":
        raise HTTPException(status_code=500, detail=job.error or "Job failed")
    return {"job_id": job.job_id, "status": job.status, "result": job.result}


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    """Get job status and metadata."""
    job = await _get_job_or_404(job_id)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "input_path": job.input_path,
        "output_dir": job.output_dir,
        "error": job.error,
        "status_url": f"/v1/jobs/{job_id}",
        "result_url": f"/v1/jobs/{job_id}/result",
    }


@app.get("/v1/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> Dict[str, Any]:
    """Get completed job output payload."""
    job = await _get_job_or_404(job_id)
    if job.status in {"queued", "running"}:
        raise HTTPException(status_code=409, detail=f"Job is still {job.status}.")
    if job.status == "failed":
        raise HTTPException(status_code=500, detail=job.error or "Job failed")
    return {"job_id": job.job_id, "status": job.status, "result": job.result}
