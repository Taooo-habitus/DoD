"""FastAPI server for concurrent document digestion jobs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
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
    job_ref: str
    file_name: str
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


def _make_job_ref(filename: str, job_id: str, created_at: str) -> str:
    """Create a human-readable job reference with uniqueness suffix."""
    stem = Path(filename).stem.lower().strip()
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    if not stem:
        stem = "document"
    try:
        dt = datetime.fromisoformat(created_at)
    except ValueError:
        dt = datetime.now(timezone.utc)
    timestamp = dt.strftime("%Y%m%d-%H%M%S")
    return f"{stem}-{timestamp}-{job_id[:6]}"


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


def _resolve_artifact_path(job: JobRecord, artifact_key: str) -> Optional[Path]:
    """Resolve an artifact path from job state, result payload, or manifest."""
    candidates: List[Path] = []

    direct_value = job.artifacts.get(artifact_key)
    if isinstance(direct_value, str) and direct_value.strip():
        candidates.append(Path(direct_value))

    result_paths = (
        job.result.get("artifact_paths", {}) if isinstance(job.result, dict) else {}
    )
    if isinstance(result_paths, dict):
        result_value = result_paths.get(artifact_key)
        if isinstance(result_value, str) and result_value.strip():
            candidates.append(Path(result_value))

    manifest_value = job.artifacts.get("manifest")
    if isinstance(manifest_value, str) and manifest_value.strip():
        manifest_path = Path(manifest_value)
        if manifest_path.exists() and manifest_path.is_file():
            try:
                manifest = read_json(manifest_path)
            except Exception:  # noqa: BLE001 - endpoint handles missing artifact below
                manifest = {}
            artifacts_map = (
                manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
            )
            if isinstance(artifacts_map, dict):
                nested_value = artifacts_map.get(artifact_key)
                if isinstance(nested_value, str) and nested_value.strip():
                    candidates.append(Path(nested_value))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


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


def _parse_page_ids(raw: Optional[str]) -> List[int]:
    """Parse comma-separated page ids into sorted unique integers."""
    if raw is None or not raw.strip():
        return []
    parsed: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        parsed.append(int(part))
    return sorted(set(parsed))


def _resolve_page_selection(
    total_pages: int,
    page_ids: Optional[str] = None,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> List[int]:
    """Resolve selected page ids from explicit ids or a closed range."""
    if total_pages <= 0:
        return []
    if page_ids:
        selected = _parse_page_ids(page_ids)
    elif start_page is not None or end_page is not None:
        start = 1 if start_page is None else start_page
        end = total_pages if end_page is None else end_page
        if start <= 0 or end <= 0 or start > end:
            raise ValueError("Invalid page range.")
        selected = list(range(start, end + 1))
    else:
        selected = list(range(1, total_pages + 1))
    if any(page_id <= 0 for page_id in selected):
        raise ValueError("Page ids must be positive integers.")
    return [page_id for page_id in selected if page_id <= total_pages]


def _format_page_ranges(page_numbers: List[int]) -> str:
    """Format page numbers as compact ranges, e.g. '1-3,7,9-10'."""
    if not page_numbers:
        return ""
    sorted_pages = sorted(set(page_numbers))
    ranges: List[str] = []
    start = sorted_pages[0]
    end = sorted_pages[0]
    for page in sorted_pages[1:]:
        if page == end + 1:
            end = page
            continue
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        start = end = page
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    return ",".join(ranges)


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
    _app.state.job_refs: Dict[str, str] = {}
    _app.state.jobs_lock = asyncio.Lock()
    yield


app = FastAPI(title="DoD Server", version="0.1.0", lifespan=_lifespan)


async def _get_job_or_404(job_ref_or_id: str) -> JobRecord:
    """Return a job record from a UUID or job_ref, else raise 404."""
    async with app.state.jobs_lock:
        job_id = app.state.jobs.get(job_ref_or_id)
        if isinstance(job_id, JobRecord):
            return job_id
        resolved_job_id = app.state.job_refs.get(job_ref_or_id, job_ref_or_id)
        job = app.state.jobs.get(resolved_job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail=f"Unknown job_ref or job_id: {job_ref_or_id}"
        )
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
                artifact_paths = (
                    result.get("artifact_paths", {}) if isinstance(result, dict) else {}
                )
                merged_artifacts = dict(artifacts)
                if isinstance(artifact_paths, dict):
                    for key, value in artifact_paths.items():
                        if isinstance(value, str):
                            merged_artifacts[key] = value
                job.artifacts = merged_artifacts
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
    job_ref = _make_job_ref(filename, job_id, now)
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
            job_ref=job_ref,
            file_name=filename,
            status="queued",
            created_at=now,
            updated_at=now,
            input_path=str(input_path),
            output_dir=str(output_dir),
            task=task,
        )
        app.state.job_refs[job_ref] = job_id

    if not wait:
        return {
            "job_id": job_id,
            "job_ref": job_ref,
            "status": "queued",
            "status_url": f"/v1/jobs/{job_id}",
            "status_ref_url": f"/v1/jobs/{job_ref}",
            "result_url": f"/v1/jobs/{job_id}/result",
            "result_ref_url": f"/v1/jobs/{job_ref}/result",
        }

    await task
    job = await _get_job_or_404(job_id)
    if job.status == "failed":
        raise HTTPException(status_code=500, detail=job.error or "Job failed")
    return {
        "job_id": job.job_id,
        "job_ref": job.job_ref,
        "file_name": job.file_name,
        "status": job.status,
        "result": job.result,
    }


@app.get("/v1/jobs")
async def list_jobs() -> Dict[str, Any]:
    """List submitted jobs with compact metadata."""
    async with app.state.jobs_lock:
        jobs = list(app.state.jobs.values())

    jobs.sort(key=lambda job: job.created_at, reverse=True)
    return {
        "jobs": [
            {
                "job_id": job.job_id,
                "job_ref": job.job_ref,
                "file_name": job.file_name,
                "status": job.status,
                "created_at": job.created_at,
            }
            for job in jobs
        ]
    }


@app.get("/v1/jobs/{job_ref_or_id}")
async def get_job(job_ref_or_id: str) -> Dict[str, Any]:
    """Get job status and metadata."""
    job = await _get_job_or_404(job_ref_or_id)
    return {
        "job_id": job.job_id,
        "job_ref": job.job_ref,
        "file_name": job.file_name,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "input_path": job.input_path,
        "output_dir": job.output_dir,
        "error": job.error,
        "status_url": f"/v1/jobs/{job.job_id}",
        "status_ref_url": f"/v1/jobs/{job.job_ref}",
        "result_url": f"/v1/jobs/{job.job_id}/result",
        "result_ref_url": f"/v1/jobs/{job.job_ref}/result",
    }


@app.get("/v1/jobs/{job_ref_or_id}/result")
async def get_job_result(job_ref_or_id: str) -> Dict[str, Any]:
    """Get completed job output payload."""
    job = await _get_job_or_404(job_ref_or_id)
    if job.status in {"queued", "running"}:
        raise HTTPException(status_code=409, detail=f"Job is still {job.status}.")
    if job.status == "failed":
        raise HTTPException(status_code=500, detail=job.error or "Job failed")
    return {
        "job_id": job.job_id,
        "job_ref": job.job_ref,
        "file_name": job.file_name,
        "status": job.status,
        "result": job.result,
    }


def _ensure_job_succeeded(job: JobRecord) -> None:
    """Raise HTTP errors when a job is not successfully completed."""
    if job.status in {"queued", "running"}:
        raise HTTPException(status_code=409, detail=f"Job is still {job.status}.")
    if job.status == "failed":
        raise HTTPException(status_code=500, detail=job.error or "Job failed")


@app.get("/v1/docs/{job_ref}/toc")
async def get_doc_toc(job_ref: str) -> Dict[str, Any]:
    """Return TOC tree for a completed job."""
    job = await _get_job_or_404(job_ref)
    _ensure_job_succeeded(job)
    toc_path = _resolve_artifact_path(job, "toc_tree")
    if toc_path is None:
        raise HTTPException(status_code=500, detail="TOC artifact not found.")
    return {
        "job_id": job.job_id,
        "job_ref": job.job_ref,
        "status": job.status,
        "toc_tree": read_json(toc_path),
    }


@app.get("/v1/docs/{job_ref}/pages/text")
async def get_doc_page_texts(
    job_ref: str,
    page_ids: Optional[str] = Query(default=None),
    start_page: Optional[int] = Query(default=None),
    end_page: Optional[int] = Query(default=None),
    max_chars_per_page: Optional[int] = Query(default=None),
    max_pages_per_call: int = Query(default=8, ge=1),
) -> Dict[str, Any]:
    """Return selected page text rows from page_table artifact."""
    job = await _get_job_or_404(job_ref)
    _ensure_job_succeeded(job)
    page_table_path = _resolve_artifact_path(job, "page_table")
    if page_table_path is None:
        raise HTTPException(status_code=500, detail="Page table artifact not found.")
    rows = _read_jsonl(page_table_path)
    try:
        selected = _resolve_page_selection(
            total_pages=len(rows),
            page_ids=page_ids,
            start_page=start_page,
            end_page=end_page,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    requested_pages = _format_page_ranges(selected)
    selected_pages = set(selected[:max_pages_per_call])
    returned_pages = _format_page_ranges(selected[:max_pages_per_call])
    output_rows: List[Dict[str, Any]] = []
    for row in rows:
        page_id = row.get("page_id")
        if not isinstance(page_id, int) or page_id not in selected_pages:
            continue
        text = row.get("text")
        if isinstance(text, str) and isinstance(max_chars_per_page, int):
            text = text[: max(1, max_chars_per_page)]
        output_rows.append({"page_id": page_id, "text": text})
    return {
        "job_id": job.job_id,
        "job_ref": job.job_ref,
        "status": job.status,
        "requested_pages": requested_pages,
        "returned_pages": returned_pages,
        "pages": output_rows,
    }


@app.get("/v1/docs/{job_ref}/pages/images")
async def get_doc_page_images(
    job_ref: str,
    page_ids: Optional[str] = Query(default=None),
    start_page: Optional[int] = Query(default=None),
    end_page: Optional[int] = Query(default=None),
    mode: str = Query(default="path"),
    max_pages_per_call: int = Query(default=4, ge=1),
) -> Dict[str, Any]:
    """Return selected page image rows from image_page_table artifact."""
    if mode not in {"path", "base64"}:
        raise HTTPException(status_code=400, detail="mode must be 'path' or 'base64'.")
    job = await _get_job_or_404(job_ref)
    _ensure_job_succeeded(job)
    image_page_table_path = _resolve_artifact_path(job, "image_page_table")
    if image_page_table_path is None:
        raise HTTPException(
            status_code=500, detail="Image page table artifact not found."
        )
    rows = _read_jsonl(image_page_table_path)
    try:
        selected = _resolve_page_selection(
            total_pages=len(rows),
            page_ids=page_ids,
            start_page=start_page,
            end_page=end_page,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    requested_pages = _format_page_ranges(selected)
    selected_pages = set(selected[:max_pages_per_call])
    returned_pages = _format_page_ranges(selected[:max_pages_per_call])
    output_rows: List[Dict[str, Any]] = []
    for row in rows:
        page_id = row.get("page_id")
        if not isinstance(page_id, int) or page_id not in selected_pages:
            continue
        if mode == "base64":
            output_rows.append(
                {"page_id": page_id, "image_b64": row.get("image_b64", "")}
            )
        else:
            output_rows.append(
                {"page_id": page_id, "image_path": row.get("image_path", "")}
            )
    return {
        "job_id": job.job_id,
        "job_ref": job.job_ref,
        "status": job.status,
        "requested_pages": requested_pages,
        "returned_pages": returned_pages,
        "mode": mode,
        "pages": output_rows,
    }
