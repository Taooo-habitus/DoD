# DoD (Document Outline Discovery)

DoD turns a document into:

1. `page_table.jsonl` (page text + metadata)
2. `toc_tree.json` (hierarchical TOC tree)
3. `image_page_table.jsonl` (raw page images encoded as base64)

Core flow:

1. PDF normalization to page images
2. Text extraction (`pymupdf` / `pymupdf4llm`)
3. TOC generation (PageIndex)
4. Artifact writing (JSON/JSONL + manifest)

## Project Structure

- `src/DoD/cli/` - CLI entrypoint
- `src/DoD/pipeline.py` - end-to-end pipeline orchestration
- `src/DoD/normalize/` - PDF/image normalization to per-page images
- `src/DoD/text_extractor/` - text extraction backends
- `src/DoD/page_table.py` - page table data model + writer
- `src/DoD/pageindex/` - PageIndex TOC builder
- `src/DoD/toc/` - TOC adapters
- `src/DoD/server/` - FastAPI server mode
- `conf/config.yaml` - default configuration
- `examples/` - sample documents

## 0. LLM API Configuration Required

Before running either the CLI package mode or the server mode, set an OpenAI-compatible endpoint and API key:

```bash
export PAGEINDEX_API_KEY="<your_api_key>"
export PAGEINDEX_BASE_URL="<your_openai_compatible_base_url>"
```

Example (Snowflake Cortex):

```bash
export PAGEINDEX_API_KEY="<snowflake_pat>"
export PAGEINDEX_BASE_URL="https://<account-identifier>.snowflakecomputing.com/api/v2/cortex/v1"
```

Then choose any model available on your configured endpoint via `toc.model`.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Install server extras when using API mode:

```bash
uv pip install -e ".[server]"
```

## 1. Run As A Package CLI

Run one document:

```bash
uv run python -m scripts.main \
  input_path=examples/Paa-vej-til-dansk.pdf \
  text_extractor.backend=pymupdf \
  toc.backend=pageindex \
  toc.model=claude-sonnet-4-5
```

Where output is written:

- Hydra run dir: `outputs/<YYYY-MM-DD>/<HH-MM-SS>/`
- Artifacts folder: `outputs/<YYYY-MM-DD>/<HH-MM-SS>/artifacts/`

Main artifact files:

- `page_table.jsonl`
- `image_page_table.jsonl`
- `toc_tree.json`
- `manifest.json`

## 2. Run As A Server

### 2.1 Start server

```bash
export DOD_SERVER_HOST=0.0.0.0
export DOD_SERVER_PORT=8000
export DOD_SERVER_MAX_CONCURRENT_DOCS=4
export DOD_SERVER_WORK_DIR=outputs/server_jobs
uv run python -m scripts.server
```

### 2.2 Health check

```bash
curl http://localhost:8000/healthz
```

### 2.3 Make requests

#### 2.3.1 Single PDF wait for final result

```bash
curl -s -X POST "http://localhost:8000/v1/digest" \
  -F "file=@examples/Paa-vej-til-dansk.pdf" \
  -F "text_extractor_backend=pymupdf" \
  -F "toc_backend=pageindex" \
  -F "toc_model=claude-sonnet-4-5" \
  -F "toc_concurrent_requests=4" \
  > result.json
```

This call blocks until the job is done and writes full result JSON to `result.json`.

#### 2.3.2 Async job submit then poll

Submit:

```bash
JOB_ID=$(curl -s -X POST "http://localhost:8000/v1/digest?wait=false" \
  -F "file=@examples/Paa-vej-til-dansk.pdf" \
  -F "text_extractor_backend=pymupdf" \
  -F "toc_backend=pageindex" \
  -F "toc_model=claude-sonnet-4-5" | jq -r '.job_id')
```

Check status:

```bash
curl -s "http://localhost:8000/v1/jobs/$JOB_ID"
```

Get final result:

```bash
curl -s "http://localhost:8000/v1/jobs/$JOB_ID/result" > result.json
```

#### 2.3.3 Process More Than One PDF At The Same Time

Use async submit (`wait=false`) + parallel curl:

```bash
mkdir -p jobs
printf "%s\n" \
  "/path/to/a.pdf" \
  "/path/to/b.pdf" \
  "/path/to/c.pdf" \
| xargs -I{} -P 3 sh -c '
  name=$(basename "{}" .pdf)
  curl -s -X POST "http://localhost:8000/v1/digest?wait=false" \
    -F "file=@{}" \
    -F "text_extractor_backend=pymupdf" \
    -F "toc_backend=pageindex" \
    -F "toc_model=claude-sonnet-4-5" \
    -F "toc_concurrent_requests=4" \
  > "jobs/${name}.submit.json"
'
```

Poll and download all results:

```bash
for f in jobs/*.submit.json; do
  job_id=$(jq -r '.job_id' "$f")
  name=$(basename "$f" .submit.json)
  until curl -sf "http://localhost:8000/v1/jobs/$job_id/result" > "jobs/${name}.result.json"; do
    sleep 2
  done
done
```

Notes:

- `-P 3` controls how many submit requests run in parallel.
- Server-side processing concurrency is capped by `DOD_SERVER_MAX_CONCURRENT_DOCS`.
- TOC runs in strict mode: if PageIndex fails, the job status becomes `failed` (no fallback TOC).

## 3. Output What And Where

### 3.1 Output in API response JSON

`/v1/digest` (sync) or `/v1/jobs/{job_id}/result` returns:

- `result.toc_tree` - TOC tree JSON
- `result.page_table` - page records (JSON array parsed from JSONL)
- `result.image_page_table` - image records (JSON array parsed from JSONL)
- `result.artifact_paths` - filesystem paths for written artifacts
- `result.manifest` - run metadata + config

Convert arrays back to JSONL if needed:

```bash
jq -c '.result.page_table[]' result.json > page_table.jsonl
jq -c '.result.image_page_table[]' result.json > image_page_table.jsonl
jq '.result.toc_tree' result.json > toc_tree.json
```

### 3.2 Output on disk Server mode

For each job:

- Job folder: `${DOD_SERVER_WORK_DIR}/<job_id>/`
- Input copy: `${DOD_SERVER_WORK_DIR}/<job_id>/input.pdf`
- Artifact folder: `${DOD_SERVER_WORK_DIR}/<job_id>/artifacts/`

Inside `artifacts/`:

- `page_table.jsonl`
- `image_page_table.jsonl`
- `toc_tree.json`
- `manifest.json`

## 4. Request Fields Server `/v1/digest`

Multipart form fields:

- `file` (required, `.pdf`)
- `text_extractor_backend` (optional, recommended: `pymupdf`)
- `normalize_max_pages` (optional int)
- `toc_backend` (optional, typically `pageindex`)
- `toc_model` (optional model name)
- `toc_concurrent_requests` (optional int)
- `toc_check_page_num` (optional int)
- `toc_api_key` (optional per-request override)
- `toc_api_base_url` (optional per-request override)

Query parameter:

- `wait` (default `true`)
  - `true`: request returns when job finishes
  - `false`: request returns immediately with `job_id`
