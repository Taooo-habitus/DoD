# DoD (Document Digestion Layer)

DoD turns documents into two artifacts:

1. `page_table.jsonl` — page-level text + metadata
2. `toc_tree.json` — hierarchical TOC tree

The pipeline normalizes documents into images, runs OCR/text extraction, builds a
page table, and then generates a structural tree using PageIndex.

**Project Structure**

- `src/DoD/cli/` — CLI entrypoint
- `src/DoD/pipeline.py` — end-to-end pipeline orchestration
- `src/DoD/normalize/` — PDF/image normalization to per-page images
- `src/DoD/ocr/` — OCR backends
- `src/DoD/page_table.py` — page table data model + writer
- `src/DoD/pageindex/` — PageIndex TOC builder
- `src/DoD/toc/` — TOC adapters
- `conf/config.yaml` — default configuration
- `examples/` — sample documents
- `outputs/` — run outputs (timestamped)

## Setup

Create the virtualenv and install dependencies:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Optional dependencies (install only what you need):

```text
openai       - PageIndex LLM calls
tiktoken     - token counting in PageIndex
PyPDF2       - PDF text extraction (PageIndex)
pymupdf      - optional PDF parser (PageIndex)
pdf2image    - PDF normalization to images (requires poppler)
glmocr       - GLM-OCR SDK backend (requires a model server or MaaS)
transformers - GLM-OCR Transformers backend (requires torch)
pillow       - image loading for Transformers backend
```

## Quickstart

```bash
uv run python -m scripts.main input_path=path/to/document.pdf
```

Artifacts are written under:

```
outputs/<date>/<time>/artifacts/
```

## Configuration

Edit `conf/config.yaml` or override on the command line:

```bash
uv run python -m scripts.main \
  input_path=path/to/document.pdf \
  ocr.backend=plain_text \
  toc.backend=pageindex \
  toc.model=gpt-4o-mini
```

Set your LLM API key via `OPENAI_API_KEY` (or `CHATGPT_API_KEY`). You can also
pass `toc.api_key=...` and `toc.api_base_url=...` directly.

## OCR Backends

### 1) GLM-OCR SDK (`ocr.backend=glm_ocr`)

This uses the GLM-OCR SDK, which expects a model server at
`/v1/chat/completions` unless you have MaaS access.

```bash
uv run python -m scripts.main \
  input_path=examples/Paa-vej-til-dansk.pdf \
  ocr.backend=glm_ocr \
  toc.backend=pageindex \
  toc.model=claude-sonnet-4-5
```

To point at a remote server:

```bash
uv run python -m scripts.main \
  input_path=examples/Paa-vej-til-dansk.pdf \
  ocr.backend=glm_ocr \
  ocr.glmocr_api_host=<SERVER_IP> \
  ocr.glmocr_api_port=8080 \
  toc.backend=pageindex \
  toc.model=claude-sonnet-4-5
```

### 2) GLM-OCR Transformers (`ocr.backend=glm_ocr_transformers`)

Runs the Hugging Face model directly (slow on CPU, works on macOS).

Install:

```bash
uv pip install git+https://github.com/huggingface/transformers.git torch pillow
```

Run:

```bash
uv run python -m scripts.main \
  input_path=examples/Paa-vej-til-dansk.pdf \
  ocr.backend=glm_ocr_transformers \
  ocr.device=cpu \
  toc.backend=pageindex \
  toc.model=claude-sonnet-4-5
```

Limit pages to speed up tests:

```bash
uv run python -m scripts.main \
  input_path=examples/Paa-vej-til-dansk.pdf \
  ocr.backend=glm_ocr_transformers \
  normalize.max_pages=5 \
  toc.backend=pageindex \
  toc.model=claude-sonnet-4-5
```

### 3) Dummy OCR (`ocr.backend=dummy`)

Runs end-to-end without OCR (empty page text). Useful for wiring tests.

## Using Snowflake Cortex (PageIndex)

Snowflake Cortex provides an OpenAI SDK compatible endpoint. Set:

```bash
export OPENAI_API_KEY="<snowflake_pat>"
export OPENAI_BASE_URL="https://<account-identifier>.snowflakecomputing.com/api/v2/cortex/v1"
```

Then run with any `toc.model` supported by your Snowflake account:

```bash
uv run python -m scripts.main \
  input_path=examples/Paa-vej-til-dansk.pdf \
  ocr.backend=glm_ocr_transformers \
  toc.backend=pageindex \
  toc.model=claude-sonnet-4-5
```
