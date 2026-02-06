# DoD (Document Digestion Layer)

DoD turns documents into two artifacts:

1. `page_table.jsonl` — page-level text + metadata
2. `toc_tree.json` — hierarchical TOC tree
3. `image_page_table.jsonl` — page images encoded as base64

The pipeline normalizes documents into images, runs text extraction, builds a
page table, and then generates a structural tree using PageIndex.

**Project Structure**

- `src/DoD/cli/` — CLI entrypoint
- `src/DoD/pipeline.py` — end-to-end pipeline orchestration
- `src/DoD/normalize/` — PDF/image normalization to per-page images
- `src/DoD/text_extractor/` — text extraction backends
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
pymupdf4llm  - PyMuPDF text extraction backend
pdf2image    - PDF normalization to images (requires poppler)
tqdm         - text extraction progress bars
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
  text_extractor.backend=pymupdf \
  toc.backend=pageindex \
  toc.model=gpt-4o-mini
```

Set your LLM API key via `PAGEINDEX_API_KEY` (or `OPENAI_API_KEY` / `CHATGPT_API_KEY`).
Set endpoint via `PAGEINDEX_BASE_URL` (or `OPENAI_BASE_URL`). You can also pass
`toc.api_key=...` and `toc.api_base_url=...` directly.

## Text Extraction

### PyMuPDF (`text_extractor.backend=pymupdf`)

Extracts text directly from PDFs with PyMuPDF (via `pymupdf4llm`), while still
keeping normalized page images for downstream encoding.

```bash
uv pip install pymupdf4llm
uv run python -m scripts.main \
  input_path=examples/Paa-vej-til-dansk.pdf \
  text_extractor.backend=pymupdf \
  toc.backend=pageindex \
  toc.model=claude-sonnet-4-5
```

## Using Snowflake Cortex (PageIndex)

Snowflake Cortex provides an OpenAI SDK compatible endpoint. Set:

```bash
export PAGEINDEX_API_KEY="<snowflake_pat>"
export PAGEINDEX_BASE_URL="https://<account-identifier>.snowflakecomputing.com/api/v2/cortex/v1"
```

Then run with any `toc.model` supported by your Snowflake account:

```bash
uv run python -m scripts.main \
  input_path=examples/Paa-vej-til-dansk.pdf \
  text_extractor.backend=pymupdf \
  toc.backend=pageindex \
  toc.model=claude-sonnet-4-5
```
