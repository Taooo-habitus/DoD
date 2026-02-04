# DoD

Document Digestion Layer.

## Overview

This project turns documents into two persistent artifacts:

1. A page-level text table (JSONL)
2. A hierarchical TOC tree (JSON)

The pipeline normalizes documents into images, performs OCR, builds a page table,
and then generates a structural tree using PageIndex.

## Quickstart

```bash
uv run python -m scripts.main input_path=path/to/document.pdf
```

Artifacts are written under `outputs/<date>/<time>/artifacts/`.

## Configuration

Edit `conf/config.yaml` or override on the command line:

```bash
uv run python -m scripts.main \
  input_path=path/to/document.md \
  ocr.backend=plain_text \
  toc.backend=pageindex \
  toc.model=gpt-4o-mini \
  toc.api_base_url=https://api.openai.com/v1
```

Set your API key via `OPENAI_API_KEY` (or `CHATGPT_API_KEY`). You can also pass
`toc.api_key=...` directly.

## Optional Dependencies

Some backends require extra packages:

```text
openai       - PageIndex LLM calls
tiktoken     - token counting in PageIndex
PyPDF2       - PDF text extraction (PageIndex)
pymupdf      - optional PDF parser (PageIndex)
pdf2image    - PDF normalization to images
glmocr       - GLM-OCR SDK backend (requires server or MaaS)
transformers - GLM-OCR backend dependencies (requires torch)
```
