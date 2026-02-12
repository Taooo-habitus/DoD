---
name: read-long-docs
description: Use when answering questions about previously uploaded documents via DoD MCP while minimizing context usage.
---

Use this skill when a user asks questions about a previously uploaded document and you need to retrieve only relevant sections without overflowing context.

## Available MCP tools

- `list_jobs()`
  - Returns submitted jobs: `file_name`, `job_id`, `job_ref`, `status`, `created_at`.
- `get_toc(job_ref)`
  - Returns document TOC tree.
- `get_page_texts(job_ref, pages)`
  - `pages` accepts flexible specs like `"110,111,89-100"`.
  - Returns `requested_pages`, `returned_pages`, and selected text rows.
- `get_page_images(job_ref, pages, mode)`
  - Same `pages` format; returns selected image rows.

## Core operating rules

1. Never assume the target job. Resolve job first with `list_jobs()`.
2. Prefer `job_ref` over `job_id` in follow-up calls.
3. Always read `requested_pages` vs `returned_pages` after page retrieval.
4. If `returned_pages != requested_pages`, continue with additional calls until covered.
5. Start with TOC, then narrow pages, then fetch text/images only for relevant ranges.
6. Fetch images only when OCR text is insufficient (e.g., diagrams, tables, scanned layout issues).

## Standard workflow

### Step 1: Resolve document

- Call `list_jobs()`.
- Match by `file_name` and `status`.
- If multiple plausible matches exist, ask user to confirm.
- Do not continue until a single `job_ref` is selected.

### Step 2: Confirm job readiness

- If selected job status is not `succeeded`, inform user and wait/retry.

### Step 3: Retrieve structure

- Call `get_toc(job_ref)`.
- Identify likely sections/pages for the user's question.
- Build a minimal initial page request (small focused ranges first).

### Step 4: Retrieve text in chunks

- Call `get_page_texts(job_ref, pages)`.
- Inspect `requested_pages` and `returned_pages`.
- If partial return, request remaining pages in next call(s).
- Summarize and cite page ids while reasoning.

### Step 5: Retrieve images only when needed

- Call `get_page_images(job_ref, pages, mode="path")` by default.
- Use `mode="base64"` only if the downstream consumer needs inline image payloads.

## Pagination strategy

Because retrieval is server-limited, large requests may return partial ranges.

- Example:
  - requested: `"100-110"`
  - returned: `"100-108"`
- Next call should request the remainder (`"109-110"`).

Keep requests tight and iterative rather than broad.

## Good page selection patterns

- TOC section: `"34-41"`
- Disjoint references: `"12,27,89-93,140"`
- Follow-up remainder after partial return: `"109-110"`

## Failure handling

- Invalid page spec: fix format and retry.
- Job not found: re-run `list_jobs()` and re-resolve.
- Job failed: report failure and ask user to re-submit document.
- Empty or low-quality text: fallback to image retrieval for same pages.

## Output style to user

- State which document/job was used (`file_name` + `job_ref`).
- State which pages were analyzed.
- If partial retrieval occurred, state that additional rounds were run.
- Keep final answer focused on the user's question, not tooling details.
