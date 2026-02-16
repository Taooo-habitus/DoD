---
name: dod-library-qa
description: Use when acting as a Q&A assistant over a personal DoD document library served through DoD MCP.
---

Use this skill when the user asks questions about documents already stored in the DoD server library. Your role is a retrieval-grounded Q&A bot: find the right document, fetch only relevant pages, and answer precisely.

## Bot objective

- Treat DoD server as the user's document library index.
- Ground all answers in retrieved artifacts, not assumptions.
- Prefer minimal retrieval first; expand only when needed.
- Keep outputs focused on user intent, with page-level provenance.

## Available MCP tools

- `list_jobs()` → `{ jobs: [{ file_name, job_id, job_ref, status, created_at }] }`
- `get_toc(job_ref)` → `{ job_id, job_ref, status, toc_tree }`
- `get_page_texts(job_ref, pages)` → `{ requested_pages, returned_pages, pages }`
- `get_page_images(job_ref, pages, mode)` → `{ requested_pages, returned_pages, mode, pages }`
  - Important: this returns **pages as images** (full page render), not embedded images inside a page.

## Core operating rules

1. Resolve document candidates with `list_jobs()` first. Never guess a job.
2. Use `job_ref` as the canonical handle for all retrieval calls.
3. Only use jobs in `succeeded` status for content Q&A.
4. Start with TOC-guided targeting before broad page reads.
5. Always compare `requested_pages` vs `returned_pages`.
6. If partial return, continue retrieval for the remainder.
7. Use image retrieval only when text retrieval is insufficient.
8. Treat page-image retrieval as relatively expensive because it is full-page image payload.
9. Explicitly state document and page basis in final answers.

## Q&A workflow

### Step 1 Resolve the target document

- Call `list_jobs()`.
- Match the user request to `file_name` and recency (`created_at`).
- If ambiguous, ask the user to confirm which document they mean.
- Continue only when one `job_ref` is clearly selected.

### Step 2 Build retrieval plan from structure

- Call `get_toc(job_ref)`.
- Map the user question to likely TOC branches.
- Select smallest likely page ranges first.

### Step 3 Retrieve text incrementally

- Call `get_page_texts(job_ref, pages)`.
- Inspect `requested_pages` and `returned_pages`.
- If partial, call again for missing pages only.
- If evidence is sufficient, answer without fetching more.

### Step 4 Retrieve images only when needed

- Trigger when OCR text is poor or layout/figures/tables require visual inspection.
- Default `mode="path"`; use `mode="base64"` only when explicitly required.
- Remember: `get_page_images` returns the whole page as an image, so avoid broad fetches.

## Pagination handling

Page calls may be truncated by server guardrails.

Example:
- requested: `"100-110"`
- returned: `"100-108"`
- next call: `"109-110"`

Prefer iterative retrieval over broad page sweeps.

## Page spec patterns

- TOC section: `"34-41"`
- Disjoint pages: `"12,27,89-93,140"`
- Remainder after truncation: `"109-110"`

## Failure handling

- Invalid page spec → correct and retry.
- Job missing/stale → rerun `list_jobs()` and re-resolve.
- Job not succeeded → inform user and request rerun/wait.
- Low-quality/empty text → fetch corresponding images.

## Answer format

- State document used: `file_name` + `job_ref`.
- State pages used for evidence.
- Mention if multiple retrieval rounds were needed.
- Answer the user's question directly, then provide supporting citations/pages.
