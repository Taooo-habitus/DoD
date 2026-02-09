"""Shared utilities for PageIndex."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import copy
import hashlib
import importlib
import json
import logging
import os
import re
import time
from datetime import datetime
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace as config
from typing import Any, Dict, Iterable, List, Optional

from DoD.pageindex.config import DEFAULT_CONFIG

DEFAULT_API_KEY = (
    os.getenv("PAGEINDEX_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("CHATGPT_API_KEY")
)
DEFAULT_BASE_URL = os.getenv("PAGEINDEX_BASE_URL") or os.getenv("OPENAI_BASE_URL")
_OPENAI_CLIENT_CACHE: Dict[tuple[Optional[str], Optional[str]], Any] = {}
_OPENAI_ASYNC_CLIENT_CACHE: Dict[tuple[Optional[str], Optional[str]], Any] = {}
_REQUEST_API_KEY: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_api_key", default=None
)
_REQUEST_BASE_URL: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_base_url", default=None
)
_REQUEST_SEMAPHORE: contextvars.ContextVar[Optional[asyncio.Semaphore]] = (
    contextvars.ContextVar("request_llm_semaphore", default=None)
)
_REQUEST_LLM_CACHE: contextvars.ContextVar[Optional["LLMCache"]] = (
    contextvars.ContextVar("request_llm_cache", default=None)
)


def set_openai_config(api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Override default OpenAI-compatible client settings."""
    global DEFAULT_API_KEY, DEFAULT_BASE_URL
    if api_key is not None:
        DEFAULT_API_KEY = api_key or None
    if base_url is not None:
        DEFAULT_BASE_URL = base_url.rstrip("/") if base_url else None
    _OPENAI_CLIENT_CACHE.clear()
    _OPENAI_ASYNC_CLIENT_CACHE.clear()


@contextlib.contextmanager
def request_openai_config(
    api_key: Optional[str] = None, base_url: Optional[str] = None
):
    """Temporarily set per-request OpenAI-compatible settings."""
    token_api_key = _REQUEST_API_KEY.set(api_key)
    normalized_base_url = (
        base_url.rstrip("/") if isinstance(base_url, str) else base_url
    )
    token_base_url = _REQUEST_BASE_URL.set(normalized_base_url)
    try:
        yield
    finally:
        _REQUEST_API_KEY.reset(token_api_key)
        _REQUEST_BASE_URL.reset(token_base_url)


@contextlib.contextmanager
def request_llm_concurrency(limit: Optional[int]):
    """Temporarily set a per-request LLM concurrency limit."""
    semaphore: Optional[asyncio.Semaphore] = None
    if isinstance(limit, int) and limit > 0:
        semaphore = asyncio.Semaphore(limit)
    token = _REQUEST_SEMAPHORE.set(semaphore)
    try:
        yield
    finally:
        _REQUEST_SEMAPHORE.reset(token)


@contextlib.contextmanager
def request_llm_cache(cache: Optional["LLMCache"]):
    """Temporarily set a per-request LLM cache."""
    token = _REQUEST_LLM_CACHE.set(cache)
    try:
        yield
    finally:
        _REQUEST_LLM_CACHE.reset(token)


class LLMCache:
    """File-based cache for LLM responses."""

    def __init__(self, cache_dir: str | Path):
        """Create a cache rooted at the given directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Return cached payload for the key if present."""
        path = self._key_to_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set(self, key: str, payload: Dict[str, Any]) -> None:
        """Persist payload for the given key."""
        path = self._key_to_path(key)
        path.write_text(json.dumps(payload), encoding="utf-8")


def _hash_prompt(
    model: str, prompt: str, chat_history: Optional[List[Dict[str, str]]]
) -> str:
    hasher = hashlib.sha256()
    hasher.update(model.encode("utf-8"))
    hasher.update(b"\n")
    if chat_history:
        hasher.update(json.dumps(chat_history, sort_keys=True).encode("utf-8"))
        hasher.update(b"\n")
    hasher.update(prompt.encode("utf-8"))
    return hasher.hexdigest()


def _resolve_api_key(api_key: Optional[str]) -> str:
    """Resolve API key for OpenAI-compatible SDK clients."""
    resolved_api_key = api_key
    if resolved_api_key is None:
        resolved_api_key = _REQUEST_API_KEY.get()
    if resolved_api_key is None:
        resolved_api_key = DEFAULT_API_KEY
    # Some OpenAI-compatible endpoints do not require auth, but the SDK expects a key.
    return resolved_api_key or "EMPTY"


def _resolve_base_url(base_url: Optional[str]) -> Optional[str]:
    """Resolve and normalize OpenAI-compatible base URL."""
    resolved_base_url = base_url
    if resolved_base_url is None:
        resolved_base_url = _REQUEST_BASE_URL.get()
    if resolved_base_url is None:
        resolved_base_url = DEFAULT_BASE_URL
    if isinstance(resolved_base_url, str):
        return resolved_base_url.rstrip("/")
    return resolved_base_url


def _get_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Create a synchronous OpenAI-compatible client."""
    try:
        openai = importlib.import_module("openai")
    except ImportError as exc:
        raise RuntimeError("openai is required for PageIndex LLM calls.") from exc

    resolved_api_key = _resolve_api_key(api_key)
    resolved_base_url = _resolve_base_url(base_url)
    cache_key = (resolved_api_key, resolved_base_url)
    client = _OPENAI_CLIENT_CACHE.get(cache_key)
    if client is not None:
        return client

    if resolved_base_url:
        client = openai.OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
    else:
        client = openai.OpenAI(api_key=resolved_api_key)
    _OPENAI_CLIENT_CACHE[cache_key] = client
    return client


def _get_openai_async_client(
    api_key: Optional[str] = None, base_url: Optional[str] = None
):
    """Create an async OpenAI-compatible client."""
    try:
        openai = importlib.import_module("openai")
    except ImportError as exc:
        raise RuntimeError("openai is required for PageIndex LLM calls.") from exc

    resolved_api_key = _resolve_api_key(api_key)
    resolved_base_url = _resolve_base_url(base_url)
    cache_key = (resolved_api_key, resolved_base_url)
    client = _OPENAI_ASYNC_CLIENT_CACHE.get(cache_key)
    if client is not None:
        return client

    if resolved_base_url:
        client = openai.AsyncOpenAI(
            api_key=resolved_api_key, base_url=resolved_base_url
        )
    else:
        client = openai.AsyncOpenAI(api_key=resolved_api_key)
    _OPENAI_ASYNC_CLIENT_CACHE[cache_key] = client
    return client


@lru_cache(maxsize=1)
def _require_tiktoken():
    """Import and cache the tiktoken module."""
    try:
        return importlib.import_module("tiktoken")
    except ImportError as exc:
        raise RuntimeError("tiktoken is required for token counting.") from exc


@lru_cache(maxsize=64)
def _get_encoding_for_model(model: Optional[str]):
    """Return a cached tokenizer encoding."""
    tiktoken = _require_tiktoken()
    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    return tiktoken.get_encoding("cl100k_base")


def _require_pypdf2():
    try:
        return importlib.import_module("PyPDF2")
    except ImportError as exc:
        raise RuntimeError("PyPDF2 is required for PDF parsing.") from exc


def _require_pymupdf():
    try:
        return importlib.import_module("pymupdf")
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF parsing.") from exc


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Count tokens for the given text with the model's tokenizer."""
    if not text:
        return 0
    enc = _get_encoding_for_model(model)
    tokens = enc.encode(text)
    return len(tokens)


def ChatGPT_API_with_finish_reason(
    model: Optional[str],
    prompt: str,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> tuple[str, str]:
    """Call a model and return content with finish reason."""
    if model is None:
        raise ValueError("model is required for LLM calls.")
    max_retries = 10
    client = _get_openai_client(api_key=api_key, base_url=api_base_url)
    cache = _REQUEST_LLM_CACHE.get()
    for i in range(max_retries):
        try:
            if chat_history:
                messages = [*chat_history, {"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": prompt}]

            cache_key = None
            if cache and model:
                cache_key = _hash_prompt(model, prompt, chat_history)
                cached = cache.get(cache_key)
                if (
                    cached
                    and cached.get("finish_reason")
                    and cached.get("content") is not None
                ):
                    return cached["content"], cached["finish_reason"]

            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0
            )
            if response.choices[0].finish_reason == "length":
                content = response.choices[0].message.content
                finish_reason = "max_output_reached"
            else:
                content = response.choices[0].message.content
                finish_reason = "finished"
            if cache and cache_key:
                cache.set(
                    cache_key, {"content": content, "finish_reason": finish_reason}
                )
            return content, finish_reason

        except Exception as exc:
            logging.error("Retrying LLM call after error: %s", exc)
            if i < max_retries - 1:
                time.sleep(1)
            else:
                logging.error("Max retries reached for prompt.")
                return "", "error"
    return "", "error"


def ChatGPT_API(
    model: Optional[str],
    prompt: str,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Call a model and return the response content."""
    if model is None:
        raise ValueError("model is required for LLM calls.")
    max_retries = 10
    client = _get_openai_client(api_key=api_key, base_url=api_base_url)
    cache = _REQUEST_LLM_CACHE.get()
    for i in range(max_retries):
        try:
            if chat_history:
                messages = [*chat_history, {"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": prompt}]

            cache_key = None
            if cache and model:
                cache_key = _hash_prompt(model, prompt, chat_history)
                cached = cache.get(cache_key)
                if cached and cached.get("content") is not None:
                    return cached["content"]

            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0
            )

            content = response.choices[0].message.content
            if cache and cache_key:
                cache.set(cache_key, {"content": content})
            return content
        except Exception as exc:
            logging.error("Retrying LLM call after error: %s", exc)
            if i < max_retries - 1:
                time.sleep(1)
            else:
                logging.error("Max retries reached for prompt.")
                return "Error"
    return "Error"


async def ChatGPT_API_with_finish_reason_async(
    model: Optional[str],
    prompt: str,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> tuple[str, str]:
    """Call a model asynchronously and return content with finish reason."""
    if model is None:
        raise ValueError("model is required for LLM calls.")
    max_retries = 10
    client = _get_openai_async_client(api_key=api_key, base_url=api_base_url)
    semaphore = _REQUEST_SEMAPHORE.get()
    cache = _REQUEST_LLM_CACHE.get()
    for i in range(max_retries):
        try:
            if chat_history:
                messages = [*chat_history, {"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": prompt}]

            cache_key = None
            if cache and model:
                cache_key = _hash_prompt(model, prompt, chat_history)
                cached = cache.get(cache_key)
                if (
                    cached
                    and cached.get("finish_reason")
                    and cached.get("content") is not None
                ):
                    return cached["content"], cached["finish_reason"]

            if semaphore:
                async with semaphore:
                    response = await client.chat.completions.create(
                        model=model, messages=messages, temperature=0
                    )
            else:
                response = await client.chat.completions.create(
                    model=model, messages=messages, temperature=0
                )
            if response.choices[0].finish_reason == "length":
                content = response.choices[0].message.content
                finish_reason = "max_output_reached"
            else:
                content = response.choices[0].message.content
                finish_reason = "finished"
            if cache and cache_key:
                cache.set(
                    cache_key, {"content": content, "finish_reason": finish_reason}
                )
            return content, finish_reason
        except Exception as exc:
            logging.error("Retrying async LLM call after error: %s", exc)
            if i < max_retries - 1:
                await asyncio.sleep(1)
            else:
                logging.error("Max retries reached for prompt.")
                return "", "error"
    return "", "error"


async def ChatGPT_API_async(
    model: Optional[str],
    prompt: str,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Call a model asynchronously and return the response content."""
    if model is None:
        raise ValueError("model is required for LLM calls.")
    max_retries = 10
    client = _get_openai_async_client(api_key=api_key, base_url=api_base_url)
    semaphore = _REQUEST_SEMAPHORE.get()
    cache = _REQUEST_LLM_CACHE.get()
    for i in range(max_retries):
        try:
            if chat_history:
                messages = [*chat_history, {"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": prompt}]

            cache_key = None
            if cache and model:
                cache_key = _hash_prompt(model, prompt, chat_history)
                cached = cache.get(cache_key)
                if cached and cached.get("content") is not None:
                    return cached["content"]

            if semaphore:
                async with semaphore:
                    response = await client.chat.completions.create(
                        model=model, messages=messages, temperature=0
                    )
            else:
                response = await client.chat.completions.create(
                    model=model, messages=messages, temperature=0
                )
            content = response.choices[0].message.content
            if cache and cache_key:
                cache.set(cache_key, {"content": content})
            return content
        except Exception as exc:
            logging.error("Retrying async LLM call after error: %s", exc)
            if i < max_retries - 1:
                await asyncio.sleep(1)
            else:
                logging.error("Max retries reached for prompt.")
                return "Error"
    return "Error"


def get_json_content(response: str) -> str:
    """Strip code fences and return JSON-like content."""
    start_idx = response.find("```json")
    if start_idx != -1:
        start_idx += 7
        response = response[start_idx:]

    end_idx = response.rfind("```")
    if end_idx != -1:
        response = response[:end_idx]

    json_content = response.strip()
    return json_content


def extract_json(content: str) -> Dict[str, Any]:
    """Extract JSON from a string that may include code fences."""
    try:
        # First, try to extract JSON enclosed within ```json and ```
        start_idx = content.find("```json")
        if start_idx != -1:
            start_idx += 7  # Adjust index to start after the delimiter
            end_idx = content.rfind("```")
            json_content = content[start_idx:end_idx].strip()
        else:
            # If no delimiters, assume entire content could be JSON
            json_content = content.strip()

        # Clean up common issues that might cause parsing errors
        json_content = json_content.replace(
            "None", "null"
        )  # Replace Python None with JSON null
        json_content = json_content.replace("\n", " ").replace(
            "\r", " "
        )  # Remove newlines
        json_content = " ".join(json_content.split())  # Normalize whitespace

        # Attempt to parse and return the JSON object
        return json.loads(json_content)
    except json.JSONDecodeError as exc:
        logging.error("Failed to extract JSON: %s", exc)
        # Try to clean up the content further if initial parsing fails
        try:
            # Remove any trailing commas before closing brackets/braces
            json_content = json_content.replace(",]", "]").replace(",}", "}")
            return json.loads(json_content)
        except Exception as exc:
            logging.error("Failed to parse JSON even after cleanup: %s", exc)
            return {}
    except Exception as exc:
        logging.error("Unexpected error while extracting JSON: %s", exc)
        return {}


def write_node_id(data: Any, node_id: int = 0) -> int:
    """Assign zero-padded node ids to a tree structure in-place."""
    if isinstance(data, dict):
        data["node_id"] = str(node_id).zfill(4)
        node_id += 1
        for key in list(data.keys()):
            if "nodes" in key:
                node_id = write_node_id(data[key], node_id)
    elif isinstance(data, list):
        for index in range(len(data)):
            node_id = write_node_id(data[index], node_id)
    return node_id


def get_nodes(structure: Any) -> List[Dict[str, Any]]:
    """Flatten a structure into a list of nodes without child pointers."""
    if isinstance(structure, dict):
        structure_node = copy.deepcopy(structure)
        structure_node.pop("nodes", None)
        nodes = [structure_node]
        for key in list(structure.keys()):
            if "nodes" in key:
                nodes.extend(get_nodes(structure[key]))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(get_nodes(item))
        return nodes
    return []


def structure_to_list(structure: Any) -> List[Dict[str, Any]]:
    """Flatten a structure into a list of nodes, preserving children."""
    if isinstance(structure, dict):
        nodes = []
        nodes.append(structure)
        if "nodes" in structure:
            nodes.extend(structure_to_list(structure["nodes"]))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(structure_to_list(item))
        return nodes
    return []


def get_leaf_nodes(structure: Any) -> List[Dict[str, Any]]:
    """Return leaf nodes from a structure."""
    if isinstance(structure, dict):
        if not structure["nodes"]:
            structure_node = copy.deepcopy(structure)
            structure_node.pop("nodes", None)
            return [structure_node]
        else:
            leaf_nodes = []
            for key in list(structure.keys()):
                if "nodes" in key:
                    leaf_nodes.extend(get_leaf_nodes(structure[key]))
            return leaf_nodes
    elif isinstance(structure, list):
        leaf_nodes = []
        for item in structure:
            leaf_nodes.extend(get_leaf_nodes(item))
        return leaf_nodes
    return []


def is_leaf_node(data: Any, node_id: str) -> bool:
    """Check whether a node id refers to a leaf node."""

    # Helper function to find the node by its node_id
    def find_node(data: Any, node_id: str) -> Optional[Dict[str, Any]]:
        if isinstance(data, dict):
            if data.get("node_id") == node_id:
                return data
            for key in data.keys():
                if "nodes" in key:
                    result = find_node(data[key], node_id)
                    if result:
                        return result
        elif isinstance(data, list):
            for item in data:
                result = find_node(item, node_id)
                if result:
                    return result
        return None

    # Find the node with the given node_id
    node = find_node(data, node_id)

    # Check if the node is a leaf node
    if node and not node.get("nodes"):
        return True
    return False


def get_last_node(structure: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the last node in a list."""
    return structure[-1]


def extract_text_from_pdf(pdf_path: str | BytesIO) -> str:
    """Extract concatenated text from a PDF."""
    pdf_reader = _require_pypdf2().PdfReader(pdf_path)
    ###return text not list
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


def get_pdf_title(pdf_path: str | BytesIO) -> str:
    """Get the title metadata from a PDF."""
    pdf_reader = _require_pypdf2().PdfReader(pdf_path)
    meta = pdf_reader.metadata
    title = meta.title if meta and meta.title else "Untitled"
    return title


def get_text_of_pages(
    pdf_path: str | BytesIO, start_page: int, end_page: int, tag: bool = True
) -> str:
    """Extract text for a range of pages."""
    pdf_reader = _require_pypdf2().PdfReader(pdf_path)
    text = ""
    for page_num in range(start_page - 1, end_page):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        if tag:
            text += f"<start_index_{page_num + 1}>\n{page_text}\n<end_index_{page_num + 1}>\n"
        else:
            text += page_text
    return text


def get_first_start_page_from_text(text: str) -> int:
    """Find the first <start_index_N> tag in a string."""
    start_page = -1
    start_page_match = re.search(r"<start_index_(\d+)>", text)
    if start_page_match:
        start_page = int(start_page_match.group(1))
    return start_page


def get_last_start_page_from_text(text: str) -> int:
    """Find the last <start_index_N> tag in a string."""
    start_page = -1
    # Find all matches of start_index tags
    start_page_matches = re.finditer(r"<start_index_(\d+)>", text)
    # Convert iterator to list and get the last match if any exist
    matches_list = list(start_page_matches)
    if matches_list:
        start_page = int(matches_list[-1].group(1))
    return start_page


def sanitize_filename(filename: str, replacement: str = "-") -> str:
    """Replace forbidden filename characters."""
    # In Linux, only '/' and '\0' (null) are invalid in filenames.
    # Null can't be represented in strings, so we only handle '/'.
    return filename.replace("/", replacement)


def get_pdf_name(pdf_path: str | BytesIO) -> str:
    """Derive a PDF name from path or metadata."""
    # Extract PDF name
    if isinstance(pdf_path, str):
        pdf_name = os.path.basename(pdf_path)
    elif isinstance(pdf_path, BytesIO):
        pdf_reader = _require_pypdf2().PdfReader(pdf_path)
        meta = pdf_reader.metadata
        pdf_name = meta.title if meta and meta.title else "Untitled"
        pdf_name = sanitize_filename(pdf_name)
    return pdf_name


class JsonLogger:
    """Simple JSON logger for PageIndex runs."""

    def __init__(self, file_path: str | BytesIO):
        """Create a logger scoped to a document."""
        # Extract PDF name for logger name
        pdf_name = get_pdf_name(file_path)

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{pdf_name}_{current_time}.json"
        os.makedirs("./logs", exist_ok=True)
        # Initialize empty list to store all messages
        self.log_data = []

    def log(self, level: str, message: Any, **kwargs: Any) -> None:
        """Persist a log record."""
        level_value = getattr(logging, level, logging.INFO)
        logging.getLogger("DoD.pageindex").log(level_value, message)
        if isinstance(message, dict):
            self.log_data.append(message)
        else:
            self.log_data.append({"message": message})
        # Add new message to the log data

        # Write entire log data to file
        with open(self._filepath(), "w") as f:
            json.dump(self.log_data, f, indent=2)

    def info(self, message: Any, **kwargs: Any) -> None:
        """Log an info-level message."""
        self.log("INFO", message, **kwargs)

    def error(self, message: Any, **kwargs: Any) -> None:
        """Log an error-level message."""
        self.log("ERROR", message, **kwargs)

    def warning(self, message: Any, **kwargs: Any) -> None:
        """Log a warning-level message."""
        self.log("WARNING", message, **kwargs)

    def debug(self, message: Any, **kwargs: Any) -> None:
        """Log a debug-level message."""
        self.log("DEBUG", message, **kwargs)

    def exception(self, message: Any, **kwargs: Any) -> None:
        """Log an exception-level message."""
        kwargs["exception"] = True
        self.log("ERROR", message, **kwargs)

    def _filepath(self) -> str:
        """Return the log file path."""
        return os.path.join("logs", self.filename)


def list_to_tree(data):
    """Convert a flat TOC list into a nested tree."""

    def get_parent_structure(structure):
        """Helper function to get the parent structure code."""
        if not structure:
            return None
        parts = str(structure).split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else None

    # First pass: Create nodes and track parent-child relationships
    nodes = {}
    root_nodes = []

    for item in data:
        structure = item.get("structure")
        node = {
            "title": item.get("title"),
            "start_index": item.get("start_index"),
            "end_index": item.get("end_index"),
            "nodes": [],
        }

        nodes[structure] = node

        # Find parent
        parent_structure = get_parent_structure(structure)

        if parent_structure:
            # Add as child to parent if parent exists
            if parent_structure in nodes:
                nodes[parent_structure]["nodes"].append(node)
            else:
                root_nodes.append(node)
        else:
            # No parent, this is a root node
            root_nodes.append(node)

    # Helper function to clean empty children arrays
    def clean_node(node):
        if not node["nodes"]:
            del node["nodes"]
        else:
            for child in node["nodes"]:
                clean_node(child)
        return node

    # Clean and return the tree
    return [clean_node(node) for node in root_nodes]


def add_preface_if_needed(data: Any) -> Any:
    """Insert a preface node when the first section starts after page 1."""
    if not isinstance(data, list) or not data:
        return data

    if data[0]["physical_index"] is not None and data[0]["physical_index"] > 1:
        preface_node = {"structure": "0", "title": "Preface", "physical_index": 1}
        data.insert(0, preface_node)
    return data


def get_page_tokens(
    pdf_path: str | BytesIO,
    model: str = "gpt-4o-2024-11-20",
    pdf_parser: str = "PyPDF2",
) -> List[tuple[str, int]]:
    """Extract (page_text, token_count) pairs from a PDF."""
    enc = _get_encoding_for_model(model)
    if pdf_parser == "PyPDF2":
        pdf_reader = _require_pypdf2().PdfReader(pdf_path)
        page_list = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            token_length = len(enc.encode(page_text))
            page_list.append((page_text, token_length))
        return page_list
    elif pdf_parser == "PyMuPDF":
        pymupdf = _require_pymupdf()
        if isinstance(pdf_path, BytesIO):
            doc = pymupdf.open(stream=pdf_path, filetype="pdf")
        elif isinstance(pdf_path, str) and os.path.isfile(pdf_path):
            doc = pymupdf.open(pdf_path)
        else:
            raise ValueError("Unsupported pdf_path for PyMuPDF parser.")
        page_list = []
        for page in doc:
            page_text = page.get_text()
            token_length = len(enc.encode(page_text))
            page_list.append((page_text, token_length))
        return page_list
    else:
        raise ValueError(f"Unsupported PDF parser: {pdf_parser}")


def get_text_of_pdf_pages(
    pdf_pages: List[tuple[str, int]], start_page: int, end_page: int
) -> str:
    """Return concatenated text for a page range."""
    chunks = [pdf_pages[page_num][0] for page_num in range(start_page - 1, end_page)]
    return "".join(chunks)


def get_text_of_pdf_pages_with_labels(
    pdf_pages: List[tuple[str, int]], start_page: int, end_page: int
) -> str:
    """Return labeled text blocks for a page range."""
    chunks = []
    for page_num in range(start_page - 1, end_page):
        chunks.append(
            f"<physical_index_{page_num + 1}>\n{pdf_pages[page_num][0]}\n<physical_index_{page_num + 1}>\n"
        )
    return "".join(chunks)


def get_number_of_pages(pdf_path: str | BytesIO) -> int:
    """Return number of pages in a PDF."""
    pdf_reader = _require_pypdf2().PdfReader(pdf_path)
    num = len(pdf_reader.pages)
    return num


def post_processing(
    structure: List[Dict[str, Any]], end_physical_index: int
) -> List[Dict[str, Any]]:
    """Compute start/end indices and build a tree."""
    # First convert page_number to start_index in flat list
    for i, item in enumerate(structure):
        item["start_index"] = item.get("physical_index")
        if i < len(structure) - 1:
            if structure[i + 1].get("appear_start") == "yes":
                item["end_index"] = structure[i + 1]["physical_index"] - 1
            else:
                item["end_index"] = structure[i + 1]["physical_index"]
        else:
            item["end_index"] = end_physical_index
    tree = list_to_tree(structure)
    if len(tree) != 0:
        return tree
    else:
        ### remove appear_start
        for node in structure:
            node.pop("appear_start", None)
            node.pop("physical_index", None)
        return structure


def clean_structure_post(data: Any) -> Any:
    """Remove post-processing fields from a structure."""
    if isinstance(data, dict):
        data.pop("page_number", None)
        data.pop("start_index", None)
        data.pop("end_index", None)
        if "nodes" in data:
            clean_structure_post(data["nodes"])
    elif isinstance(data, list):
        for section in data:
            clean_structure_post(section)
    return data


def remove_fields(data: Any, fields: Optional[List[str]] = None) -> Any:
    """Remove fields from a nested structure."""
    if fields is None:
        fields = ["text"]
    if isinstance(data, dict):
        return {k: remove_fields(v, fields) for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data


def print_toc(tree: List[Dict[str, Any]], indent: int = 0) -> None:
    """Print a table of contents to stdout."""
    for node in tree:
        print("  " * indent + node["title"])
        if node.get("nodes"):
            print_toc(node["nodes"], indent + 1)


def print_json(data: Any, max_len: int = 40, indent: int = 2) -> None:
    """Pretty-print a JSON-like object with truncated strings."""

    def simplify_data(obj):
        if isinstance(obj, dict):
            return {k: simplify_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [simplify_data(item) for item in obj]
        elif isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + "..."
        else:
            return obj

    simplified = simplify_data(data)
    print(json.dumps(simplified, indent=indent, ensure_ascii=False))


def remove_structure_text(data: Any) -> Any:
    """Remove text fields from a structure."""
    if isinstance(data, dict):
        data.pop("text", None)
        if "nodes" in data:
            remove_structure_text(data["nodes"])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


def check_token_limit(structure: Any, limit: int = 110000) -> None:
    """Print nodes that exceed a token limit."""
    nodes = structure_to_list(structure)
    for node in nodes:
        num_tokens = count_tokens(node["text"], model="gpt-4o")
        if num_tokens > limit:
            print(f"Node ID: {node['node_id']} has {num_tokens} tokens")
            print("Start Index:", node["start_index"])
            print("End Index:", node["end_index"])
            print("Title:", node["title"])
            print("\n")


def convert_physical_index_to_int(data: Any) -> Any:
    """Normalize physical index fields into integers."""
    if isinstance(data, list):
        for i in range(len(data)):
            # Check if item is a dictionary and has 'physical_index' key
            if isinstance(data[i], dict) and "physical_index" in data[i]:
                if isinstance(data[i]["physical_index"], str):
                    if data[i]["physical_index"].startswith("<physical_index_"):
                        data[i]["physical_index"] = int(
                            data[i]["physical_index"].split("_")[-1].rstrip(">").strip()
                        )
                    elif data[i]["physical_index"].startswith("physical_index_"):
                        data[i]["physical_index"] = int(
                            data[i]["physical_index"].split("_")[-1].strip()
                        )
    elif isinstance(data, str):
        if data.startswith("<physical_index_"):
            data = int(data.split("_")[-1].rstrip(">").strip())
        elif data.startswith("physical_index_"):
            data = int(data.split("_")[-1].strip())
        # Check data is int
        if isinstance(data, int):
            return data
        else:
            return None
    return data


def convert_page_to_int(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert page values to integers where possible."""
    for item in data:
        if "page" in item and isinstance(item["page"], str):
            try:
                item["page"] = int(item["page"])
            except ValueError:
                # Keep original value if conversion fails
                pass
    return data


def add_node_text(node: Any, pdf_pages: List[tuple[str, int]]) -> None:
    """Attach page text to nodes."""
    if isinstance(node, dict):
        start_page = node.get("start_index")
        end_page = node.get("end_index")
        node["text"] = get_text_of_pdf_pages(pdf_pages, start_page, end_page)
        if "nodes" in node:
            add_node_text(node["nodes"], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text(node[index], pdf_pages)
    return


def add_node_text_with_labels(node: Any, pdf_pages: List[tuple[str, int]]) -> None:
    """Attach labeled page text to nodes."""
    if isinstance(node, dict):
        start_page = node.get("start_index")
        end_page = node.get("end_index")
        node["text"] = get_text_of_pdf_pages_with_labels(
            pdf_pages, start_page, end_page
        )
        if "nodes" in node:
            add_node_text_with_labels(node["nodes"], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text_with_labels(node[index], pdf_pages)
    return


async def generate_node_summary(
    node: Dict[str, Any], model: Optional[str] = None
) -> str:
    """Generate a summary for a node."""
    prompt = f"""You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

    Partial Document Text: {node["text"]}

    Directly return the description, do not include any other text.
    """
    response = await ChatGPT_API_async(model, prompt)
    return response


async def generate_summaries_for_structure(
    structure: Any, model: Optional[str] = None
) -> Any:
    """Generate summaries for all nodes in a structure."""
    nodes = structure_to_list(structure)
    tasks = [generate_node_summary(node, model=model) for node in nodes]
    summaries = await asyncio.gather(*tasks)

    for node, summary in zip(nodes, summaries):
        node["summary"] = summary
    return structure


def create_clean_structure_for_description(structure: Any) -> Any:
    """Create a clean structure for description generation."""
    if isinstance(structure, dict):
        clean_node = {}
        # Only include essential fields for description
        for key in ["title", "node_id", "summary", "prefix_summary"]:
            if key in structure:
                clean_node[key] = structure[key]

        # Recursively process child nodes
        if "nodes" in structure and structure["nodes"]:
            clean_node["nodes"] = create_clean_structure_for_description(
                structure["nodes"]
            )

        return clean_node
    elif isinstance(structure, list):
        return [create_clean_structure_for_description(item) for item in structure]
    else:
        return structure


async def generate_doc_description(structure: Any, model: Optional[str] = None) -> str:
    """Generate a one-sentence description for the document."""
    prompt = f"""Your are an expert in generating descriptions for a document.
    You are given a structure of a document. Your task is to generate a one-sentence description for the document, which makes it easy to distinguish the document from other documents.

    Document Structure: {structure}

    Directly return the description, do not include any other text.
    """
    response = await ChatGPT_API_async(model, prompt)
    return response


def reorder_dict(data: Dict[str, Any], key_order: Iterable[str]) -> Dict[str, Any]:
    """Reorder keys in a dict according to a list."""
    if not key_order:
        return data
    return {key: data[key] for key in key_order if key in data}


def format_structure(structure: Any, order: Optional[List[str]] = None) -> Any:
    """Reorder structure keys and clean empty nodes."""
    if not order:
        return structure
    if isinstance(structure, dict):
        if "nodes" in structure:
            structure["nodes"] = format_structure(structure["nodes"], order)
        if not structure.get("nodes"):
            structure.pop("nodes", None)
        structure = reorder_dict(structure, order)
    elif isinstance(structure, list):
        structure = [format_structure(item, order) for item in structure]
    return structure


class ConfigLoader:
    """Merge user config with defaults."""

    def __init__(self, default_dict: Optional[Dict[str, Any]] = None):
        """Initialize with a default configuration dictionary."""
        self._default_dict = default_dict or DEFAULT_CONFIG

    def _validate_keys(self, user_dict: Dict[str, Any]) -> None:
        """Validate config keys against defaults."""
        unknown_keys = set(user_dict) - set(self._default_dict)
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {unknown_keys}")

    def load(self, user_opt: Optional[Dict[str, Any]] = None) -> config:
        """Load the configuration, merging user options with defaults."""
        if user_opt is None:
            user_dict = {}
        elif isinstance(user_opt, config):
            user_dict = vars(user_opt)
        elif isinstance(user_opt, dict):
            user_dict = user_opt
        else:
            raise TypeError("user_opt must be dict, config(SimpleNamespace) or None")

        self._validate_keys(user_dict)
        merged = {**self._default_dict, **user_dict}
        return config(**merged)
