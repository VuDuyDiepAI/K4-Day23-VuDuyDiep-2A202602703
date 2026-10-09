"""Host-side research tools with bounded retries and secret redaction."""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote
import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool
load_dotenv()
ARXIV_URL = "https://export.arxiv.org/api/query"
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"
_ARXIV_LOCK = threading.Lock()
_ARXIV_LAST = 0.0

class RetryableError(Exception):
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after

def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            delay = exc.retry_after if exc.retry_after is not None else base * 2**attempt + random.uniform(0, base)
            time.sleep(min(cap, max(0, delay)))

def _request(method, url, **kwargs):
    try:
        response = httpx.request(method, url, timeout=60, follow_redirects=True, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(str(exc)) from exc
    if response.status_code in {429, 500, 502, 503, 504}:
        try:
            delay = float(response.headers.get("Retry-After", ""))
        except ValueError:
            delay = None
        raise RetryableError(f"HTTP {response.status_code}", delay)
    response.raise_for_status()
    return response

def _error(exc):
    message = f"{type(exc).__name__}: {exc}"
    for key in ("EXA_API_KEY", "LAB_API_KEY", "OPENAI_API_KEY", "DAYTONA_API_KEY", "OPENAI_KEY"):
        value = os.getenv(key)
        if value:
            message = message.replace(value, "[REDACTED]").replace(quote(value, safe=""), "[REDACTED]")
    message = re.sub(r"(?i)(exaApiKey=)[^&\s]+", r"\1[REDACTED]", message)
    return "ERROR: " + message

def _text(value):
    return " ".join(str(value or "").split())

def _records(records):
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"

@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv by short keywords, newest first; JSON records with id, url, published, title, summary."""
    try:
        terms = [t for t in re.findall(r"[^\W_]+(?:-[^\W_]+)*", query, re.UNICODE) if t.lower() not in {"and", "or", "all", "ti", "au", "abs"}]
        if not terms:
            return "NO RESULTS"
        def call():
            global _ARXIV_LAST
            with _ARXIV_LOCK:
                wait = 3 - (time.monotonic() - _ARXIV_LAST)
                if wait > 0:
                    time.sleep(wait)
                _ARXIV_LAST = time.monotonic()
                return _request("GET", ARXIV_URL, params={"search_query": " AND ".join("all:" + t for t in terms), "sortBy": "submittedDate", "sortOrder": "descending", "max_results": max(1, min(30, max_results)), "start": 0})
        response = with_retry(call, attempts=7, cap=60)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in ET.fromstring(response.text).findall("a:entry", ns):
            identifier = re.sub(r"v\d+$", "", entry.findtext("a:id", "", ns).split("/abs/")[-1])
            if not identifier or identifier.endswith("errors"):
                continue
            records.append({"id": identifier, "url": "https://arxiv.org/abs/" + identifier, "published": entry.findtext("a:published", "", ns)[:10], "title": _text(entry.findtext("a:title", "", ns)), "summary": _text(entry.findtext("a:summary", "", ns))[:600]})
        return _records(records)
    except Exception as exc:
        return _error(exc)

def _hf(items, search=False):
    records = []
    for item in items:
        paper = item.get("paper") or {}
        identifier = paper.get("id")
        if not identifier:
            continue
        summary = (paper.get("ai_summary") or item.get("ai_summary")) if search else None
        records.append({"id": identifier, "url": "https://huggingface.co/papers/" + identifier, "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10], "title": _text(paper.get("title") or item.get("title")), "summary": _text(summary or paper.get("summary") or item.get("summary"))[:600], "upvotes": paper.get("upvotes") or item.get("upvotes") or 0, "github": paper.get("githubRepo") or item.get("githubRepo") or "", "stars": paper.get("githubStars") or item.get("githubStars") or 0})
    return records

@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Get trending Hugging Face papers, optionally on YYYY-MM-DD; keyword filters titles/summaries; sorted by upvotes."""
    try:
        params = {"limit": max(1, min(100, limit))}
        if date:
            params["date"] = date
        records = _hf(with_retry(lambda: _request("GET", HF_DAILY_URL, params=params)).json())
        if keyword:
            records = [r for r in records if keyword.casefold() in (r["title"] + " " + r["summary"]).casefold()]
        return _records(sorted(records, key=lambda r: r["upvotes"], reverse=True))
    except Exception as exc:
        return _error(exc)

@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic; JSON records include metadata, short summary, votes and GitHub."""
    try:
        if not query.strip():
            return "NO RESULTS"
        response = with_retry(lambda: _request("GET", HF_SEARCH_URL, params={"q": query, "limit": max(1, min(50, limit))}))
        return _records(_hf(response.json(), search=True))
    except Exception as exc:
        return _error(exc)

def _mcp(name, arguments):
    def call():
        key = os.getenv("EXA_API_KEY", "").strip()
        response = _request("POST", EXA_URL, params={"exaApiKey": key} if key else {}, headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}, json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}})
        payloads = []
        if response.text.lstrip().startswith("{"):
            payloads.append(response.json())
        else:
            for event in re.split(r"\r?\n\r?\n", response.text):
                data = "\n".join(line[5:].lstrip() for line in event.splitlines() if line.startswith("data:"))
                if data:
                    payloads.append(json.loads(data))
        payload = next((p for p in reversed(payloads) if "result" in p or "error" in p), None)
        if payload is None:
            raise ValueError("Exa returned no JSON-RPC result")
        if "error" in payload:
            error = json.dumps(payload["error"])
            if re.search(r"rate.?limit|too many requests|429", error, re.I):
                raise RetryableError("Exa rate limited")
            raise ValueError(error)
        result = payload["result"]
        text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
        metadata = json.dumps(result.get("_meta", {}))
        if re.search(r'(rate.?limit[^,}]*true|"[^" ]*(?:rate.?limit|retry)[^" ]*"\s*:\s*(?:true|[1-9]))', metadata, re.I) or re.search(r"rate.?limit(?:ed| exceeded)|too many requests", text, re.I):
            raise RetryableError("Exa rate limited")
        if result.get("isError"):
            raise ValueError(text)
        return text.strip() or "NO RESULTS"
    return with_retry(call, attempts=7, base=2, cap=60)

@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search web with Exa MCP; returns source text and URLs. Describe desired evidence in objective."""
    try:
        if not query.strip():
            return "NO RESULTS"
        return _mcp("web_search_exa", {"query": query, "objective": objective or "Find reliable sources about " + query, "numResults": max(1, min(10, num_results))})
    except Exception as exc:
        return _error(exc)

@tool
def web_fetch(url: str) -> str:
    """Fetch one URL through Exa MCP for evidence; returns text truncated to 12000 characters."""
    try:
        return _mcp("web_fetch_exa", {"urls": [url]})[:12000]
    except Exception as exc:
        return _error(exc)

SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]

if __name__ == "__main__":
    for fn, args in [(arxiv_search, {"query": "world model", "max_results": 3}), (hf_daily_papers, {"limit": 20}), (hf_search_papers, {"query": "world model", "limit": 3}), (web_search, {"query": "survey paper on world models", "num_results": 2}), (web_fetch, {"url": "https://arxiv.org/abs/1803.10122"})]:
        print(f"== {fn.name}\n{fn.invoke(args)[:400]}\n")
