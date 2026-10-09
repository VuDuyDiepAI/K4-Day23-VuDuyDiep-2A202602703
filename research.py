"""Run a complete deep research survey in an ephemeral sandbox."""
import json
import os
import re
import sys
import time
import tempfile
from collections import Counter
from pathlib import Path
from langchain_core.callbacks import BaseCallbackHandler
from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from check_citations import check
from model import make_model
from sandbox import download, open_sandbox, upload
ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"
FAMILIES = {"arxiv", "hf-daily", "hf-search", "web"}

def slugify(topic):
    return re.sub(r"[^\w]+", "-", str(topic).lower(), flags=re.UNICODE).strip("-_")[:60].rstrip("-_") or "topic"

def build_prompt(topic):
    return ("Produce an evidence-grounded English survey of this topic (JSON-encoded): "
            + json.dumps(topic, ensure_ascii=False)
            + ". Follow the full research workflow: at least three parallel researcher tasks, "
              "three source labels, sandbox finalizer and validator, then citation spot checks. "
              "Write the report and sources to the specified sandbox paths.")

def _field(message, key, default=None):
    return message.get(key, default) if isinstance(message, dict) else getattr(message, key, default)

def summarize(messages, elapsed, model_name):
    calls, tokens = Counter(), Counter(input=0, output=0)
    for message in messages:
        for call in _field(message, "tool_calls", []) or []:
            calls[call.get("name", "unknown")] += 1
        usage = _field(message, "usage_metadata", {}) or {}
        tokens["input"] += usage.get("input_tokens", 0) or 0
        tokens["output"] += usage.get("output_tokens", 0) or 0
    return {"model": model_name, "elapsed_s": round(elapsed, 1), "subagent_calls": calls["task"],
            "tool_calls": dict(calls), "tokens": dict(tokens)}

def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes, source_bytes = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report_bytes or not report_bytes.strip() or not source_bytes:
        raise RuntimeError("sandbox did not produce a nonempty report and sources")
    try:
        report_text = report_bytes.decode("utf-8")
        sources = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise RuntimeError("invalid downloaded report or sources JSON") from exc
    problems = check(report_text, sources)
    if problems:
        raise RuntimeError("invalid downloaded citations: " + "; ".join(problems[:5]))
    families = sorted({s.get("source", "") for s in sources})
    if not set(families) <= FAMILIES or len(families) < 3:
        raise RuntimeError("final report needs at least three valid source labels")
    for source in sources:
        family, url, identifier = source["source"], source["url"], source.get("id")
        if family == "arxiv" and (not identifier or url != f"https://arxiv.org/abs/{identifier}" or re.search(r"v\d+$", identifier)):
            raise RuntimeError("invalid arxiv source URL/id")
        if family.startswith("hf-") and (not identifier or url != f"https://huggingface.co/papers/{identifier}"):
            raise RuntimeError(f"invalid Hugging Face source URL/id for source [{source.get('n')}]: id={identifier!r}, url={url!r}")
    meta = {"topic": topic, **summarize(messages, elapsed, model_name), "n_sources": len(sources), "source_families": families}
    if meta["subagent_calls"] < 3:
        raise RuntimeError("lead did not delegate at least three tasks")
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    outputs = {reports_dir / f"{slug}.md": report_bytes,
               reports_dir / f"{slug}.sources.json": source_bytes,
               reports_dir / f"{slug}.meta.json": (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8")}
    # Validate everything before writing; stage files and roll back if publication fails.
    backups = {path: path.read_bytes() if path.exists() else None for path in outputs}
    staged, published = {}, []
    try:
        for path, data in outputs.items():
            with tempfile.NamedTemporaryFile(dir=reports_dir, delete=False) as handle:
                staged[path] = Path(handle.name)
                handle.write(data)
        for path, temporary in staged.items():
            os.replace(temporary, path)
            published.append(path)
    except OSError:
        for path in published:
            if backups[path] is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(backups[path])
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    return reports_dir / f"{slug}.md"

def _execute_ok(backend, command):
    result = backend.execute(command)
    if result.exit_code != 0:
        raise RuntimeError(f"sandbox command failed: {result.output[:1000]}")
    return result.output

class ProgressLogger(BaseCallbackHandler):
    """Show activity without exposing prompts, page content or credentials."""
    def __init__(self):
        self.started = time.monotonic()

    def _log(self, message):
        elapsed = time.monotonic() - self.started
        print(f"[agent {elapsed:.0f}s] {message}", flush=True)

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self._log("Waiting for model response...")

    def on_llm_end(self, response, **kwargs):
        self._log("Model response received.")

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._log("Tool started: " + str((serialized or {}).get("name", "unknown")))

    def on_tool_end(self, output, **kwargs):
        self._log("Tool completed.")

    def on_tool_error(self, error, **kwargs):
        self._log("Tool failed: " + type(error).__name__)

    def on_llm_error(self, error, **kwargs):
        self._log("Model failed: " + type(error).__name__)


def main(topic):
    if not topic.strip():
        print('Usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    try:
        print("[1/6] Configuring model...", flush=True)
        model = make_model()
        start = time.monotonic()
        print("[2/6] Creating sandbox...", flush=True)
        with open_sandbox() as backend:
            print("[3/6] Sandbox ready; preparing files...", flush=True)
            _execute_ok(backend, f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
            _execute_ok(backend, f"test -s {VALIDATOR_PATH} && test -s {FINALIZER_PATH}")
            agent = build_lead_agent(backend, model)
            print("[4/6] Running research agent (may take several minutes)...", flush=True)
            result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]}, config={"recursion_limit": 1000, "callbacks": [ProgressLogger()]})
            # Final validation is performed in sandbox; downloaded bytes are never rewritten.
            print("[5/6] Validating report in sandbox...", flush=True)
            output = _execute_ok(backend, f"python3 {VALIDATOR_PATH}")
            if not output.startswith("OK:"):
                raise RuntimeError("sandbox validator did not confirm OK")
            print("[6/6] Downloading and saving outputs...", flush=True)
            report = save_outputs(backend, topic, result.get("messages", []), time.monotonic() - start,
                                  os.getenv("LAB_MODEL") or os.getenv("OPENAI_DEPLOYMENT_MODEL", "unknown"))
        print(f"Saved {report}")
        return 0
    except Exception as exc:
        from tools import _error
        print("FAILED: " + _error(exc), file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
