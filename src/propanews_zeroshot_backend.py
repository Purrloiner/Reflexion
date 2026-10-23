"""Reuse llm.call_llm and its settings; add explicit no-truncation preflight."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import llm


def runtime_metadata() -> dict:
    packages = {}
    for name in ("torch", "transformers", "accelerate", "tokenizers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "executable": sys.executable, "packages": packages}


def model_inventory() -> list[dict]:
    root = Path(llm.MODEL_PATH)
    if not root.is_dir():
        raise RuntimeError(f"existing Qwen3 model directory is missing: {root}")
    result = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix in {".json", ".txt", ".safetensors", ".bin", ".model"}:
            stat = path.stat()
            record = {"file": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            # Weight shards use size + mtime; tokenizer/config files use content hashes too.
            if path.suffix not in {".safetensors", ".bin"}:
                record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            result.append(record)
    if not any(r["file"].endswith((".safetensors", ".bin")) for r in result):
        raise RuntimeError("no local model weights found")
    return result


class LocalBackend:
    def __init__(self):
        settings = llm.generation_metadata()
        required = {"model": "Qwen3-8B", "temperature": 0.7, "max_new_tokens": 1024, "thinking": False}
        if any(settings.get(k) != v for k, v in required.items()):
            raise RuntimeError(f"llm.py settings differ from the requested experiment: {settings}")
        self.metadata = {"generation": settings, "runtime": runtime_metadata(), "model_files": model_inventory()}
        missing = [k for k, v in self.metadata["runtime"]["packages"].items() if v is None]
        if missing:
            raise RuntimeError(f"missing inference dependencies: {missing}; use the existing reflexion conda environment")
        self.last_call_info = {}

    def __call__(self, prompt: str) -> str:
        # Same lazy loader, chat template and exact generation call as llm.py.
        _, tokenizer, model = llm._load_backend()
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        ids = tokenizer(rendered, truncation=False)["input_ids"]
        limit = model.config.max_position_embeddings
        if len(ids) + llm.MAX_NEW_TOKENS > limit:
            raise RuntimeError(
                f"Context overflow: input={len(ids)}, output_budget={llm.MAX_NEW_TOKENS}, "
                f"context_limit={limit}. Article was NOT truncated."
            )
        self.last_call_info = {
            "input_tokens": len(ids), "context_limit": limit, "input_truncated": False,
            "device_map": {k: str(v) for k, v in getattr(model, "hf_device_map", {}).items()},
        }
        return llm.call_llm(prompt)


class MockBackend:
    """Deterministic plumbing test only; never reads labels or article metadata."""
    metadata = {"backend": "mock", "version": 2, "research_result": False}
    last_call_info = {"mock": True}

    def __call__(self, prompt: str) -> str:
        if prompt.startswith("Review the brief rationale"):
            return "Issue: The attribution needs closer scrutiny. Re-check who supports the quoted claim and distinguish observation from inference."
        prediction = "real" if int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 2 else "fake"
        result = "Rationale: The article's attribution and factual consistency inform this assessment.\nFinish[" + prediction + "]"
        if prompt.startswith("Review the previous brief rationale"):
            result = "Reflection: Re-check the attribution against the article before drawing an inference.\n" + result
        return result
