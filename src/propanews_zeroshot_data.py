"""Exact, pinned official PropaNews test split; never sample the local full set."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from propanews_common import PromptArticle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data/propanews_official_test_500/test.jsonl"
UPSTREAM_COMMIT = "916494181e1a39610b2dd541dde1ff162880144e"
SOURCE_URL = (
    "https://raw.githubusercontent.com/khuangaf/FakingFakeNews/"
    f"{UPSTREAM_COMMIT}/data/test.jsonl"
)
EXPECTED_GIT_BLOB = "fcf23938476d56a09b3dd7df0b1c0084150d51c0"
EXPECTED_SHA256 = "7bbee07a6d1437a1e1fe4b2b62eaed0dd9be87afe7aab40ce2776f49e9a92c07"
EXPECTED_BYTES = 3_121_149
EXPECTED_COUNTS = {"real": 243, "fake": 257}


class OfficialTestError(ValueError):
    """The exact official split is unavailable or has failed integrity checks."""


@dataclass(frozen=True)
class OfficialTest:
    articles: tuple[PromptArticle, ...]
    gold: dict[int, str]  # Evaluation-only; never pass this object to a prompt.
    validation: dict


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: object) -> None:
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def _unique_keys(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise OfficialTestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_test_bytes(payload: bytes) -> OfficialTest:
    digest = hashlib.sha256(payload).hexdigest()
    blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
    if (len(payload), digest, blob) != (EXPECTED_BYTES, EXPECTED_SHA256, EXPECTED_GIT_BLOB):
        raise OfficialTestError(
            "Official test.jsonl byte/hash mismatch (missing, altered or truncated data). "
            f"Got bytes={len(payload)}, sha256={digest}, git_blob={blob}. "
            "No replacement or sampling is allowed."
        )
    try:
        lines = payload.decode("utf-8").splitlines()
        if len(lines) != 500 or any(not line.strip() for line in lines):
            raise OfficialTestError("official test.jsonl must contain exactly 500 nonempty rows")
        rows = [json.loads(line, object_pairs_hook=_unique_keys) for line in lines]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OfficialTestError(f"invalid official JSONL: {exc}") from exc
    articles, gold = [], {}
    seen_text, seen_paths = set(), set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"txt", "label", "path"}:
            raise OfficialTestError(f"row {index}: expected txt, label, path fields")
        if type(row["label"]) is not bool:
            raise OfficialTestError(f"row {index}: label must be a JSON boolean")
        text, source_path = row["txt"], row["path"]
        if not isinstance(text, str) or not text.strip() or "\ufffd" in text:
            raise OfficialTestError(f"row {index}: empty/invalid article")
        if not isinstance(source_path, str) or not source_path.strip():
            raise OfficialTestError(f"row {index}: missing source identity")
        # Whitespace normalization is ONLY for duplicate detection, never for prompts.
        identity = " ".join(text.split())
        if identity in seen_text or source_path in seen_paths:
            raise OfficialTestError(f"row {index}: duplicate article or source path")
        seen_text.add(identity)
        seen_paths.add(source_path)
        articles.append(PromptArticle(index=index, article=text))
        gold[index] = "real" if row["label"] else "fake"
    counts = dict(Counter(gold.values()))
    if counts != EXPECTED_COUNTS:
        raise OfficialTestError(f"incorrect label distribution: {counts}")
    return OfficialTest(tuple(articles), gold, {
        "source_url": SOURCE_URL, "upstream_commit": UPSTREAM_COMMIT,
        "git_blob_sha1": blob, "sha256": digest, "bytes": len(payload),
        "num_articles": len(articles), "class_counts": counts,
        "source_text_field": "txt", "prompt_text_field": "article",
        "label_mapping": {"true": "real", "false": "fake"},
        "index_mapping": "zero-based line number, original order",
        "duplicate_articles": 0, "duplicate_paths": 0, "missing_articles": 0,
        "article_characters_min": min(len(a.article) for a in articles),
        "article_characters_max": max(len(a.article) for a in articles),
        "text_preprocessing": "none", "input_truncation": False,
    })


def load_official_test(path: Path = DEFAULT_DATA_PATH, *, download: bool = False) -> OfficialTest:
    path = Path(path)
    if not path.exists():
        if not download:
            raise OfficialTestError(
                f"Official test set unavailable: {path}. Run --prepare-data first. "
                "No random or substitute dataset will be used."
            )
        try:
            with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
                payload = response.read()
        except Exception as exc:
            raise OfficialTestError(f"cannot download official test set from {SOURCE_URL}: {exc}") from exc
        validated = validate_test_bytes(payload)
        atomic_write(path, payload)  # Publish only a fully verified download.
    else:
        try:
            validated = validate_test_bytes(path.read_bytes())
        except OSError as exc:
            raise OfficialTestError(f"cannot read official test set: {exc}") from exc
    return validated
