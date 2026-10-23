"""Durable per-call ledger. Raw outputs commit before parsing; one writer per run."""

from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from propanews_common import is_oom_error, recover_from_oom
from propanews_zeroshot_data import write_json
from propanews_zeroshot_prompts import ResponseParseError


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class CheckpointError(RuntimeError):
    pass


class ModelCallBudgetExhausted(CheckpointError):
    pass


class Ledger:
    def __init__(self, directory: Path, provenance: dict, *,
                 compatible_provenances: tuple[dict, ...] = ()):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / ".run.lock").open("a+")
        self.db = None
        self.max_calls_per_article = None
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.db = sqlite3.connect(self.directory / "checkpoint.sqlite3")
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS manifest (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS calls (
                    id INTEGER PRIMARY KEY, article_index INTEGER NOT NULL,
                    trial INTEGER NOT NULL, kind TEXT NOT NULL, attempt INTEGER NOT NULL,
                    prompt TEXT NOT NULL, prompt_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL, response TEXT, parsed TEXT,
                    error_type TEXT, error_message TEXT, started_at TEXT NOT NULL,
                    completed_at TEXT, backend_info TEXT,
                    UNIQUE(article_index, trial, kind, attempt)
                );
            """)
            manifest = {"fingerprint": fingerprint(provenance), "provenance": provenance}
            stored = self.db.execute("SELECT payload FROM manifest WHERE id=1").fetchone()
            if stored:
                stored_manifest = json.loads(stored[0])
                if stored_manifest != manifest:
                    compatible = any(stored_manifest == {
                        "fingerprint": fingerprint(previous), "provenance": previous,
                    } for previous in compatible_provenances)
                    if not compatible:
                        raise CheckpointError(
                            "Checkpoint provenance mismatch: dataset, prompts, code, backend, "
                            "model files/settings or runtime changed. Choose a NEW --output-dir."
                        )
                    # Upgrade metadata under the writer lock; retain provenance history.
                    write_json(self.directory / f"manifest.previous.{stored_manifest['fingerprint']}.json", stored_manifest)
                    with self.db:
                        self.db.execute("UPDATE manifest SET payload=? WHERE id=1", (json.dumps(manifest),))
            else:
                with self.db:
                    self.db.execute("INSERT INTO manifest VALUES (1, ?)", (json.dumps(manifest),))
            self.manifest = manifest
            write_json(self.directory / "manifest.json", manifest)
            # A process died while the call was in flight. Its completion is unknown.
            with self.db:
                self.db.execute(
                    "UPDATE calls SET status='uncertain', error_type='InterruptedCall', "
                    "error_message='Previous process ended before a durable response', completed_at=? "
                    "WHERE status='running'", (timestamp(),)
                )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.db is not None:
            self.db.close()
            self.db = None
        if not self.lock.closed:
            self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def latest(self, index: int, trial: int, kind: str) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM calls WHERE article_index=? AND trial=? AND kind=? ORDER BY attempt DESC LIMIT 1",
            (index, trial, kind),
        ).fetchone()
        return dict(row) if row else None

    def all_calls(self) -> list[dict]:
        return [dict(row) for row in self.db.execute("SELECT * FROM calls ORDER BY id")]

    def limit_calls_per_article(self, maximum: int) -> None:
        """Tighten the lifetime budget, including failed and interrupted attempts."""
        if type(maximum) is not int or maximum < 1:
            raise ValueError("maximum must be a positive integer")
        self.max_calls_per_article = min(self.max_calls_per_article or maximum, maximum)

    def call_count(self, index: int) -> int:
        return self.db.execute("SELECT COUNT(*) FROM calls WHERE article_index=?", (index,)).fetchone()[0]

    def call(self, index: int, trial: int, kind: str, prompt: str,
             parser: Callable, backend: Callable, *, retry_errors: bool = False,
             retry_uncertain: bool = False) -> dict | None:
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        previous = self.latest(index, trial, kind)
        if previous is not None:
            if previous["prompt_sha256"] != digest:
                raise CheckpointError(f"cached prompt mismatch: article={index} trial={trial} kind={kind}")
            if previous["status"] == "ok":
                return json.loads(previous["parsed"])
            if previous["status"] == "raw":
                return self._parse(previous["id"], previous["response"], parser)
            # A malformed but completed generation is NEVER silently regenerated.
            if previous["status"] == "parse_error":
                return None
            if previous["status"] == "uncertain" and not retry_uncertain:
                return None
            if previous["status"] == "execution_error" and not retry_errors:
                return None
        attempt = previous["attempt"] + 1 if previous else 1
        if self.max_calls_per_article is not None and self.call_count(index) >= self.max_calls_per_article:
            raise ModelCallBudgetExhausted(
                f"article {index}: lifetime model-call limit {self.max_calls_per_article} reached; no additional call"
            )
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO calls (article_index,trial,kind,attempt,prompt,prompt_sha256,status,started_at) "
                "VALUES (?,?,?,?,?,?,'running',?)",
                (index, trial, kind, attempt, prompt, digest, timestamp()),
            )
            call_id = cursor.lastrowid
        try:
            response = backend(prompt)
            if not isinstance(response, str):
                raise TypeError("backend returned a non-string response")
        except BaseException as exc:
            status = "uncertain" if not isinstance(exc, Exception) else "execution_error"
            with self.db:
                self.db.execute(
                    "UPDATE calls SET status=?,error_type=?,error_message=?,completed_at=? WHERE id=?",
                    (status, "CUDA_OOM" if is_oom_error(exc) else type(exc).__name__, str(exc), timestamp(), call_id),
                )
            if is_oom_error(exc):
                recover_from_oom()
            if not isinstance(exc, Exception):
                raise
            print(f"[execution_error] index={index} trial={trial} {kind}: {exc}", flush=True)
            return None
        # Commit output before parsing or progressing to another model call.
        with self.db:
            self.db.execute(
                "UPDATE calls SET status='raw',response=?,completed_at=?,backend_info=? WHERE id=?",
                (response, timestamp(), json.dumps(getattr(backend, "last_call_info", {})), call_id),
            )
        return self._parse(call_id, response, parser)

    def _parse(self, call_id: int, response: str, parser: Callable) -> dict | None:
        try:
            parsed = parser(response)
        except ResponseParseError as exc:
            with self.db:
                self.db.execute(
                    "UPDATE calls SET status='parse_error',error_type=?,error_message=? WHERE id=?",
                    (type(exc).__name__, str(exc), call_id),
                )
            print(f"[parse_error] call={call_id}: {exc}", flush=True)
            return None
        with self.db:
            self.db.execute("UPDATE calls SET status='ok',parsed=? WHERE id=?", (json.dumps(parsed), call_id))
        return parsed
