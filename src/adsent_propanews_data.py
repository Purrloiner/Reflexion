"""Validated PropaNews loading for prompt-only AdSent experiments.

This module deliberately performs no file I/O at import time.  Call
``load_propanews_dataset`` (or ``load_default_propanews_dataset``) explicitly.

The public record types separate the three kinds of experiment data:

* :class:`PromptInput` contains only information that may be shown to a model.
* :class:`GoldRecord` contains the canonical target label.
* :class:`EvalMetadata` contains bookkeeping information, including the
  manipulation subtype inferred from the private dataset path.

Keeping these values in separate objects makes it harder to accidentally leak a
filename suffix or a gold label into a detector prompt.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Final,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    TypeAlias,
    TypeGuard,
    cast,
)


CanonicalLabel: TypeAlias = Literal["human_original", "ai_manipulated"]
InputMode: TypeAlias = Literal["article_only", "reference_assisted"]
ManipulationSubtype: TypeAlias = Literal[
    "human_original",
    "app_auth_final",
    "sent_fake",
    "ll_filled",
]

HUMAN_ORIGINAL: Final[CanonicalLabel] = "human_original"
AI_MANIPULATED: Final[CanonicalLabel] = "ai_manipulated"
ARTICLE_ONLY: Final[InputMode] = "article_only"
REFERENCE_ASSISTED: Final[InputMode] = "reference_assisted"

DEFAULT_ARTICLES_PATH: Final[Path] = (
    Path(__file__).parent.parent / "data" / "propanews_150_sample_final.json"
)
DEFAULT_REFERENCES_PATH: Final[Path] = (
    Path(__file__).parent.parent / "data" / "propanews_GT_final.json"
)

_SUBTYPE_SUFFIXES: Final[tuple[tuple[str, ManipulationSubtype], ...]] = (
    (".app-auth-final", "app_auth_final"),
    (".sent-fake", "sent_fake"),
    (".ll_filled", "ll_filled"),
)
_SOURCE_FIELDS: Final[tuple[str, ...]] = (
    "publisher",
    "author",
    "date",
    "original_filename",
    "title",
    "retrieval_status",
)


class DatasetIntegrityError(ValueError):
    """Raised when the two PropaNews files cannot form a safe, exact join."""


class _DuplicateJSONKey(ValueError):
    """Internal signal used to reject duplicate object keys in source JSON."""


@dataclass(frozen=True, slots=True)
class PromptInput:
    """Model-visible input for one PropaNews item.

    ``reference_context`` is absent in ``article_only`` mode and required in
    ``reference_assisted`` mode.  This type intentionally has no label, source
    path, or manipulation-subtype field.
    """

    index: int
    mode: InputMode
    article: str
    reference_context: str | None = None

    def __post_init__(self) -> None:
        _validate_index(self.index, "PromptInput.index")
        _require_nonempty_string(self.article, "PromptInput.article")
        if self.mode not in (ARTICLE_ONLY, REFERENCE_ASSISTED):
            raise DatasetIntegrityError(f"unsupported input mode: {self.mode!r}")
        if self.mode == ARTICLE_ONLY and self.reference_context is not None:
            raise DatasetIntegrityError(
                "article_only PromptInput must not contain reference_context"
            )
        if self.mode == REFERENCE_ASSISTED:
            _require_nonempty_string(
                self.reference_context, "PromptInput.reference_context"
            )

    def sections(self) -> tuple[tuple[str, str], ...]:
        """Return model-visible sections in a deterministic display order."""

        if self.mode == ARTICLE_ONLY:
            return (("TARGET ARTICLE", self.article),)
        # ``__post_init__`` guarantees this is a string in assisted mode.
        reference = cast(str, self.reference_context)
        return (
            ("REFERENCE CONTEXT", reference),
            ("TARGET ARTICLE", self.article),
        )

    def render(self) -> str:
        """Render the model-visible payload without adding task instructions."""

        return "\n\n".join(f"{title}:\n{text}" for title, text in self.sections())


@dataclass(frozen=True, slots=True)
class GoldRecord:
    """Private canonical target for one item."""

    index: int
    label: CanonicalLabel

    def __post_init__(self) -> None:
        _validate_index(self.index, "GoldRecord.index")
        if self.label not in (HUMAN_ORIGINAL, AI_MANIPULATED):
            raise DatasetIntegrityError(f"invalid canonical label: {self.label!r}")


@dataclass(frozen=True, slots=True)
class EvalMetadata:
    """Evaluation-only provenance that must never be included in a prompt."""

    index: int
    manipulation_subtype: ManipulationSubtype
    source_path: str
    publisher: str
    author: str
    date: str
    original_filename: str
    source_title: str
    retrieval_status: str

    def __post_init__(self) -> None:
        _validate_index(self.index, "EvalMetadata.index")
        if self.manipulation_subtype not in {
            "human_original",
            "app_auth_final",
            "sent_fake",
            "ll_filled",
        }:
            raise DatasetIntegrityError(
                f"invalid manipulation subtype: {self.manipulation_subtype!r}"
            )
        for field_name in (
            "source_path",
            "publisher",
            "author",
            "date",
            "original_filename",
            "source_title",
            "retrieval_status",
        ):
            _require_nonempty_string(
                getattr(self, field_name), f"EvalMetadata.{field_name}"
            )


@dataclass(frozen=True, slots=True)
class PropaNewsRecord:
    """A joined sample whose prompt, gold, and metadata remain explicit."""

    prompt: PromptInput
    gold: GoldRecord
    metadata: EvalMetadata

    @property
    def index(self) -> int:
        """Return the stable join key shared by all record components."""

        return self.prompt.index


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """Summary returned after a successful dataset integrity check."""

    record_count: int
    human_original_count: int
    ai_manipulated_count: int
    subtype_counts: Mapping[str, int]
    input_mode: InputMode | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "record_count": self.record_count,
            "human_original_count": self.human_original_count,
            "ai_manipulated_count": self.ai_manipulated_count,
            "subtype_counts": dict(self.subtype_counts),
            "input_mode": self.input_mode,
        }


def canonical_label(raw_label: object) -> CanonicalLabel:
    """Convert the legacy PropaNews boolean to the canonical experiment label.

    In the supplied PropaNews file, ``True`` denotes the unmodified human news
    article and ``False`` denotes an AI-manipulated article.  Integers ``0`` and
    ``1`` are rejected even though ``bool`` subclasses ``int`` in Python.
    """

    if type(raw_label) is not bool:
        raise DatasetIntegrityError(
            "PropaNews label must be a JSON boolean (true=human original, "
            "false=AI manipulated)"
        )
    return HUMAN_ORIGINAL if raw_label else AI_MANIPULATED


def derive_manipulation_subtype(source_path: str) -> ManipulationSubtype:
    """Infer the private subtype from a PropaNews path suffix.

    The returned value belongs in :class:`EvalMetadata` only.  Unknown suffixes
    are rejected instead of being silently treated as original.
    """

    _require_nonempty_string(source_path, "source path")
    filename = Path(source_path).name
    for suffix, subtype in _SUBTYPE_SUFFIXES:
        if filename.endswith(suffix):
            return subtype
    if filename.endswith(".rsd.txt"):
        return cast(ManipulationSubtype, HUMAN_ORIGINAL)
    raise DatasetIntegrityError(
        f"unrecognized PropaNews path suffix for {source_path!r}"
    )


def format_reference_context(raw_context: Mapping[str, object]) -> str:
    """Validate and render a ``GT_context`` object for a model prompt."""

    if not isinstance(raw_context, Mapping):
        raise DatasetIntegrityError("GT_context must be a JSON object")
    titles = raw_context.get("title")
    sentence_groups = raw_context.get("sentences")
    if not _is_sequence(titles) or not _is_sequence(sentence_groups):
        raise DatasetIntegrityError(
            "GT_context.title and GT_context.sentences must be JSON arrays"
        )
    if not titles:
        raise DatasetIntegrityError("GT_context.title must not be empty")
    if len(titles) != len(sentence_groups):
        raise DatasetIntegrityError(
            "GT_context.title and GT_context.sentences must have equal lengths"
        )

    sections: list[str] = []
    for position, (title, sentences) in enumerate(zip(titles, sentence_groups)):
        title_text = _require_nonempty_string(
            title, f"GT_context.title[{position}]"
        )
        if not _is_sequence(sentences) or not sentences:
            raise DatasetIntegrityError(
                f"GT_context.sentences[{position}] must be a non-empty array"
            )
        rendered_sentences = [
            _require_nonempty_string(
                sentence, f"GT_context.sentences[{position}][{sentence_position}]"
            )
            for sentence_position, sentence in enumerate(sentences)
        ]
        sections.append("\n".join((f"Title: {title_text}", *rendered_sentences)))
    return "\n\n".join(sections)


def build_prompt_input(
    *,
    index: int,
    article: str,
    mode: InputMode,
    reference_context: str | None = None,
) -> PromptInput:
    """Create a leak-resistant prompt input for one of the two input modes.

    Any supplied reference is intentionally discarded in ``article_only`` mode.
    Assisted mode requires a non-empty, already-rendered reference context.
    """

    if mode == ARTICLE_ONLY:
        return PromptInput(index=index, mode=mode, article=article)
    if mode == REFERENCE_ASSISTED:
        return PromptInput(
            index=index,
            mode=mode,
            article=article,
            reference_context=reference_context,
        )
    raise DatasetIntegrityError(f"unsupported input mode: {mode!r}")


def load_propanews_dataset(
    articles_path: str | PathLike[str],
    references_path: str | PathLike[str],
    *,
    mode: InputMode = ARTICLE_ONLY,
) -> tuple[PropaNewsRecord, ...]:
    """Load, strictly join, validate, and return PropaNews records.

    Both JSON roots must be arrays.  Rows are joined by their integer ``index``
    fields rather than array position.  Duplicate keys and any difference
    between the two index sets are fatal.  Output is sorted by index and returned
    as a tuple so the evaluated cohort cannot be changed accidentally.
    """

    if mode not in (ARTICLE_ONLY, REFERENCE_ASSISTED):
        raise DatasetIntegrityError(f"unsupported input mode: {mode!r}")

    article_rows = _load_json_array(articles_path, "article dataset")
    reference_rows = _load_json_array(references_path, "reference dataset")
    articles_by_index = _index_rows(article_rows, "article dataset")
    references_by_index = _index_rows(reference_rows, "reference dataset")
    _require_same_indices(articles_by_index, references_by_index)

    records: list[PropaNewsRecord] = []
    for index in sorted(articles_by_index):
        article_row = articles_by_index[index]
        reference_row = references_by_index[index]
        article = _require_nonempty_string(
            article_row.get("txt"), f"article dataset index {index}: txt"
        )
        label = canonical_label(article_row.get("label"))
        source_path = _require_nonempty_string(
            article_row.get("path"), f"article dataset index {index}: path"
        )
        subtype = derive_manipulation_subtype(source_path)
        _validate_label_subtype(index, label, subtype)

        raw_context = reference_row.get("GT_context")
        if not isinstance(raw_context, Mapping):
            raise DatasetIntegrityError(
                f"reference dataset index {index}: GT_context must be an object"
            )
        reference_context = format_reference_context(raw_context)
        raw_source = reference_row.get("GT_source")
        source = _validate_source(raw_source, index)
        _validate_matching_filename(source_path, source["original_filename"], index)

        prompt = build_prompt_input(
            index=index,
            article=article,
            mode=mode,
            reference_context=reference_context,
        )
        gold = GoldRecord(index=index, label=label)
        metadata = EvalMetadata(
            index=index,
            manipulation_subtype=subtype,
            source_path=source_path,
            publisher=source["publisher"],
            author=source["author"],
            date=source["date"],
            original_filename=source["original_filename"],
            source_title=source["title"],
            retrieval_status=source["retrieval_status"],
        )
        records.append(PropaNewsRecord(prompt=prompt, gold=gold, metadata=metadata))

    result = tuple(records)
    validate_records(result)
    return result


def load_default_propanews_dataset(
    *, mode: InputMode = ARTICLE_ONLY
) -> tuple[PropaNewsRecord, ...]:
    """Load the repository's supplied PropaNews and GT files explicitly."""

    return load_propanews_dataset(
        DEFAULT_ARTICLES_PATH,
        DEFAULT_REFERENCES_PATH,
        mode=mode,
    )


def validate_records(records: Iterable[PropaNewsRecord]) -> IntegrityReport:
    """Validate component alignment, label/subtype consistency, and uniqueness.

    The function accepts any iterable, consumes it once, and returns aggregate
    counts on success.  It raises :class:`DatasetIntegrityError` at the first
    integrity violation.
    """

    materialized = tuple(records)
    seen: set[int] = set()
    label_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    modes: set[InputMode] = set()

    for position, record in enumerate(materialized):
        if not isinstance(record, PropaNewsRecord):
            raise DatasetIntegrityError(
                f"records[{position}] is not a PropaNewsRecord"
            )
        indices = {record.prompt.index, record.gold.index, record.metadata.index}
        if len(indices) != 1:
            raise DatasetIntegrityError(
                f"record components have different indices at position {position}"
            )
        index = record.index
        if index in seen:
            raise DatasetIntegrityError(f"duplicate joined record index: {index}")
        seen.add(index)
        _validate_label_subtype(
            index, record.gold.label, record.metadata.manipulation_subtype
        )
        _validate_matching_filename(
            record.metadata.source_path,
            record.metadata.original_filename,
            index,
        )
        label_counts[record.gold.label] += 1
        subtype_counts[record.metadata.manipulation_subtype] += 1
        modes.add(record.prompt.mode)

    if len(modes) > 1:
        raise DatasetIntegrityError(
            "a dataset cohort must not mix article_only and reference_assisted inputs"
        )
    input_mode = next(iter(modes)) if modes else None
    return IntegrityReport(
        record_count=len(materialized),
        human_original_count=label_counts[HUMAN_ORIGINAL],
        ai_manipulated_count=label_counts[AI_MANIPULATED],
        subtype_counts=dict(sorted(subtype_counts.items())),
        input_mode=input_mode,
    )


def _load_json_array(
    path: str | PathLike[str], description: str
) -> list[Mapping[str, object]]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_nonstandard_json_constant,
            )
    except (OSError, UnicodeError, ValueError) as exc:
        raise DatasetIntegrityError(f"cannot load {description} from {path!s}: {exc}") from exc
    if not isinstance(payload, list):
        raise DatasetIntegrityError(f"{description} root must be a JSON array")
    rows: list[Mapping[str, object]] = []
    for position, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise DatasetIntegrityError(
                f"{description} row {position} must be a JSON object"
            )
        rows.append(row)
    return rows


def _index_rows(
    rows: Iterable[Mapping[str, object]], description: str
) -> dict[int, Mapping[str, object]]:
    indexed: dict[int, Mapping[str, object]] = {}
    for position, row in enumerate(rows):
        index = row.get("index")
        _validate_index(index, f"{description} row {position}: index")
        index = cast(int, index)
        if index in indexed:
            raise DatasetIntegrityError(f"{description} has duplicate index {index}")
        indexed[index] = row
    return indexed


def _require_same_indices(
    articles: Mapping[int, object], references: Mapping[int, object]
) -> None:
    article_indices = set(articles)
    reference_indices = set(references)
    if article_indices == reference_indices:
        return
    missing_references = sorted(article_indices - reference_indices)
    missing_articles = sorted(reference_indices - article_indices)
    raise DatasetIntegrityError(
        "index sets differ: "
        f"missing from reference dataset={missing_references}; "
        f"missing from article dataset={missing_articles}"
    )


def _validate_source(raw_source: object, index: int) -> dict[str, str]:
    if not isinstance(raw_source, Mapping):
        raise DatasetIntegrityError(
            f"reference dataset index {index}: GT_source must be an object"
        )
    result: dict[str, str] = {}
    for field_name in _SOURCE_FIELDS:
        result[field_name] = _require_nonempty_string(
            raw_source.get(field_name),
            f"reference dataset index {index}: GT_source.{field_name}",
        )
    return result


def _validate_label_subtype(
    index: int, label: CanonicalLabel, subtype: ManipulationSubtype
) -> None:
    if label == HUMAN_ORIGINAL and subtype != HUMAN_ORIGINAL:
        raise DatasetIntegrityError(
            f"index {index}: human_original label conflicts with subtype {subtype!r}"
        )
    if label == AI_MANIPULATED and subtype == HUMAN_ORIGINAL:
        raise DatasetIntegrityError(
            f"index {index}: ai_manipulated label has an unmodified path"
        )


def _validate_matching_filename(
    source_path: str, original_filename: str, index: int
) -> None:
    candidate = Path(source_path).name
    for suffix, _ in _SUBTYPE_SUFFIXES:
        if candidate.endswith(suffix):
            candidate = candidate[: -len(suffix)]
            break
    if candidate != original_filename:
        raise DatasetIntegrityError(
            f"index {index}: path basename {candidate!r} does not match "
            f"GT_source.original_filename {original_filename!r}"
        )


def _validate_index(value: object, description: str) -> None:
    if type(value) is not int or cast(int, value) < 0:
        raise DatasetIntegrityError(f"{description} must be a non-negative integer")


def _require_nonempty_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetIntegrityError(f"{description} must be a non-empty string")
    return value


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant: {value}")


__all__ = [
    "AI_MANIPULATED",
    "ARTICLE_ONLY",
    "CanonicalLabel",
    "DEFAULT_ARTICLES_PATH",
    "DEFAULT_REFERENCES_PATH",
    "DatasetIntegrityError",
    "EvalMetadata",
    "GoldRecord",
    "HUMAN_ORIGINAL",
    "InputMode",
    "IntegrityReport",
    "ManipulationSubtype",
    "PromptInput",
    "PropaNewsRecord",
    "REFERENCE_ASSISTED",
    "build_prompt_input",
    "canonical_label",
    "derive_manipulation_subtype",
    "format_reference_context",
    "load_default_propanews_dataset",
    "load_propanews_dataset",
    "validate_records",
]
