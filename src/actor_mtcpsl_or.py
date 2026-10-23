"""Compatibility re-export for the canonical article-only SPRS actor."""

if __package__:
    from .actor_mtcpsl import (
        PROMPT_VERSION,
        SPRS_actor,
        SPRS_prompt,
        get_article,
        get_available_indices,
    )
else:
    from actor_mtcpsl import (
        PROMPT_VERSION,
        SPRS_actor,
        SPRS_prompt,
        get_article,
        get_available_indices,
    )

__all__ = [
    "PROMPT_VERSION",
    "SPRS_actor",
    "SPRS_prompt",
    "get_article",
    "get_available_indices",
]
