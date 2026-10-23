"""Compatibility re-export for canonical label-blind SPRS Reflexion."""

if __package__:
    from .reflection_mtcpsl import (
        SPRS_reflection_prompt,
        make_reflection_SPRS,
        sanitize_reflection,
    )
else:
    from reflection_mtcpsl import (
        SPRS_reflection_prompt,
        make_reflection_SPRS,
        sanitize_reflection,
    )

__all__ = [
    "SPRS_reflection_prompt",
    "make_reflection_SPRS",
    "sanitize_reflection",
]
