"""Deterministic local Qwen3-8B adapter for the PCoT experiment.

Both PCoT stages and PCoT Reflexion use :func:`call_llm`. Model loading stays
lazy so graph code can read generation metadata without importing PyTorch,
Transformers, or allocating model weights.
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Final


PCOT_MODEL_PATH: Final[str] = "/home/user/models/Qwen3-8B"
PCOT_MODEL: Final[str] = "Qwen3-8B"
PCOT_TEMPERATURE: Final[float] = 0.0
PCOT_SEED: Final[int] = 42
PCOT_MAX_NEW_TOKENS: Final[int] = 1024

_BACKEND: tuple[Any, Any, Any] | None = None
_LOAD_LOCK = Lock()
_CALL_LOCK = Lock()


def generation_metadata() -> dict[str, object]:
    """Return the exact PCoT generation settings for cache provenance."""

    return {
        "backend": "transformers_local",
        "model": PCOT_MODEL,
        "model_path": PCOT_MODEL_PATH,
        "torch_dtype": "auto",
        "device_map": "auto",
        "temperature": PCOT_TEMPERATURE,
        "do_sample": False,
        "seed": PCOT_SEED,
        "max_new_tokens": PCOT_MAX_NEW_TOKENS,
        "thinking": False,
    }


def _load_backend() -> tuple[Any, Any, Any]:
    """Load the PCoT backend once, only on the first inference call."""

    global _BACKEND
    if _BACKEND is None:
        with _LOAD_LOCK:
            if _BACKEND is None:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    PCOT_MODEL_PATH,
                    local_files_only=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    PCOT_MODEL_PATH,
                    torch_dtype="auto",
                    device_map="auto",
                    local_files_only=True,
                )
                model.eval()
                _BACKEND = (torch, tokenizer, model)
    return _BACKEND


def _input_device(model: Any) -> Any:
    """Return the device holding the input embedding under device_map."""

    device = model.get_input_embeddings().weight.device
    if getattr(device, "type", None) == "meta":
        raise RuntimeError(
            "Qwen input embeddings are offloaded to the meta device; "
            "choose a device_map that keeps the input embedding on CPU/GPU"
        )
    return device


def call_llm(prompt: str) -> str:
    """Generate one greedy Qwen3-8B response for a PCoT stage/reflection."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    torch, tokenizer, model = _load_backend()
    with _CALL_LOCK:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(_input_device(model))
        prompt_length = inputs["input_ids"].shape[-1]

        torch.manual_seed(PCOT_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(PCOT_SEED)

        # Preserve the original OOM exception for graph-level skip/checkpoint.
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=PCOT_MAX_NEW_TOKENS,
                do_sample=False,
            )

        generated = outputs[0, prompt_length:].detach().cpu()
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()

    if not response:
        raise RuntimeError("PCoT LLM returned an empty response")
    return response


__all__ = [
    "PCOT_MAX_NEW_TOKENS",
    "PCOT_MODEL",
    "PCOT_MODEL_PATH",
    "PCOT_SEED",
    "PCOT_TEMPERATURE",
    "call_llm",
    "generation_metadata",
]
