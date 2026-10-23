"""Shared local-model adapter for Vanilla, CoT, and SPRS experiments.

Model loading and generation live only in this module. Importing it is cheap:
PyTorch, Transformers, the tokenizer, and the weights are loaded lazily on the
first real inference call. Graph modules may therefore read
``generation_metadata()`` without allocating GPU memory.
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Final


MODEL_PATH: Final[str] = "/home/sscc/models/Qwen3-8B"
MODEL_NAME: Final[str] = "Qwen3-8B"
TEMPERATURE: Final[float] = 0.7
SEED: Final[int] = 42
MAX_NEW_TOKENS: Final[int] = 1024
TOP_P: Final[float] = 0.8
TOP_K: Final[int] = 20
MIN_P: Final[float] = 0.0

_BACKEND: tuple[Any, Any, Any] | None = None
_LOAD_LOCK = Lock()
_CALL_LOCK = Lock()


def generation_metadata() -> dict[str, object]:
    """Return the exact generation settings used for cache provenance."""

    return {
        "backend": "transformers_local",
        "model": MODEL_NAME,
        "model_path": MODEL_PATH,
        "torch_dtype": "auto",
        "device_map": "auto",
        "temperature": TEMPERATURE,
        "do_sample": True,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "min_p": MIN_P,
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "thinking": False,
    }


def _load_backend() -> tuple[Any, Any, Any]:
    """Load torch, tokenizer, and model once, only when inference is needed."""

    global _BACKEND
    if _BACKEND is None:
        with _LOAD_LOCK:
            if _BACKEND is None:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_PATH,
                    local_files_only=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    torch_dtype="auto",
                    device_map="auto",
                    local_files_only=True,
                )
                model.eval()
                _BACKEND = (torch, tokenizer, model)
    return _BACKEND


def _input_device(model: Any) -> Any:
    """Return the real device holding the input embedding under device_map."""

    device = model.get_input_embeddings().weight.device
    if getattr(device, "type", None) == "meta":
        raise RuntimeError(
            "Qwen input embeddings are offloaded to the meta device; "
            "choose a device_map that keeps the input embedding on CPU/GPU"
        )
    return device


def call_llm(prompt: str) -> str:
    """Generate one Qwen3-8B response for Vanilla, CoT, or SPRS."""

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

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        # Do not catch or wrap OOM exceptions: graph-level OOM handling needs
        # the original exception in order to checkpoint and skip the article.
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=TOP_P,
                top_k=TOP_K,
                min_p=MIN_P,
            )

        generated = outputs[0, prompt_length:].detach().cpu()
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()

    if not response:
        raise RuntimeError("LLM returned an empty response")
    return response


__all__ = [
    "MAX_NEW_TOKENS",
    "MIN_P",
    "MODEL_NAME",
    "MODEL_PATH",
    "SEED",
    "TEMPERATURE",
    "TOP_K",
    "TOP_P",
    "call_llm",
    "generation_metadata",
]
