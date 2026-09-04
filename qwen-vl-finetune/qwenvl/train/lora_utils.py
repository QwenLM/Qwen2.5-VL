"""Helpers for composing LoRA and multimodal fine-tuning."""

from __future__ import annotations

from typing import Iterable, Optional


def get_lora_modules_to_save(
    modules_to_save: Optional[Iterable[str]] = None,
    *,
    tune_mm_vision: bool = False,
    tune_mm_mlp: bool = False,
    model=None,
) -> Optional[list[str]]:
    """Merge user-selected PEFT modules with the multimodal components.

    ``modules_to_save`` contains full (non-LoRA) modules that PEFT should keep
    in an adapter checkpoint.  Preserve that list, append the components
    selected by the multimodal flags, and remove duplicates while retaining
    the original order.  Returning ``None`` when there is nothing to save
    keeps PEFT's default behavior unchanged.
    """
    if modules_to_save is None:
        merged = []
    elif isinstance(modules_to_save, str):
        merged = [modules_to_save]
    else:
        merged = list(modules_to_save)

    # Qwen-VL releases have used both a top-level ``visual`` module and a
    # ``model.visual`` layout.  Resolve the relative path before PEFT wraps
    # the model so ``modules_to_save`` points at the actual module.
    visual_path = _visual_module_path(model) if model is not None else "visual"
    visual_path = visual_path or "visual"
    if tune_mm_vision:
        merged.append(visual_path)
    if tune_mm_mlp:
        merged.append(f"{visual_path}.merger")

    return list(dict.fromkeys(merged)) or None


def _unwrap_model(model):
    """Return the underlying transformer model from an optional PEFT wrapper."""
    # PeftModel exposes the wrapped model as ``base_model.model``.  Keeping the
    # lookup here makes the component flags work both before and after PEFT
    # injection, and also avoids relying on implementation-specific wrapper
    # attribute forwarding.
    current = model
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        wrapped = getattr(current, "base_model", None)
        if wrapped is None:
            break
        nested = getattr(wrapped, "model", None)
        if nested is None or nested is current:
            break
        current = nested
    return current


def _visual_module_path(model) -> Optional[str]:
    """Return the PEFT-relative path of the model's visual tower, if present."""
    model = _unwrap_model(model)
    if getattr(model, "visual", None) is not None:
        return "visual"

    nested_model = getattr(model, "model", None)
    if nested_model is not None and getattr(nested_model, "visual", None) is not None:
        return "model.visual"
    return None


def _get_visual_module(model):
    """Return a visual tower from either supported Qwen-VL layout."""
    model = _unwrap_model(model)
    visual = getattr(model, "visual", None)
    if visual is not None:
        return visual

    nested_model = getattr(model, "model", None)
    return getattr(nested_model, "visual", None) if nested_model is not None else None


def _set_requires_grad(module, requires_grad: bool) -> None:
    if module is None:
        return

    # PEFT keeps an immutable base copy next to each trainable
    # ``modules_to_save`` copy.  Toggling every parameter on the wrapper would
    # also train that unused base copy, doubling optimizer state for large
    # vision towers.  Keep the original frozen and only toggle active saved
    # modules, mirroring PEFT's own adapter semantics.
    original_module = getattr(module, "original_module", None)
    saved_modules = getattr(module, "modules_to_save", None)
    if original_module is not None and saved_modules is not None:
        adapters_disabled = bool(getattr(module, "disable_adapters", False))
        _set_requires_grad(original_module, requires_grad and adapters_disabled)
        active_adapters = set(getattr(module, "active_adapters", ()))
        for adapter_name, saved_module in saved_modules.items():
            is_active = not adapters_disabled and adapter_name in active_adapters
            _set_requires_grad(saved_module, requires_grad and is_active)
        return

    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def enable_multimodal_components(
    model,
    *,
    tune_mm_vision: bool = False,
    tune_mm_mlp: bool = False,
) -> None:
    """Enable the requested vision and projector parameters for fine-tuning.

    The LoRA path freezes the complete base model before injecting adapters.
    That used to make ``tune_mm_vision`` and ``tune_mm_mlp`` ineffective when
    ``lora_enable`` was set.  Apply the component policy after PEFT wrapping so
    selected multimodal modules remain trainable while all other base weights
    stay frozen.
    """
    visual = _get_visual_module(model)
    if visual is None:
        return

    # The merger is nested under ``visual`` in Qwen-VL.  Set the complete
    # vision tower first, then apply the dedicated merger flag so
    # ``tune_mm_vision=True, tune_mm_mlp=False`` keeps the projector frozen,
    # matching the non-LoRA training path.
    _set_requires_grad(visual, tune_mm_vision)
    _set_requires_grad(getattr(visual, "merger", None), tune_mm_mlp)


def trainable_parameter_names(model) -> Iterable[str]:
    """Yield trainable parameter names (useful for lightweight regression tests)."""
    return (name for name, parameter in model.named_parameters() if parameter.requires_grad)
