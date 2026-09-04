import pytest
from torch import nn

from qwenvl.train.lora_utils import enable_multimodal_components, get_lora_modules_to_save


class _VisualTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.merger = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 4))


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _VisualTower()
        self.language_model = nn.Module()
        self.language_model.q_proj = nn.Linear(4, 4)


class _NestedModel(nn.Module):
    """Match the ``ForConditionalGeneration.model.visual`` layout."""

    def __init__(self, model=None):
        super().__init__()
        self.model = model or _Model()


class _PeftLikeWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = model


def _freeze(model):
    for parameter in model.parameters():
        parameter.requires_grad = False


def test_lora_component_flags_enable_only_requested_modules():
    model = _Model()
    _freeze(model)

    enable_multimodal_components(model, tune_mm_mlp=True)

    assert all(parameter.requires_grad for parameter in model.visual.merger.parameters())
    assert not any(parameter.requires_grad for parameter in model.visual.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.language_model.parameters())


def test_lora_component_flags_work_after_peft_wrapping():
    model = _Model()
    _freeze(model)
    wrapped = _PeftLikeWrapper(model)

    enable_multimodal_components(wrapped, tune_mm_vision=True)

    assert all(parameter.requires_grad for parameter in model.visual.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.visual.merger.parameters())
    assert not any(parameter.requires_grad for parameter in model.language_model.parameters())


def test_lora_modules_to_save_preserves_and_deduplicates_configuration():
    assert get_lora_modules_to_save(
        ["language_model", "visual"],
        tune_mm_vision=True,
        tune_mm_mlp=True,
    ) == ["language_model", "visual", "visual.merger"]
    assert get_lora_modules_to_save("language_model") == ["language_model"]
    assert get_lora_modules_to_save() is None
    assert get_lora_modules_to_save(tune_mm_vision=True, model=_NestedModel()) == ["model.visual"]
    assert get_lora_modules_to_save(tune_mm_mlp=True, model=_NestedModel()) == ["model.visual.merger"]


def _make_peft_model(modules_to_save, nested=False):
    peft = pytest.importorskip("peft")
    model = _NestedModel() if nested else _Model()
    _freeze(model)
    config = peft.LoraConfig(
        target_modules=["q_proj"],
        modules_to_save=modules_to_save,
    )
    return peft.get_peft_model(model, config)


@pytest.mark.parametrize(
    (
        "tune_mm_vision",
        "tune_mm_mlp",
        "expected_encoder_saved",
        "expected_merger_saved",
        "expected_encoder_trainable",
        "expected_merger_trainable",
        "nested",
    ),
    [
        (False, False, False, False, False, False, False),
        (True, False, True, True, True, False, False),
        (False, True, False, True, False, True, False),
        (True, True, True, True, True, True, False),
        (False, False, False, False, False, False, True),
        (True, False, True, True, True, False, True),
        (False, True, False, True, False, True, True),
        (True, True, True, True, True, True, True),
    ],
)
def test_lora_state_dict_keeps_enabled_multimodal_weights(
    tune_mm_vision,
    tune_mm_mlp,
    expected_encoder_saved,
    expected_merger_saved,
    expected_encoder_trainable,
    expected_merger_trainable,
    nested,
):
    peft = pytest.importorskip("peft")
    base_model = _NestedModel() if nested else _Model()
    modules_to_save = get_lora_modules_to_save(
        tune_mm_vision=tune_mm_vision,
        tune_mm_mlp=tune_mm_mlp,
        model=base_model,
    )
    wrapped = _make_peft_model(modules_to_save, nested=nested)
    enable_multimodal_components(
        wrapped,
        tune_mm_vision=tune_mm_vision,
        tune_mm_mlp=tune_mm_mlp,
    )

    state_dict = peft.get_peft_model_state_dict(wrapped)
    visual_keys = [key for key in state_dict if ".visual." in key]
    merger_keys = [key for key in visual_keys if ".merger." in key]
    encoder_keys = [key for key in visual_keys if ".encoder." in key]

    assert bool(encoder_keys) is expected_encoder_saved
    assert bool(merger_keys) is expected_merger_saved
    trainable_names = [name for name, parameter in wrapped.named_parameters() if parameter.requires_grad]
    trainable_visual_names = [name for name in trainable_names if ".visual." in name]
    assert any(".encoder." in name for name in trainable_visual_names) == expected_encoder_trainable
    assert any(".merger." in name for name in trainable_visual_names) == expected_merger_trainable
    assert not any(".original_module." in name for name in trainable_visual_names)
