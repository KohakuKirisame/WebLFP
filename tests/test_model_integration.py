from pathlib import Path

import numpy as np
import torch

from weblfp.inference import run_inference
from weblfp.model_service import load_runtime
from weblfp.profile import default_model_dir
from weblfp.recording import SourceConfig


TRAINING_STATE_TOKENS = {
    "optimizer",
    "scheduler",
    "scaler",
    "teacher",
    "student",
    "decoder",
    "mask_token",
    "global_step",
    "best_val_loss",
}


def _assert_no_training_state(payload: dict[str, object]) -> None:
    assert TRAINING_STATE_TOKENS.isdisjoint(payload)
    for section_name in ("feature_extractor", "head"):
        section = payload[section_name]
        assert isinstance(section, dict)
        assert not any(
            token in key.lower()
            for key in section
            for token in TRAINING_STATE_TOKENS
        )


def test_unified_checkpoint_produces_lfp_features(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    path = tmp_path / "synthetic-lfp.npy"
    np.save(path, rng.normal(size=(4, 750)).astype(np.float32))

    result = run_inference(
        source=SourceConfig(
            path=str(path),
            format="npy",
            sampling_rate_hz=1875,
            channel_axis="first",
        ),
        start_sec=0,
        end_sec=0.4,
        channel_ids=["0", "1", "2", "3"],
        batch_size=2,
        device_choice="cpu",
    )

    assert result.embeddings.shape == (2, 256)
    assert np.all(np.isfinite(result.embeddings))
    assert result.profile.epoch == 9
    assert result.device == "cpu"


def test_unified_checkpoint_is_locked_for_inference_only() -> None:
    payload = torch.load(
        default_model_dir() / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert set(payload) == {"format_version", "model_type", "feature_extractor", "head"}
    assert payload["model_type"] == "spike_type_inference"
    _assert_no_training_state(payload)

    model, _, _ = load_runtime(device_choice="cpu")
    assert all(not parameter.requires_grad for parameter in model.parameters())
