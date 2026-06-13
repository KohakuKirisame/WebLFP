from pathlib import Path

import numpy as np
import torch

from weblfp.inference import run_inference
from weblfp.model_service import load_runtime
from weblfp.profile import default_model_dir
from weblfp.recording import SourceConfig


def test_best_clip_checkpoint_produces_normalized_embeddings(tmp_path: Path) -> None:
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

    assert result.embeddings.shape == (4, 128)
    np.testing.assert_allclose(np.linalg.norm(result.embeddings, axis=1), 1, atol=1e-5)
    assert result.profile.epoch == 20
    assert result.device == "cpu"


def test_clip_checkpoint_is_locked_for_inference_only() -> None:
    payload = torch.load(
        default_model_dir() / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert set(payload) == {"format_version", "model_type", "state_dict"}
    assert payload["model_type"] == "clip_lfp_inference"
    assert not any("decoder" in key or "mask_token" in key for key in payload["state_dict"])

    model, _, _ = load_runtime(device_choice="cpu")
    assert all(not parameter.requires_grad for parameter in model.parameters())
