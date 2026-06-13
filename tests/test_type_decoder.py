from pathlib import Path

import numpy as np
import torch

from weblfp.type_decoder import (
    decode_spike_types,
    default_decoder_dir,
    load_decoder_profile,
    load_decoder_runtime,
)


def test_reference_decoder_predicts_presence_and_count_from_lfp_features() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(2, 256)).astype(np.float32)
    result = decode_spike_types(
        features=features,
        window_start_sec=np.array([0.0, 0.2], dtype=np.float64),
        batch_size=2,
        device_choice="cpu",
    )

    assert result.window_start_sec == [0.0, 0.2]
    assert np.asarray(result.predicted_counts).shape == (2, 2)
    assert np.asarray(result.rounded_counts).shape == (2, 2)
    probabilities = np.asarray(result.presence_probabilities)
    assert probabilities.shape == (2, 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert [label.id for label in result.labels] == ["narrow", "non_narrow"]


def test_reference_decoder_profile_matches_original_task() -> None:
    profile = load_decoder_profile()

    assert profile.window_sec == 0.2
    assert profile.hop_sec == 0.2
    assert profile.presence_threshold == 0.5
    assert profile.feature_dim == 256


def test_spike_type_checkpoint_is_locked_for_inference_only() -> None:
    payload = torch.load(
        default_decoder_dir() / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert set(payload) == {"format_version", "model_type", "feature_extractor", "head"}
    assert payload["model_type"] == "spike_type_inference"
    assert not any(
        "decoder" in key or "mask_token" in key for key in payload["feature_extractor"]
    )

    head, _, _ = load_decoder_runtime(device_choice="cpu")
    assert all(not parameter.requires_grad for parameter in head.parameters())


def test_decoder_checkpoint_matches_spike_count_clip_source_when_available() -> None:
    source_checkpoint = Path(r"M:\LFP\checkpoints\spike_count\clip\best.pt")
    if not source_checkpoint.is_file():
        return

    source = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    packaged = torch.load(
        default_decoder_dir() / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    source_encoder = source["feature_extractor"]
    packaged_encoder = packaged["feature_extractor"]
    source_head = source["model"]
    packaged_head = packaged["head"]

    assert packaged_head.keys() == source_head.keys()
    for key, value in packaged_head.items():
        assert torch.equal(value, source_head[key])

    assert all(key in source_encoder for key in packaged_encoder)
    for key, value in packaged_encoder.items():
        assert torch.equal(value, source_encoder[key])
