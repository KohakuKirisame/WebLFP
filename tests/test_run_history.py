from pathlib import Path

import numpy as np

from weblfp.inference import InferenceResult
from weblfp.profile import default_model_dir, load_model_profile
from weblfp.run_history import RunStore


def test_run_store_persists_and_restores_result_data(tmp_path: Path) -> None:
    profile = load_model_profile(default_model_dir())
    embeddings = np.eye(3, profile.embedding_dim, dtype=np.float32)
    result = InferenceResult(
        embeddings=embeddings,
        window_start_sec=np.array([0.0, 0.05, 0.1]),
        window_end_sec=np.array([0.2, 0.25, 0.3]),
        pca_2d=np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]], dtype=np.float32),
        adjacent_cosine_similarity=np.array([0.2, 0.4], dtype=np.float32),
        device="cpu",
        profile=profile,
        source_sample_rate_hz=1875,
        selected_channel_ids=["0", "1"],
    )
    request = {
        "source": {"path": str(tmp_path / "sample.plx"), "format": "plexon"},
        "start_sec": 0,
        "end_sec": 0.3,
        "channel_ids": ["0", "1"],
        "batch_size": 32,
        "device": "cpu",
    }

    stored = RunStore(tmp_path / "runs").add(result, request)
    store = RunStore(tmp_path / "runs")
    store.save_downstream(stored.run_id, {"decoder_id": "reference", "presence_rates": {}})
    reopened = RunStore(tmp_path / "runs")

    assert reopened.list()[0].run_id == stored.run_id
    assert reopened.list()[0].source_name == "sample.plx"
    restored = reopened.get(stored.run_id)
    assert restored.window_count == 3
    assert restored.embedding_dim == profile.embedding_dim
    assert restored.window_start_sec == [0.0, 0.05, 0.1]
    assert len(restored.pca_2d) == 3
    assert restored.downstream == {"decoder_id": "reference", "presence_rates": {}}
    assert reopened.arrays_path(stored.run_id).is_file()
    assert reopened.metadata_path(stored.run_id).is_file()
