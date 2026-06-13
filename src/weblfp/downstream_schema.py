from typing import Literal

from pydantic import BaseModel, Field


class SpikeTypeLabel(BaseModel):
    id: Literal["narrow", "non_narrow"]
    name: str = Field(min_length=1)


class SpikeTypeDecodeResult(BaseModel):
    decoder_id: str
    display_name: str
    device: str
    labels: list[SpikeTypeLabel]
    window_sec: float
    hop_sec: float
    window_start_sec: list[float]
    predicted_counts: list[list[float]]
    rounded_counts: list[list[int]]
    presence_probabilities: list[list[float]]
    presence: list[list[bool]]
    mean_counts: dict[str, float]
    total_predicted_counts: dict[str, float]
    presence_rates: dict[str, float]
    reference_metrics: dict[str, float]
    limitations: list[str]
