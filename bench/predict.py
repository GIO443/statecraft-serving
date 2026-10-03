"""Written-down KV cache prediction, compared against vLLM's startup log after each launch.

vLLM sizes the KV cache as:
    total_vram * gpu_memory_utilization - weights - peak activations/non-torch - CUDA graphs
In compiled mode (the default) graphs and compile-mode memory together cost far more than the
CUDA graph figure vLLM logs: on the 1.5B model, --enforce-eager freed 1.08 GiB, not 0.35.
and each token costs 2 (K and V) * layers * kv_heads * head_dim * dtype_bytes.
"""

from __future__ import annotations

from pydantic import BaseModel

from bench.config import ModelConfig, ServerConfig

GIB = 1024**3


class KVPrediction(BaseModel):
    total_vram_gib: float
    kv_bytes_per_token: int
    kv_cache_gib: float
    kv_cache_tokens: int
    max_concurrency: float  # at max_model_len tokens per request, as vLLM reports it


def kv_bytes_per_token(model: ModelConfig) -> int:
    a = model.architecture
    return 2 * a.num_layers * a.num_kv_heads * a.head_dim * a.kv_dtype_bytes


def predict_kv(model: ModelConfig, server: ServerConfig, total_vram_gib: float) -> KVPrediction:
    compiled = 0.0 if "--enforce-eager" in server.extra_flags else model.eager_savings_gib
    budget = total_vram_gib * server.gpu_memory_utilization
    kv_gib = max(0.0, budget - model.weights_gib - model.overhead_gib - compiled)
    per_token = kv_bytes_per_token(model)
    tokens = int(kv_gib * GIB / per_token)
    return KVPrediction(
        total_vram_gib=total_vram_gib,
        kv_bytes_per_token=per_token,
        kv_cache_gib=kv_gib,
        kv_cache_tokens=tokens,
        max_concurrency=tokens / server.max_model_len,
    )
