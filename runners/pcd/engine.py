from __future__ import annotations

from typing import Any

from runners.pcd import engine_torch as _selected_torch

USE_MLX = False
_mlx_engine: Any = None
_torch_engine: Any = None

_torch_engine = _selected_torch
get_engine = _torch_engine.get_engine
run_parallel_generation = _torch_engine.run_parallel_generation
run_naive_generation = _torch_engine.run_naive_generation
stream_naive_generation = _torch_engine.stream_naive_generation
run_rlcd_generation = _torch_engine.run_rlcd_generation

BACKEND = "torch"

__all__ = [
    "USE_MLX",
    "get_engine",
    "run_naive_generation",
    "run_parallel_generation",
    "run_rlcd_generation",
    "stream_naive_generation",
]
