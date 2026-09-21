"""
Unified Engine Router for Parallel Constrained Decoding.
Automatically selects MLX backend on Apple Silicon macOS,
or PyTorch / CUDA backend on Linux, Docker, and Hugging Face Spaces.
"""

import os
import platform

USE_MLX = False
if platform.system() == "Darwin" and os.environ.get("BACKEND", "").lower() != "torch":
    try:
        import mlx.core as mx
        import mlx_lm
        USE_MLX = True
    except Exception:
        USE_MLX = False

if USE_MLX:
    from runners.pcd.engine_mlx import (
        get_engine,
        run_parallel_generation,
        run_naive_generation,
        stream_naive_generation,
        run_rlcd_generation,
    )
else:
    from runners.pcd.engine_torch import (
        get_torch_engine as get_engine,
        run_parallel_generation_torch as run_parallel_generation,
        run_naive_generation_torch as run_naive_generation,
        stream_naive_generation_torch as stream_naive_generation,
    )
    run_rlcd_generation = run_parallel_generation

__all__ = [
    "get_engine",
    "run_parallel_generation",
    "run_naive_generation",
    "stream_naive_generation",
    "run_rlcd_generation",
    "USE_MLX",
]
