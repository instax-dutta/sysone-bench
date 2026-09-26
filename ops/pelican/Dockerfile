FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    XDG_CACHE_HOME=/models/cache \
    TRANSFORMERS_CACHE=/models/huggingface/hub \
    PATH="/workspace/.venv/bin:$PATH"

WORKDIR /workspace
COPY pyproject.toml uv.lock ./
RUN apt-get update \
    && apt-get install -y --no-install-recommends util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir uv==0.8.17 \
    && uv venv --python 3.12 \
    && uv pip install --index-url https://download.pytorch.org/whl/cpu --no-deps torch==2.14.0+cpu \
    && uv export --frozen --no-dev --no-emit-project --extra qwen --no-hashes --no-annotate --no-header --format requirements-txt > /tmp/runtime-requirements.txt \
    && sed -E '/^[[:space:]-]*(cuda|nvidia|triton|torch)([^A-Za-z0-9]|$)/d' /tmp/runtime-requirements.txt > /tmp/runtime-requirements-cpu.txt \
    && ! grep -E '^[[:space:]-]*(cuda|nvidia|triton|torch)([^A-Za-z0-9]|$)' /tmp/runtime-requirements-cpu.txt \
    && uv pip install --no-deps --requirement /tmp/runtime-requirements-cpu.txt \
    && /workspace/.venv/bin/python -c 'import torch; raise SystemExit(1 if torch.version.cuda else 0)' \
    && /workspace/.venv/bin/python -c 'import importlib.util; raise SystemExit(1 if any(importlib.util.find_spec(name) is not None for name in ("nvidia", "triton")) else 0)' \
    && rm -rf /root/.cache/uv
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin sysone \
    && mkdir -p /workspace /models
COPY --chown=sysone:sysone benchmark ./benchmark
COPY --chown=sysone:sysone datasets/__init__.py ./datasets/
COPY --chown=sysone:sysone runners ./runners
COPY --chown=sysone:sysone schemas ./schemas
COPY --chown=sysone:sysone run.py compare.py ./
USER sysone
