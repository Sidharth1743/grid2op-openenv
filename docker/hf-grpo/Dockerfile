FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1
ENV PATH="/root/.local/bin:/workspace/.venv/bin:${PATH}"
ENV HF_HOME="/data/hf"
ENV HUGGINGFACE_HUB_CACHE="/data/hf/hub"
ENV TRANSFORMERS_CACHE="/data/hf/transformers"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace

COPY . /workspace

RUN uv sync --frozen --no-dev
RUN uv pip install --no-cache-dir -r docker/hf-grpo/requirements-train.txt

CMD ["bash"]
