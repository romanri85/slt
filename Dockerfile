# RunPod Serverless LoRA Training Worker
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv curl zip git git-lfs wget vim \
    libgl1 libglib2.0-0 python3-dev build-essential gcc \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python build tools
RUN pip install --no-cache-dir ninja packaging

FROM base AS final

# PyTorch
RUN pip install torch torchvision torchaudio

# Clone diffusion-pipe
RUN git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe /diffusion_pipe

# Install diffusion-pipe requirements (excluding flash-attn — installed at runtime)
RUN grep -v -i "flash-attn\|flash-attention" /diffusion_pipe/requirements.txt > /tmp/requirements_no_flash.txt && \
    pip install -r /tmp/requirements_no_flash.txt

# Pre-install flash-attn wheel for H100 (sm_90) — most common RunPod GPU
RUN pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl || true

# Upgrade key packages
RUN pip install transformers -U && \
    pip install --upgrade "huggingface_hub[cli]" && \
    pip install --upgrade "peft>=0.17.0" && \
    pip uninstall -y diffusers && \
    pip install git+https://github.com/huggingface/diffusers

# Install handler requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY handler.py /app/handler.py
COPY modules/ /app/modules/
COPY toml_templates/ /app/toml_templates/

WORKDIR /app

# Default network volume path
ENV NETWORK_VOLUME=/runpod-volume

CMD ["python", "handler.py"]
