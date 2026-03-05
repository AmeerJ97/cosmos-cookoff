FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Python training deps
RUN pip install --no-cache-dir \
    transformers>=4.51.0 \
    peft>=0.13.0 \
    bitsandbytes>=0.44.0 \
    datasets>=2.19.0 \
    accelerate>=0.34.0 \
    trl>=0.12.0 \
    scipy \
    sentencepiece \
    "jinja2>=3.1.0"

# Flash Attention 2 (A100 optimized)
RUN pip install --no-cache-dir flash-attn --no-build-isolation 2>/dev/null || true

# Copy training script
COPY scripts/train_qlora.py train_qlora.py
COPY scripts/vertex_train.py vertex_train.py

# Copy SFT data (baked into container for fast startup)
COPY data/sft_dataset.openai.jsonl sft_dataset.openai.jsonl

# Default: run QLoRA training
ENTRYPOINT ["python3", "train_qlora.py"]
CMD ["--data", "sft_dataset.openai.jsonl", "--output", "/gcs/output/clasp-lora-checkpoint"]
