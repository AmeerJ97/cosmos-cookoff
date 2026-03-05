# Vertex AI VLM Fine-Tuning Pipeline: Research & Implementation Guide
## For ABEE Project — NVIDIA Cosmos-Reason2-8B on Google Cloud Vertex AI

**Research Date:** 2026-03-05
**Author:** Research synthesis via Claude Code
**Status:** Reference document — validate pricing against official Google Cloud pricing page before committing to any job

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Vertex AI Training Architecture Overview](#vertex-ai-training-architecture-overview)
3. [Compute Resources: GPU Types and Pricing](#compute-resources-gpu-types-and-pricing)
4. [Custom Container Training for VLMs](#custom-container-training-for-vlms)
5. [SFT Dataset Pipeline: Format and GCS Layout](#sft-dataset-pipeline-format-and-gcs-layout)
6. [Full Train / Validation / Test Pipeline](#full-train--validation--test-pipeline)
7. [Experiment Tracking, Hyperparameter Tuning, Model Registry](#experiment-tracking-hyperparameter-tuning-model-registry)
8. [NVIDIA NIM + Vertex AI Integration](#nvidia-nim--vertex-ai-integration)
9. [Cost Optimization Strategies](#cost-optimization-strategies)
10. [Cosmos-Reason2-8B Specific Considerations](#cosmos-reason2-8b-specific-considerations)
11. [ABEE Project Recommended Architecture](#abee-project-recommended-architecture)
12. [Implementation Quickstart](#implementation-quickstart)
13. [Known Limitations and Gotchas](#known-limitations-and-gotchas)
14. [Sources and References](#sources-and-references)

---

## Executive Summary

Google Vertex AI supports full end-to-end fine-tuning of vision-language models through its **Custom Training Jobs** API using custom Docker containers. Cosmos-Reason2-8B is based on Qwen3-VL-8B-Instruct and can be fine-tuned using the cosmos-reason2 GitHub repository's training scripts (built on cosmos-rl/NeMo), packaged into a container, and submitted to Vertex AI.

Key findings:

- Vertex AI does **not** offer a managed SFT pipeline for arbitrary open-source VLMs — you must use the **Custom Training Job** path with a custom container image.
- Managed SFT exists only for Google-first models (Gemini) and a small set of Model Garden models (Llama 3.x, Mistral 7B). Cosmos-Reason2 is not in Model Garden.
- An A100 40GB single GPU job costs ~$3.67/hr on-demand; A100 80GB (a2-ultragpu) runs ~$5.50/hr; H100 SXM is ~$6–$8/hr. Spot discounts run 60–91% off but apply to preemptible workloads only.
- Dynamic Workload Scheduler (DWS) with `FLEX_START` is the recommended cost path for training jobs that can tolerate queuing (not immediate-start).
- NVIDIA NIM has limited native Vertex AI support (one NeMo Retriever model confirmed). For Cosmos-Reason2 inference, deploy with vLLM on Vertex AI Endpoints using GPU machines.
- The ABEE project's SFT dataset output (JSONL conversation format) maps directly to what Vertex AI custom training containers can consume from GCS.

---

## Vertex AI Training Architecture Overview

Vertex AI provides two training pathways:

### Pathway 1: Managed Tuning (Model Garden / Generative AI)
- **Supported models:** Gemini 1.5/2.0, Gemini Flash, Llama 3.x, Mistral 7B
- **Interface:** `vertexai.tuning.sft.train()` or Console UI
- **Data format:** JSONL uploaded to GCS
- **Not applicable** to Cosmos-Reason2 — model is not in this catalog

### Pathway 2: Custom Training Jobs (the correct path for ABEE)
- Submit a **Custom Job** with a Docker container containing your training code
- Full control over model, framework (PyTorch, NeMo, HuggingFace TRL), dataset loading
- GPU/TPU machine selection with any accelerator type
- Supports multi-node DDP training via NCCL
- Outputs saved to GCS bucket (`AIP_MODEL_DIR`, `AIP_CHECKPOINT_DIR`)
- Integrates with Vertex AI Experiments (TensorBoard), Vizier (hyperparameter tuning), and Model Registry

### Core Vertex AI SDK Object Hierarchy

```
Project
  └── Location (region, e.g., us-central1)
        ├── CustomJob / CustomTrainingJob
        │     ├── WorkerPoolSpec (machine_type, accelerator, container_uri)
        │     ├── Scheduling (SPOT / FLEX_START / STANDARD)
        │     └── Environment Variables (auto-set: AIP_MODEL_DIR, AIP_TENSORBOARD_LOG_DIR)
        ├── Experiment / ExperimentRun
        ├── TensorBoard instance
        ├── HyperparameterTuningJob (Vizier-backed)
        └── Model Registry
              └── Model (artifact URI → GCS)
                    └── Endpoint (vLLM serving container)
```

---

## Compute Resources: GPU Types and Pricing

### Available GPU Accelerators in Vertex AI Training (us-central1)

| GPU Type | Machine Series | VRAM | On-Demand $/GPU/hr | Spot $/GPU/hr (approx) | Notes |
|---|---|---|---|---|---|
| NVIDIA T4 | n1-standard-* | 16 GB | ~$0.35 | ~$0.11 | Inference-only for large VLMs; too small for 8B |
| NVIDIA L4 | g2-standard-* | 24 GB | ~$0.71 | ~$0.21 | Borderline for 8B with 4-bit quant; good for inference |
| NVIDIA A100 40GB | a2-highgpu-* | 40 GB | ~$3.67 | ~$1.10–$1.47 | Primary training target for 8B models |
| NVIDIA A100 80GB | a2-ultragpu-* | 80 GB | ~$5.50 | ~$1.65–$2.20 | More comfortable headroom; needed for full fine-tuning |
| NVIDIA H100 80GB | a3-highgpu-* | 80 GB | ~$6.98 | ~$2.10–$2.80 | Fastest; use for multi-node or large batch SFT |
| NVIDIA H200 141GB | a3-megagpu-* | 141 GB | ~$8.50+ | ~$2.50+ | Overkill for 8B; good for 70B+ |
| NVIDIA B200 | a4-* | 192 GB | TBD (very recent) | TBD | Bleeding edge; check availability |

**DWS-eligible GPUs:** L4, A100, H100, H200, B200 — these can use `FLEX_START` scheduling.

**Billing granularity:** 30-second increments.

### Recommended Configuration for Cosmos-Reason2-8B SFT

| Scenario | Machine | GPUs | Per-hour (on-demand) | Per-hour (spot/flex) |
|---|---|---|---|---|
| LoRA / QLoRA (4-bit) | n1-standard-8 + A100 40GB | 1 | ~$3.67 | ~$1.10–$1.47 |
| Full fine-tuning (BF16) | a2-ultragpu-1g + A100 80GB | 1 | ~$5.50 | ~$1.65 |
| Full fine-tuning (BF16) distributed | a2-highgpu-4g + A100 40GB | 4 | ~$14.68 | ~$4.40 |
| Fast full fine-tuning | a3-highgpu-2g + H100 80GB | 2 | ~$13.96 | ~$4.20 |

**Machine type note:** The `machine_type` string in Vertex AI refers to the Compute Engine machine series (e.g., `a2-highgpu-1g` for 1x A100 40GB). Confirm exact strings with `gcloud compute machine-types list --filter="zone:us-central1-*"`.

### Pricing Warning

These prices are estimates from multiple sources as of early 2026. Always verify against the official pricing page before submitting long jobs:
- https://cloud.google.com/vertex-ai/pricing
- https://cloud.google.com/compute/gpus-pricing

---

## Custom Container Training for VLMs

### Container Strategy for Cosmos-Reason2

Cosmos-Reason2's training stack requires:
- Python 3.10+
- PyTorch 2.x with CUDA 12.x
- `cosmos-reason2` package (from GitHub) + `cosmos-rl` dependencies
- HuggingFace `transformers`, `accelerate`, `peft`, `trl`
- Optional: `flash-attn` for memory efficiency

Two container build options:

#### Option A: Build on NVIDIA PyTorch Base (Recommended)

```dockerfile
# Dockerfile.cosmos_train
FROM nvcr.io/nvidia/pytorch:24.12-py3

WORKDIR /workspace

# Install cosmos-reason2 training dependencies
RUN pip install --no-cache-dir \
    transformers==4.47.0 \
    accelerate==1.2.1 \
    peft==0.14.0 \
    trl==0.13.0 \
    bitsandbytes==0.45.0 \
    flash-attn==2.7.4.post1 \
    qwen-vl-utils \
    google-cloud-storage \
    google-cloud-aiplatform

# Clone cosmos-reason2 for training scripts
RUN git clone https://github.com/nvidia-cosmos/cosmos-reason2.git /workspace/cosmos-reason2

# Copy your custom training entry point
COPY train_abee.py /workspace/train_abee.py

ENV PYTHONPATH="/workspace/cosmos-reason2:${PYTHONPATH}"

ENTRYPOINT ["python", "/workspace/train_abee.py"]
```

#### Option B: Use HuggingFace PyTorch DLC (Simpler)

Google provides a pre-built HuggingFace TRL container that works with Vertex AI:

```
us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-cu121.2-2.transformers.4-44.ubuntu2204.py311
```

This container includes HuggingFace TRL/PEFT/transformers and is directly usable for SFT with LoRA.

### Build and Push Container

```bash
# Set variables
PROJECT_ID="your-gcp-project-id"
REGION="us-central1"
REPO="cosmos-training"
IMAGE_NAME="cosmos-reason2-trainer"
IMAGE_TAG="v1.0"

# Create Artifact Registry repository
gcloud artifacts repositories create ${REPO} \
    --repository-format=docker \
    --location=${REGION} \
    --project=${PROJECT_ID}

# Build and push
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${IMAGE_TAG} \
    -f Dockerfile.cosmos_train .

docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${IMAGE_TAG}
```

### Training Script Entry Point Pattern

Your training script must handle Vertex AI environment variables:

```python
# train_abee.py — Vertex AI training entry point

import os
import json
import argparse
from pathlib import Path

# Vertex AI injects these environment variables automatically
MODEL_DIR = os.environ.get("AIP_MODEL_DIR", "/tmp/model_output")
CHECKPOINT_DIR = os.environ.get("AIP_CHECKPOINT_DIR", "/tmp/checkpoints")
TENSORBOARD_LOG_DIR = os.environ.get("AIP_TENSORBOARD_LOG_DIR", "/tmp/tb_logs")
DATA_FORMAT = os.environ.get("AIP_DATA_FORMAT", "jsonl")  # or "csv"

# Your GCS paths passed as custom args
parser = argparse.ArgumentParser()
parser.add_argument("--train-data-uri", type=str, required=True,
                    help="GCS URI for training JSONL, e.g. gs://bucket/data/train.jsonl")
parser.add_argument("--val-data-uri", type=str, required=True)
parser.add_argument("--base-model-id", type=str,
                    default="nvidia/Cosmos-Reason2-8B")
parser.add_argument("--lora-rank", type=int, default=16)
parser.add_argument("--lora-alpha", type=int, default=32)
parser.add_argument("--learning-rate", type=float, default=2e-4)
parser.add_argument("--num-epochs", type=int, default=3)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--use-4bit", action="store_true", default=True)
args = parser.parse_args()


def download_from_gcs(gcs_uri: str, local_path: str) -> str:
    """Download a GCS file to local path."""
    from google.cloud import storage
    client = storage.Client()
    bucket_name = gcs_uri.replace("gs://", "").split("/")[0]
    blob_path = "/".join(gcs_uri.replace("gs://", "").split("/")[1:])
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    return local_path


def upload_to_gcs(local_path: str, gcs_uri: str):
    """Upload a local file/dir to GCS."""
    from google.cloud import storage
    import subprocess
    subprocess.run(["gsutil", "-m", "cp", "-r", local_path, gcs_uri], check=True)


def train():
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer

    # --- 1. Download datasets from GCS ---
    download_from_gcs(args.train_data_uri, "/tmp/train.jsonl")
    download_from_gcs(args.val_data_uri, "/tmp/val.jsonl")

    # --- 2. Load model ---
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_id,
        trust_remote_code=True,
    )

    # --- 3. LoRA config ---
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # --- 4. Training arguments ---
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_dir=TENSORBOARD_LOG_DIR,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",
        dataloader_num_workers=4,
    )

    # --- 5. Load dataset ---
    from datasets import load_dataset
    train_dataset = load_dataset("json", data_files="/tmp/train.jsonl", split="train")
    val_dataset = load_dataset("json", data_files="/tmp/val.jsonl", split="train")

    # --- 6. Train ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",  # or custom formatting_func
        max_seq_length=4096,
    )
    trainer.train()

    # --- 7. Save and upload to AIP_MODEL_DIR ---
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    train()
```

---

## SFT Dataset Pipeline: Format and GCS Layout

### ABEE SFT Dataset Format (JSONL)

The ABEE system generates SFT samples from its curated game episodes. Each sample should be one JSON object per line, in the multi-turn conversation format compatible with Qwen3-VL / Cosmos-Reason2:

```jsonl
{"id": "abee_ep001_frame012", "conversations": [{"role": "user", "content": [{"type": "video", "video": "gs://abee-data/videos/ep001.mp4", "video_start": 0.0, "video_end": 4.0, "fps": 2}, {"type": "text", "text": "Analyze the robot arm trajectory and determine if it is safe to initiate human handoff. Reason step by step."}]}, {"role": "assistant", "content": "<think>\nThe robot arm is in deceleration phase. Velocity vectors converge to near-zero at frame 11. The human is positioned at 45-degree offset, outside the kinematic envelope. The gripper load has been released as indicated by the torque signal drop at frame 9. All motion has settled.\n</think>\n\nACT: RELEASE — Safe handoff window confirmed. Robot velocity < 0.02 m/s, human outside kinematic reach envelope, gripper empty."}]}
```

For image-based (frame-level) samples:

```jsonl
{"id": "abee_ep002_frame008", "conversations": [{"role": "user", "content": [{"type": "image", "image": "gs://abee-data/frames/ep002_f008.jpg"}, {"type": "text", "text": "What is the current state of the robot arm? Is it safe to release the human?"}]}, {"role": "assistant", "content": "<think>\nArm is mid-trajectory. Velocity appears non-zero. Human within 0.8m range.\n</think>\n\nTHINK: Trajectory incomplete. Do not release."}]}
```

### GCS Bucket Layout

```
gs://abee-sft-data/
  ├── raw/
  │   └── episodes/              # Raw video + annotation JSON from ABEE orchestrator
  ├── processed/
  │   ├── train.jsonl            # 70% of curated SFT samples
  │   ├── val.jsonl              # 15% validation split
  │   └── test.jsonl             # 15% held-out test split
  ├── splits/
  │   ├── fold_0/                # Optional: k-fold cross-validation
  │   │   ├── train.jsonl
  │   │   └── val.jsonl
  │   └── fold_1/ ...
  └── artifacts/
      ├── checkpoints/           # AIP_CHECKPOINT_DIR target
      └── models/                # AIP_MODEL_DIR target (final model)
          └── run_20260305_001/
```

### Data Split Recommendation for ABEE

Given the sequential nature of ABEE episodes (temporal ordering matters):

```
Split Strategy: Temporal Split (NOT random)
  - Train: Episodes 1 through N*0.70 (chronological order)
  - Validation: Episodes N*0.70 through N*0.85
  - Test: Episodes N*0.85 onward (never seen during training)
```

Random splits would cause temporal leakage — earlier frames in a sequence could appear in training while later frames appear in validation, which is not realistic for this task.

### Data Upload Script

```python
# upload_sft_data.py
import json
import random
from pathlib import Path
from google.cloud import storage

def upload_split_datasets(
    raw_samples_path: str,
    bucket_name: str,
    gcs_prefix: str = "processed",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    temporal_split: bool = True,
    seed: int = 42,
):
    """Split and upload ABEE SFT samples to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    with open(raw_samples_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    n = len(samples)
    if temporal_split:
        # Maintain episode ordering — no shuffle
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        train_set = samples[:train_end]
        val_set = samples[train_end:val_end]
        test_set = samples[val_end:]
    else:
        random.seed(seed)
        random.shuffle(samples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        train_set = samples[:train_end]
        val_set = samples[train_end:val_end]
        test_set = samples[val_end:]

    for split_name, split_data in [("train", train_set), ("val", val_set), ("test", test_set)]:
        blob = bucket.blob(f"{gcs_prefix}/{split_name}.jsonl")
        content = "\n".join(json.dumps(s) for s in split_data)
        blob.upload_from_string(content, content_type="application/jsonl")
        print(f"Uploaded {len(split_data)} samples to gs://{bucket_name}/{gcs_prefix}/{split_name}.jsonl")
```

---

## Full Train / Validation / Test Pipeline

### Option A: Simple Custom Job (Recommended Starting Point)

```python
# submit_training_job.py
import google.cloud.aiplatform as aiplatform
from datetime import datetime

PROJECT_ID = "your-gcp-project"
REGION = "us-central1"
BUCKET = "gs://abee-sft-data"
CONTAINER_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/cosmos-training/cosmos-reason2-trainer:v1.0"

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

job = aiplatform.CustomJob(
    display_name=f"abee-cosmos-sft-{timestamp}",
    worker_pool_specs=[
        {
            "machine_spec": {
                "machine_type": "a2-highgpu-1g",   # 1x A100 40GB
                "accelerator_type": "NVIDIA_TESLA_A100",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": CONTAINER_URI,
                "args": [
                    "--train-data-uri", f"{BUCKET}/processed/train.jsonl",
                    "--val-data-uri", f"{BUCKET}/processed/val.jsonl",
                    "--base-model-id", "nvidia/Cosmos-Reason2-8B",
                    "--lora-rank", "16",
                    "--learning-rate", "2e-4",
                    "--num-epochs", "3",
                    "--batch-size", "4",
                    "--use-4bit",
                ],
                "env": [
                    # Additional custom env vars (Vertex sets AIP_* automatically)
                    {"name": "WANDB_DISABLED", "value": "true"},
                    {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
                ],
            },
        }
    ],
    # Use Spot VM to cut costs ~60-70%
    scheduling=aiplatform.gapic.Scheduling(
        strategy=aiplatform.gapic.Scheduling.Strategy.SPOT
    ),
    base_output_dir=f"{BUCKET}/artifacts",
)

job.run(sync=True)
print(f"Job complete. Model at: {job.output_info.artifact_uri}")
```

### Option B: Full Vertex AI Pipeline (KFP v2 / Kubeflow)

For reproducible, tracked runs with automatic data versioning and model registration:

```python
# pipeline_abee_sft.py
from kfp import dsl
from kfp.v2 import compiler
import google.cloud.aiplatform as aiplatform
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp

PROJECT_ID = "your-gcp-project"
REGION = "us-central1"
PIPELINE_ROOT = "gs://abee-sft-data/pipeline_runs"
CONTAINER_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/cosmos-training/cosmos-reason2-trainer:v1.0"


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-storage", "scikit-learn"],
)
def prepare_and_split_data(
    raw_data_uri: str,
    bucket_name: str,
    output_prefix: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> str:
    """Component: split JSONL data temporally and upload to GCS."""
    import json
    from google.cloud import storage

    client = storage.Client()
    # Download raw data
    blob_path = raw_data_uri.replace(f"gs://{bucket_name}/", "")
    bucket = client.bucket(bucket_name)
    content = bucket.blob(blob_path).download_as_text()
    samples = [json.loads(l) for l in content.splitlines() if l.strip()]

    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    splits = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }

    for name, data in splits.items():
        blob = bucket.blob(f"{output_prefix}/{name}.jsonl")
        blob.upload_from_string("\n".join(json.dumps(s) for s in data))
        print(f"{name}: {len(data)} samples")

    return f"gs://{bucket_name}/{output_prefix}"


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-aiplatform"],
)
def register_model(
    artifact_uri: str,
    display_name: str,
    project: str,
    location: str,
) -> str:
    """Component: register trained model in Vertex AI Model Registry."""
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=location)
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=(
            "us-docker.pkg.dev/vertex-ai/vertex-model-garden-pytorch/pytorch-vllm-serve:20250204_0916_RC00"
        ),
        serving_container_args=[
            "--model", display_name,
            "--tensor-parallel-size", "1",
            "--max-model-len", "4096",
        ],
    )
    print(f"Registered model: {model.resource_name}")
    return model.resource_name


@dsl.pipeline(
    name="abee-cosmos-sft-pipeline",
    description="ABEE SFT training pipeline for Cosmos-Reason2-8B",
    pipeline_root=PIPELINE_ROOT,
)
def abee_sft_pipeline(
    raw_data_uri: str,
    bucket_name: str,
    run_id: str,
    lora_rank: int = 16,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
):
    # Step 1: Prepare data splits
    data_prep = prepare_and_split_data(
        raw_data_uri=raw_data_uri,
        bucket_name=bucket_name,
        output_prefix=f"runs/{run_id}/data",
    )

    # Step 2: Train
    training_job = CustomTrainingJobOp(
        project=PROJECT_ID,
        location=REGION,
        display_name=f"abee-sft-{run_id}",
        worker_pool_specs=[{
            "machine_spec": {
                "machine_type": "a2-highgpu-1g",
                "accelerator_type": "NVIDIA_TESLA_A100",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": CONTAINER_URI,
                "args": [
                    "--train-data-uri",
                    f"{data_prep.output}/train.jsonl",
                    "--val-data-uri",
                    f"{data_prep.output}/val.jsonl",
                    "--lora-rank", str(lora_rank),
                    "--learning-rate", str(learning_rate),
                    "--num-epochs", str(num_epochs),
                    "--use-4bit",
                ],
            },
        }],
        base_output_dir=f"gs://{bucket_name}/runs/{run_id}/artifacts",
        scheduling={
            "strategy": "SPOT",
        },
    ).after(data_prep)

    # Step 3: Register model
    register_model(
        artifact_uri=f"gs://{bucket_name}/runs/{run_id}/artifacts/model",
        display_name=f"abee-cosmos-reason2-{run_id}",
        project=PROJECT_ID,
        location=REGION,
    ).after(training_job)


# Compile and submit
compiler.Compiler().compile(abee_sft_pipeline, "abee_sft_pipeline.json")

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=PIPELINE_ROOT)
pipeline_job = aiplatform.PipelineJob(
    display_name="abee-sft-run-001",
    template_path="abee_sft_pipeline.json",
    parameter_values={
        "raw_data_uri": "gs://abee-sft-data/raw/all_curated_samples.jsonl",
        "bucket_name": "abee-sft-data",
        "run_id": "run_20260305",
        "lora_rank": 16,
        "learning_rate": 2e-4,
        "num_epochs": 3,
    },
)
pipeline_job.run(sync=True)
```

---

## Experiment Tracking, Hyperparameter Tuning, Model Registry

### Vertex AI Experiments + TensorBoard

Vertex AI Experiments wrap your training runs with metadata tracking. Your training container logs to `AIP_TENSORBOARD_LOG_DIR` and Vertex AI streams the data to a managed TensorBoard instance.

```python
# Setup before submitting jobs
import google.cloud.aiplatform as aiplatform

aiplatform.init(project=PROJECT_ID, location=REGION)

# Create a TensorBoard instance (one-time)
tensorboard = aiplatform.Tensorboard.create(
    display_name="abee-training-board",
    project=PROJECT_ID,
    location=REGION,
)

# Create an experiment
experiment = aiplatform.Experiment.create(
    experiment_name="abee-cosmos-sft-experiments",
    description="ABEE SFT experiments for Cosmos-Reason2-8B",
    tensorboard=tensorboard.resource_name,
)

# Log custom metrics from within training
with aiplatform.start_run(run="run-20260305-001") as run:
    run.log_params({"lora_rank": 16, "lr": 2e-4, "epochs": 3})
    # Log metrics at each step from your training loop:
    run.log_metrics({"train_loss": 1.23, "val_loss": 1.18, "step": 100})
```

### Hyperparameter Tuning with Vertex AI Vizier

Vizier uses Bayesian optimization. Your training script must report the metric back via `cloudml-hypertune`:

```python
# In your training script (train_abee.py)
import hypertune

hpt = hypertune.HyperTune()

# After each epoch, report the metric
def report_eval_metric(eval_loss: float, step: int):
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="eval_loss",
        metric_value=eval_loss,
        global_step=step,
    )
```

Submit the HPT job:

```python
from google.cloud.aiplatform import HyperparameterTuningJob
from google.cloud.aiplatform.aiplatform import gapic

hpt_job = HyperparameterTuningJob(
    display_name="abee-cosmos-hpt",
    custom_job=job,  # your CustomJob spec from above
    metric_spec={"eval_loss": "minimize"},
    parameter_spec={
        "lora_rank": aiplatform.IntegerParameterSpec(min=8, max=64, scale="log"),
        "learning_rate": aiplatform.DoubleParameterSpec(min=1e-5, max=1e-3, scale="log"),
        "batch_size": aiplatform.DiscreteParameterSpec(values=[2, 4, 8], scale="linear"),
    },
    max_trial_count=12,
    parallel_trial_count=2,
)
hpt_job.run(sync=True)
```

### Model Registry

After training, register your model artifact for version tracking and deployment:

```python
model = aiplatform.Model.upload(
    display_name="abee-cosmos-reason2-8b-lora-v1",
    artifact_uri="gs://abee-sft-data/artifacts/models/run_20260305_001",
    serving_container_image_uri=(
        # Use Vertex AI's vLLM container for inference
        "us-docker.pkg.dev/vertex-ai/vertex-model-garden-pytorch/pytorch-vllm-serve:20250204_0916_RC00"
    ),
    serving_container_predict_route="/generate",
    serving_container_health_route="/health",
    serving_container_ports=[7080],
    serving_container_args=[
        "--model", "/model",
        "--tensor-parallel-size", "1",
        "--max-model-len", "8192",
        "--trust-remote-code",
        "--enable-lora",
    ],
    labels={
        "project": "abee",
        "model_type": "cosmos_reason2_8b",
        "tuning": "lora_r16",
    },
)
model.wait()
print(f"Model: {model.resource_name}")
```

---

## NVIDIA NIM + Vertex AI Integration

### Current State (March 2026)

NVIDIA NIM's native Vertex AI integration is limited. The officially documented deployment of NIM on Vertex AI covers only the **NeMo Retriever Text Embedding** model (NV-EmbedQA-E5-v5). Cosmos-Reason2 is NOT deployable as a managed NIM container on Vertex AI.

**However**, NIM containers can be deployed manually on Vertex AI using the custom container serving path:

```
NIM Inference Architecture on Vertex AI:
  Vertex AI Endpoint
    └── Custom serving container: nvcr.io/nim/nvidia/cosmos-reason2-8b:latest
          └── GPU machine: a2-highgpu-1g (A100 40GB)
          └── NVIDIA_API_KEY env variable required
```

### NIM-on-Vertex Deployment (Manual Path)

```python
# deploy_nim_endpoint.py
import google.cloud.aiplatform as aiplatform

aiplatform.init(project=PROJECT_ID, location=REGION)

# Upload a stub model (NIM reads from its own cache, not model artifacts)
model = aiplatform.Model.upload(
    display_name="cosmos-reason2-8b-nim",
    artifact_uri="gs://abee-sft-data/artifacts/nim-placeholder/",  # empty dir
    serving_container_image_uri="nvcr.io/nim/nvidia/cosmos-reason2-8b:latest",
    serving_container_environment_variables={
        "NVIDIA_API_KEY": "YOUR_NVIDIA_API_KEY",
        "NIM_CACHE_PATH": "/model-store",
    },
    serving_container_ports=[8000],
    serving_container_predict_route="/v1/chat/completions",
    serving_container_health_route="/v1/health/ready",
)

endpoint = model.deploy(
    machine_type="a2-highgpu-1g",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=1,
    min_replica_count=1,
    max_replica_count=1,
)
```

**Caveats:**
1. NIM containers pull model weights from NVIDIA NGC on first boot — this requires outbound internet access from the serving container and a valid `NGC_API_KEY`.
2. NIM containers are large (~20GB+ compressed). Cold start times are significant.
3. For ABEE's cloud distillation use case (Claude 3.5 Sonnet generating teacher labels), NIM on Vertex is overkill. Use NVIDIA's hosted NIM API (`https://build.nvidia.com/nvidia/cosmos-reason2-8b`) directly via API calls.

### Recommended Inference Path for ABEE

| Use Case | Recommended Approach |
|---|---|
| Local development / ABEE agent inference | Local cosmos-reason2 4-bit via `local_inference.py` |
| Cloud distillation (teacher labeling) | NVIDIA hosted NIM API (`build.nvidia.com`) |
| Evaluation of fine-tuned LoRA | vLLM on Vertex AI Endpoint (custom container) |
| Production serving | vLLM with `--enable-lora` on Vertex AI Endpoint |

---

## Cost Optimization Strategies

### Strategy 1: Spot VMs (Best for Fault-Tolerant Training)

Spot VMs can reduce GPU costs by 60–91%. Requirements:
- Your training script must implement **checkpointing** (save to GCS every N steps)
- Spot VMs can be preempted with 30-second notice
- Use `transformers.TrainingArguments` with `save_strategy="steps"` and `save_steps=100`

```python
# In your CustomJob spec:
scheduling=aiplatform.gapic.Scheduling(
    strategy=aiplatform.gapic.Scheduling.Strategy.SPOT
)
```

### Strategy 2: Dynamic Workload Scheduler (FLEX_START)

For A100/H100/H200/B200 jobs that can tolerate queue wait time (minutes to hours):

```python
scheduling=aiplatform.gapic.Scheduling(
    strategy=aiplatform.gapic.Scheduling.Strategy.FLEX_START
)
```

DWS uses preemptible quota but resources are **not actually preemptible** once running — they behave like standard resources. You pay DWS pricing (cheaper than on-demand, more stable than spot).

### Strategy 3: Right-Size the Accelerator

For 8B parameter models with LoRA + 4-bit quantization:

| Configuration | VRAM Usage (approx) | Can Use |
|---|---|---|
| 4-bit QLoRA, batch=1 | ~8–10 GB | L4 (24GB), A100 40GB, A100 80GB |
| 4-bit QLoRA, batch=4 | ~14–18 GB | A100 40GB, A100 80GB |
| BF16 LoRA, batch=4 | ~28–35 GB | A100 80GB, H100 |
| Full FP16 fine-tune, batch=4 | ~65–80 GB | A100 80GB (barely), H100 |

Start with L4 for development iteration (cheapest with DWS), move to A100 40GB for real training runs.

### Strategy 4: Minimize Idle Time

- Use `sync=False` to submit jobs and poll separately
- Pre-download model weights to GCS and load from GCS (avoids HuggingFace download delay billing)
- Use `per_device_train_batch_size` + `gradient_accumulation_steps` to simulate large batches without requiring more GPUs

### Strategy 5: Preload Model Weights to GCS

Avoid re-downloading Cosmos-Reason2-8B (~16GB) from HuggingFace every training run:

```bash
# One-time: download to GCS
pip install huggingface_hub google-cloud-storage
python -c "
from huggingface_hub import snapshot_download
import subprocess
path = snapshot_download('nvidia/Cosmos-Reason2-8B', local_dir='/tmp/cosmos-reason2-8b')
subprocess.run(['gsutil', '-m', 'cp', '-r', path, 'gs://abee-sft-data/model_cache/cosmos-reason2-8b/'])
"
```

Then in your container, load from GCS instead:

```python
# Load model weights from GCS cache instead of HuggingFace Hub
import subprocess
subprocess.run(["gsutil", "-m", "cp", "-r",
    "gs://abee-sft-data/model_cache/cosmos-reason2-8b/",
    "/tmp/base_model/"], check=True)
model = AutoModelForCausalLM.from_pretrained("/tmp/base_model/", ...)
```

### Estimated Cost for a Typical ABEE SFT Run

Assumptions: 1000 SFT samples, 3 epochs, A100 40GB spot, ~2 hours training time

| Component | Estimated Cost |
|---|---|
| A100 40GB spot (2hr) | ~$2.20–$2.94 |
| GCS storage (50GB, 1 month) | ~$1.00 |
| Artifact Registry storage | ~$0.10 |
| Network egress (minimal) | ~$0.10 |
| **Total per run** | **~$3.50–$4.50** |

For 10 HPT trials: ~$35–$45 total. Very affordable at this scale.

---

## Cosmos-Reason2-8B Specific Considerations

### Architecture Notes

- **Base:** Qwen3-VL-8B-Instruct (confirmed by NVIDIA documentation)
- **Architecture:** Vision Transformer (ViT) + Dense Transformer LLM
- **Input:** Text + image/video (multimodal)
- **Training paradigm:** SFT + Reinforcement Learning (GRPO/PPO)
- **Normalized coordinates:** Uses 0–1024 range for bounding boxes (Qwen3-VL convention)

### Fine-Tuning Approach Options

| Method | VRAM Required | Quality | Training Speed | Recommendation |
|---|---|---|---|---|
| Full fine-tuning (BF16) | ~65–80 GB | Best | Slowest | A100 80GB x1 or A100 40GB x2 |
| LoRA (r=16, BF16) | ~24–30 GB | Very good | Fast | A100 40GB x1 |
| QLoRA (r=16, 4-bit) | ~10–14 GB | Good | Medium | L4 or A100 40GB x1 |
| DoRA (Weight-Decomposed LoRA) | ~24–30 GB | Better than LoRA | Similar to LoRA | A100 40GB x1 |

**ABEE Recommendation:** Start with QLoRA (4-bit, r=16) on a single A100 40GB spot VM. The Life-Points game mechanics produce high-quality behavioral signal per sample — you need fewer samples than typical SFT, so QLoRA quality should be sufficient.

### Official Training Framework

The cosmos-reason2 GitHub repository uses `cosmos-rl`, which is configured via TOML files:

```toml
# Example: abee_sft_config.toml
[model]
model_name_or_path = "nvidia/Cosmos-Reason2-8B"
trust_remote_code = true

[training]
learning_rate = 2e-4
num_train_epochs = 3
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
warmup_ratio = 0.05
lr_scheduler_type = "cosine"
bf16 = true
output_dir = "/tmp/checkpoints"

[lora]
use_lora = true
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

[data]
annotation_path = "/tmp/train.jsonl"
media_path = "/tmp/frames/"
val_annotation_path = "/tmp/val.jsonl"

# Weighted data blending (for multi-source ABEE datasets)
[[data.sources]]
name = "abee_high_quality"
weight = 0.60

[[data.sources]]
name = "abee_standard"
weight = 0.40
```

### Video vs. Frame Input for ABEE

ABEE generates both frame-level and sequence-level observations. Both are supported by Cosmos-Reason2:

- **Frame-level:** Each training sample = single image + ACT/THINK label
  - Simpler to manage in JSONL
  - Loses temporal context within the sample
  - Better for high-precision labeling

- **Sequence-level:** Each training sample = short video clip (4–8 frames) + chain-of-thought + ACT/THINK
  - More representative of how agents see the world
  - Larger data size; need to store video clips in GCS
  - Matches Cosmos-Reason2's training distribution better

Recommendation: use sequence-level (short video clips) for final training runs, frame-level for fast iteration and debugging.

---

## ABEE Project Recommended Architecture

### Phase 1: Development (Local + Vertex AI Dev)

```
[Local RTX 4060 Ti 16GB]
  - cosmos-reason2 4-bit inference for ABEE game loop
  - Collect SFT samples from game episodes
  - Upload SFT JSONL to GCS bucket

[Vertex AI — L4 Spot VM, ~$0.21/hr with DWS]
  - Quick SFT iteration runs (few hundred samples)
  - Validate training pipeline end-to-end
  - Verify data format compatibility
```

### Phase 2: Full SFT Training

```
[Vertex AI — A100 40GB Spot or FLEX_START]
  - Full 3-epoch SFT run on curated ABEE dataset
  - Hyperparameter search: 8–12 Vizier trials
  - TensorBoard monitoring via Vertex AI Experiments
  - Model registered in Vertex AI Model Registry
```

### Phase 3: Evaluation and Distillation

```
[Vertex AI Endpoint — vLLM with LoRA]
  - Serve fine-tuned Cosmos-Reason2-8B for evaluation
  - Run test set through endpoint; compute ACT/THINK accuracy

[NVIDIA NIM API (build.nvidia.com) — for teacher distillation]
  - Cloud Cosmos-Reason2-8B for label generation
  - Vertex AI Claude 3.5 Sonnet via Vertex AI Model Garden
    OR direct Anthropic API for teacher signal
```

### GCS Bucket Structure

```
gs://abee-sft-data/          # Main data bucket
gs://abee-model-artifacts/   # Model outputs, checkpoints
gs://abee-pipeline-root/     # KFP pipeline run artifacts
```

---

## Implementation Quickstart

### Step 1: GCP Setup

```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable \
    aiplatform.googleapis.com \
    artifactregistry.googleapis.com \
    storage.googleapis.com \
    compute.googleapis.com

# Create GCS buckets
gsutil mb -l us-central1 gs://abee-sft-data
gsutil mb -l us-central1 gs://abee-model-artifacts
gsutil mb -l us-central1 gs://abee-pipeline-root

# Create Artifact Registry for Docker images
gcloud artifacts repositories create cosmos-training \
    --repository-format=docker \
    --location=us-central1 \
    --description="ABEE Cosmos training containers"

# Grant IAM permissions for Vertex AI service account
PROJECT_NUMBER=$(gcloud projects describe YOUR_PROJECT_ID --format="value(projectNumber)")
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-aiplatform.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"
```

### Step 2: Install Python SDK

```bash
pip install google-cloud-aiplatform google-cloud-storage kfp google-cloud-pipeline-components
```

### Step 3: Build and Push Container

```bash
cd /mnt/nv4/User_Data/development/cosmos-cookoff

# Configure Docker auth
gcloud auth configure-docker us-central1-docker.pkg.dev

docker build -t us-central1-docker.pkg.dev/YOUR_PROJECT/cosmos-training/cosmos-reason2-trainer:v1.0 \
    -f docker/Dockerfile.cosmos_train .

docker push us-central1-docker.pkg.dev/YOUR_PROJECT/cosmos-training/cosmos-reason2-trainer:v1.0
```

### Step 4: Upload SFT Data

```bash
# From ABEE orchestrator output
python upload_sft_data.py \
    --input /mnt/nv4/User_Data/development/cosmos-cookoff/data/sft_samples.jsonl \
    --bucket abee-sft-data \
    --temporal-split
```

### Step 5: Submit Training Job

```bash
python submit_training_job.py \
    --project YOUR_PROJECT_ID \
    --region us-central1 \
    --use-spot \
    --lora-rank 16
```

---

## Known Limitations and Gotchas

### Vertex AI Limitations

1. **No managed SFT for Cosmos-Reason2.** The managed `sft.train()` API is Gemini-only (and a small set of Llama/Mistral models). You must use Custom Training Jobs.

2. **Spot VM preemption with no warning on long jobs.** Use `save_steps=50` or `save_steps=100` checkpointing. Vertex AI does NOT auto-resume from checkpoint — you must handle restart logic in your training script.

3. **Container image must be in Artifact Registry** (not Docker Hub). Vertex AI will reject training jobs that reference public Docker Hub images directly for security reasons.

4. **A100 quota.** By default, new GCP projects have zero A100 quota. File a quota increase request in the GCP Console before starting: IAM & Admin > Quotas > search "NVIDIA_TESLA_A100".

5. **Region availability.** Not all GPU types are available in all regions. `us-central1` has the widest GPU selection. `us-east4` and `europe-west4` are secondary options.

6. **Multimodal data in managed datasets.** Vertex AI Managed Datasets do NOT natively support storing video/image binary data inline. Use GCS URIs in your JSONL and load media files from GCS inside your training container.

### Cosmos-Reason2 Specific

1. **HuggingFace Hub access from Vertex AI containers.** Outbound internet is allowed from Vertex AI training containers, but it is slow for large downloads. Pre-cache model weights in GCS (see Step above).

2. **Flash Attention compatibility.** Flash Attention 2 (`flash-attn`) requires CUDA 11.6+ and specific PyTorch versions. Use the NVIDIA PyTorch base image `nvcr.io/nvidia/pytorch:24.12-py3` to avoid compatibility issues.

3. **Video frame extraction.** If using video-level SFT samples, extract frames before training (not during) to avoid bottlenecking on I/O during training. Use `ffmpeg` in a preprocessing step.

4. **cosmos-rl vs. TRL.** The official `cosmos-rl` framework (TOML-based) is the tested path for Cosmos-Reason2 SFT. TRL-based training with the Qwen3-VL processor should also work (since Cosmos-Reason2 follows Qwen3-VL architecture) but is less officially supported.

### NIM Integration

1. **NIM containers are not officially supported on Vertex AI for arbitrary models.** The only documented integration is NeMo Retriever for embeddings. Running inference NIM containers requires manual configuration of the serving endpoint.

2. **NGC API key required.** NIM containers check for `NVIDIA_API_KEY` at startup. Store this as a GCP Secret and inject it as an environment variable.

---

## Sources and References

### Google Cloud Official Documentation
- [Vertex AI Custom Training Overview](https://cloud.google.com/vertex-ai/docs/training/overview)
- [Vertex AI PyTorch Integration](https://docs.cloud.google.com/vertex-ai/docs/start/pytorch)
- [Create a Serverless Training Job](https://docs.cloud.google.com/vertex-ai/docs/training/create-custom-job)
- [Custom Containers Overview](https://docs.cloud.google.com/vertex-ai/docs/training/containers-overview)
- [vLLM Serving for Multimodal Models on Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/vllm/use-vllm)
- [Tune Open Models on Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/open-model-tuning)
- [Vertex AI Pipelines Introduction](https://docs.cloud.google.com/vertex-ai/docs/pipelines/introduction)
- [Vertex AI Vizier Overview](https://docs.cloud.google.com/vertex-ai/docs/vizier/overview)
- [Use Spot VMs with Training](https://docs.cloud.google.com/vertex-ai/docs/training/use-spot-vms)
- [Schedule Training Jobs with DWS](https://docs.cloud.google.com/vertex-ai/docs/training/schedule-jobs-dws)
- [Vertex AI Pricing](https://cloud.google.com/vertex-ai/pricing)
- [Google Cloud GPU Pricing](https://cloud.google.com/compute/gpus-pricing)

### NVIDIA Cosmos-Reason2 Resources
- [Cosmos-Reason2 HuggingFace Model Card](https://huggingface.co/nvidia/Cosmos-Reason2-8B)
- [cosmos-reason2 GitHub Repository](https://github.com/nvidia-cosmos/cosmos-reason2)
- [Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/index.html)
- [Post-train Cosmos Reason 2 for AV Video Captioning](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason2/video_caption_vqa/post_training.html)
- [NVIDIA Cosmos Reason 2 Blog Post (HuggingFace)](https://huggingface.co/blog/nvidia/nvidia-cosmos-reason-2-brings-advanced-reasoning)
- [cosmos-reason2-8b on NVIDIA NIM](https://build.nvidia.com/nvidia/cosmos-reason2-8b)
- [Cosmos Documentation](https://docs.nvidia.com/cosmos/latest/reason2/index.html)

### NIM Integration
- [NeMo Retriever NIM on Vertex AI](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/deploying-vertex-ai.html)
- [NIM for Developers](https://developer.nvidia.com/nim)
- [NIM PEFT Fine-Tuning](https://docs.nvidia.com/nim/large-language-models/latest/peft.html)

### Fine-Tuning Guides and Community Resources
- [Fine-tune Mistral 7B with SFT on Vertex AI (HuggingFace Docs)](https://huggingface.co/docs/google-cloud/en/examples/vertex-ai-notebooks-trl-full-sft-fine-tuning-on-vertex-ai)
- [Vertex AI TRL Fine-Tuning Gemma (GoogleCloudPlatform)](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/open-models/fine-tuning/vertex_ai_trl_fine_tuning_gemma.ipynb)
- [Qwen2-VL Fine-Tuning Guide](https://github.com/2U1/Qwen2-VL-Finetune)
- [Finetuning Cosmos-Reason2 (Datature Blog)](https://datature.io/blog/finetuning-your-own-cosmos-reason2-model)
- [Vertex AI Pipelines End-to-End Samples](https://github.com/GoogleCloudPlatform/vertex-pipelines-end-to-end-samples)

### Pricing Research Sources
- [Vertex AI Pricing Review 2026 (Finout)](https://www.finout.io/blog/top-16-vertex-services-in-2026)
- [H100 Rental Prices Cloud Comparison 2026 (IntuitionLabs)](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
- [Cloud GPU Pricing Comparison 2026 (GPUnex)](https://www.gpunex.com/gpunex.com/blog/cloud-gpu-pricing-comparison-2026/)
- [Reserved GPU Cluster Training on Vertex AI](https://oneuptime.com/blog/post/2026-02-17-how-to-use-vertex-ai-training-with-reserved-gpu-clusters-for-predictable-workloads/view)

---

*Document generated by Claude Code research synthesis — verify pricing against official Google Cloud pricing pages before submitting jobs.*
