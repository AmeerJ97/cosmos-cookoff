# NVIDIA Cloud Training Platforms for VLM Fine-Tuning
## Research Document for CLASP / Cosmos Cookoff

**Research Date:** March 5, 2026
**Scope:** NVIDIA Brev, DGX Cloud, NeMo Framework, NIM deployment, GPU cloud comparison, Cosmos Cookoff specifics

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [NVIDIA Brev](#1-nvidia-brev)
3. [NVIDIA DGX Cloud](#2-nvidia-dgx-cloud)
4. [NVIDIA NeMo Framework](#3-nvidia-nemo-framework)
5. [Cosmos-Reason2-8B Fine-Tuning](#4-cosmos-reason2-8b-fine-tuning)
6. [NVIDIA NIM + Deployment Pipeline](#5-nvidia-nim--deployment-pipeline)
7. [Alternative GPU Cloud Providers](#6-alternative-gpu-cloud-providers)
8. [GPU Pricing Comparison Table](#7-gpu-pricing-comparison-table)
9. [Cosmos Cookoff Infrastructure](#8-cosmos-cookoff-infrastructure)
10. [Recommendations for CLASP](#9-recommendations-for-clasp)
11. [Sources and References](#10-sources-and-references)
12. [Methodology](#11-methodology)

---

## Executive Summary

NVIDIA has built a vertically integrated ecosystem spanning cloud GPU access (Brev, DGX Cloud Lepton), training frameworks (NeMo AutoModel, Cosmos-RL, TAO Toolkit), and production deployment (NIM microservices). For the CLASP project targeting Cosmos-Reason2-8B SFT fine-tuning:

- **Brev** has been acquired by NVIDIA (mid-2024) and is now the primary self-serve GPU access portal, aggregating multiple cloud providers. Third-place Cosmos Cookoff prize is $500 in Brev credits.
- **DGX Cloud Lepton** is NVIDIA's new unified multi-cloud orchestration platform (announced 2025), replacing the CSP-direct DGX Cloud approach for developers.
- **NeMo AutoModel** supports VLM SFT/LoRA with FSDP2 distributed training and is the recommended path for architectures like Qwen3-VL (which underlies Cosmos-Reason2).
- **Cosmos-RL** (in the cosmos-reason2 repo) is the native fine-tuning framework for Cosmos-Reason2, requiring minimum 4x 80GB GPUs for the 8B model.
- **NIM for VLMs** officially supports Cosmos-Reason2 (both 2B and 8B) for production deployment post fine-tuning.
- **Nebius AI Cloud** is the Cosmos Cookoff's compute sponsor with the most competitive pricing for H100s at $2.95/hr (HGX) and L40S from $1.55/hr.
- For the CLASP project's local RTX 4060 Ti 16GB, full 8B SFT is not feasible locally; cloud inference/training is required for any SFT runs.

---

## 1. NVIDIA Brev

### 1.1 What Is Brev?

NVIDIA Brev is an AI and ML platform that provides developers with streamlined access to GPU instances across multiple cloud providers, automatic environment setup, and flexible deployment options. It was originally founded as Brev.dev by a San Francisco-based startup (Nama Ventures) and **acquired by NVIDIA in mid-2024**. It is now fully integrated into the NVIDIA developer ecosystem at `developer.nvidia.com/brev`.

Brev is not a GPU cloud itself — it is an **aggregator and orchestration layer** that brokers GPU access from multiple underlying cloud providers, then wraps them with:
- Preconfigured CUDA/Python/Docker environments
- One-click "Launchables" from the NGC catalog
- IDE integration (VS Code, Cursor, Windsurf, code-server, tmux)
- NIM serverless deployment endpoints
- Shareable development environment links

### 1.2 Key Features

**GPU-backed Sandboxes**
Virtual machines preconfigured with Python, CUDA, Docker, and Jupyter Notebooks. Users connect via SSH (`brev shell <instance-name>`) or VS Code (`brev open <instance-name>`).

**Launchables**
Fully optimized, shareable environments that deploy with a single click. Users bundle:
1. Git repository or embedded container
2. Runtime configuration (setup scripts, Docker Compose)
3. Jupyter/networking configuration
4. GPU hardware selection (filterable by VRAM, cloud provider, price)
5. Named, shareable link with usage metrics

The NGC catalog integration enables 1-click deploy of NVIDIA AI software packages that previously required hours of manual setup, reduced to 2-3 minutes.

**NIM Serverless**
Brev provides auto-scaling serverless endpoints for NVIDIA NIM microservices, enabling production VLM deployment without managing infrastructure.

**NVIDIA Blueprint Integration**
Access to curated NVIDIA Blueprints at `build.nvidia.com`, including the VLM fine-tuning playbook.

### 1.3 GPU Availability

Brev's documentation intentionally does not publish a fixed GPU catalog — it dynamically aggregates from multiple cloud providers. The console allows filtering by:
- VRAM amount
- Cloud provider preference
- Compute attributes (SXM vs PCIe, NVLink, etc.)
- Price range

In practice, Brev surfaces GPUs from providers including AWS, GCP, CoreWeave, and others that are part of NVIDIA's partner ecosystem.

### 1.4 Pricing

Brev charges GPU hours at rates determined by the underlying cloud provider; it does not significantly mark up the compute cost. Specific pricing is not published in documentation and must be viewed in the Brev console at time of instance creation. The platform passes through underlying provider rates with its value-add in environment automation.

**Cosmos Cookoff relevance:** Third place prize is **$500 in Brev credits**, which at ~$1.99-2.99/hr for an H100 translates to roughly 170-250 GPU-hours.

### 1.5 Fine-Tuning on Brev

An NGC catalog entry exists for fine-tuning workflows: `catalog.ngc.nvidia.com/orgs/nvidia/teams/csp_launcher/resources/finetune_mistral7b` demonstrates the pattern. For Cosmos-Reason2, the expected workflow is:
1. Launch a Brev instance with sufficient GPU VRAM (4x+ 80GB for 8B SFT)
2. Clone the `nvidia-cosmos/cosmos-reason2` repository
3. Use the Cosmos-RL framework for SFT/RL post-training
4. Export the fine-tuned weights in HuggingFace format
5. Optionally deploy via NIM from within the same instance

### 1.6 Status Assessment

Brev is **actively developed and maintained** as of early 2026. It is NVIDIA's primary self-serve developer GPU access layer and is central to the Cosmos Cookoff competition infrastructure.

---

## 2. NVIDIA DGX Cloud

### 2.1 DGX Cloud Ecosystem Overview

DGX Cloud is NVIDIA's brand for enterprise-grade GPU infrastructure. As of 2025-2026, it has bifurcated into distinct offerings:

**DGX Cloud Lepton** (new, developer-focused)
A unified AI platform and compute marketplace launched in 2025, connecting developers to tens of thousands of GPUs from a global network of cloud providers. Announced at CES 2026 with major software updates. Designed as the developer-accessible tier.

**DGX Cloud for CSPs** (enterprise)
Optimized GPU infrastructure on major cloud service providers (AWS, GCP, Azure, Oracle, etc.) with flexible contract terms. Targeted at enterprises building foundation models.

**NVIDIA Cloud Functions (NVCF)**
Serverless AI inference platform with auto-scaling and event-driven deployment across multi-cloud and on-premises environments.

### 2.2 DGX Cloud Lepton

The key product for developers. Its capabilities:
- **Dev pods**: Interactive environments with Jupyter notebooks, SSH, VS Code for prototyping
- **Batch jobs**: Large-scale multi-node training and data processing with performance monitoring
- **Inference endpoints**: Model deployment with automatic scaling and health monitoring
- **Multi-cloud GPU access**: Partners include AWS, CoreWeave, Crusoe, Firebird, Fluidstack, and others
- **NVIDIA Blackwell availability**: GB200 and B200 instances through partner integrations
- **NeMo + NIM integration**: Seamless with the NVIDIA software stack

DGX Cloud Lepton is in **early access** as of the research date. Access via: `developer.nvidia.com/dgx-cloud/get-lepton`

### 2.3 DGX Spark (for Local/Edge Development)

DGX Spark is the desktop-class AI supercomputer (announced CES 2025, updated CES 2026) running NVIDIA's GB10 chip with 128GB unified memory. It supports:
- VLM fine-tuning using InternVL3-8B and Qwen2.5-VL-7B workflows
- GRPO (Group Relative Policy Optimization) training with configurable LoRA rank/alpha
- Both image and video VLM fine-tuning
- NeMo fine-tuning integration

**Important:** DGX Spark is hardware, not cloud. It's relevant if you have access to one (first-place Cookoff prize is a DGX Spark).

### 2.4 VLM Fine-Tuning on DGX Cloud

Official VLM fine-tuning playbook at `build.nvidia.com/spark/vlm-finetuning` covers:
- **Image VLM**: Qwen2.5-VL-7B for wildfire detection via GRPO
- **Video VLM**: InternVL3-8B for dangerous driving detection
- Both use Docker-based environment setup
- Training configuration: LoRA rank 8-64, alpha 8-64, batch 1-16, LR 1e-6 to 1e-2, AdamW/Adafactor optimizers

The playbook does not directly address Cosmos-Reason2, though the underlying Qwen3-VL architecture makes the approach transferable.

### 2.5 Pricing

DGX Cloud pricing is **not publicly listed**. Enterprise contracts are negotiated directly. For the DGX Cloud Lepton early-access program, trial access is available via request at `nvidia.com/en-us/data-center/dgx-cloud/trial/`.

---

## 3. NVIDIA NeMo Framework

### 3.1 Architecture Overview (2025-2026)

NeMo has been significantly restructured. The original monolithic NeMo 2.0 repository now focuses on **speech AI** (ASR/TTS). Other components have been modularized into:

| Component | Purpose | Repo |
|-----------|---------|------|
| **NeMo AutoModel** | PyTorch DTensor-native training for LLMs/VLMs | `NVIDIA-NeMo/Automodel` |
| **NeMo RL** | Reinforcement learning toolkit (GRPO, PPO, SFT) | `NVIDIA-NeMo/RL` |
| **NeMo Microservices** | Cloud-native managed fine-tuning service | docs.nvidia.com/nemo/microservices |
| **Megatron-Bridge** | Training library for Megatron-based models with HF conversion | `NVIDIA-NeMo/Megatron-Bridge` |
| **NeMo Curator** | Data curation for foundation model training | — |

**For VLM fine-tuning on standard HuggingFace-format models (like Cosmos-Reason2-8B based on Qwen3-VL), NeMo AutoModel is the recommended path.**

### 3.2 NeMo AutoModel for VLMs

NeMo AutoModel is a PyTorch DTensor-native training library designed to streamline and scale training and fine-tuning for LLMs and VLMs, enabling experiments from single-GPU to massive multi-GPU multi-node deployments.

**Supported VLM architectures (confirmed tested):**
- Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, Qwen3-Omni
- Kimi-VL-A3B, Kimi-K25-VL
- Gemma 3-4B/27B, Gemma 3n
- InternVL3.5-4B, Ministral3, Phi-4-multimodal
- Llava variants, SmolVLM, Llama-4

Since **Cosmos-Reason2-8B is based on Qwen3-VL-8B-Instruct architecture**, it should be fully supported under NeMo AutoModel's VLM training path.

**Required container version:** NeMo container 25.11.00 or later for VLM support.

### 3.3 SFT Configuration

Full YAML configuration structure for NeMo AutoModel SFT:

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained
  pretrained_model_name_or_path: nvidia/Cosmos-Reason2-8B
  is_meta_device: false   # set true for very large models > single GPU

trainer:
  strategy: fsdp2          # Fully Sharded Data Parallel 2
  dp_size: 4               # data parallel degree (= number of GPUs for SFT)
  tp_size: 1               # tensor parallelism (increase for >8B models)
  cp_size: 1               # context parallelism

training:
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-5
  optimizer:
    _target_: torch.optim.Adam
    betas: [0.9, 0.999]
  max_epochs: 3
  checkpoint_every_n_steps: 10
  val_check_interval: 10

dataset:
  _target_: nemo_automodel.datasets.VLMDataset
  path_or_dataset: <your_dataset_path>
  split: train
  dataloader:
    collate_fn: <model_specific_collate>
```

**Key parameters for the 8B model:**
- Use `is_meta_device: true` if model exceeds single GPU VRAM
- Set `tp_size: 2` or higher if using fewer GPUs with less VRAM
- Minimum 4x 80GB GPUs for full precision SFT (aligns with Cosmos-RL recommendation)

### 3.4 PEFT / LoRA Configuration

```yaml
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules: "*.proj"   # or specific: q_proj, v_proj, k_proj, o_proj
  dim: 16                    # LoRA rank r (8-256, higher = more capacity)
  alpha: 32                  # LoRA alpha (usually 2x rank)
  use_triton: True           # enable Triton kernels for performance
```

**QLoRA** (4-bit quantization + LoRA): Reduces GPU memory by ~75% through NF4 quantization with BFloat16 precision. The Datature tutorial confirms this works for Cosmos-Reason2-8B on 4x A10G (24GB each).

**PEFT checkpoint output:** Lightweight adapter files (`adapter_model.safetensors`, `adapter_config.json`) requiring the base model at inference time.

**SFT checkpoint output:** Full consolidated model in safetensors format with config and tokenizer.

### 3.5 Distributed Training Setup

NeMo AutoModel uses FSDP2 (Fully Sharded Data Parallel 2) as its primary distributed strategy:
- **NCCL backend** with 1-minute timeout (configure higher for large clusters)
- Data parallelism: set `dp_size` = number of GPUs for simple SFT
- Tensor parallelism: increase `tp_size` for memory-constrained scenarios
- Context parallelism: `cp_size` for very long sequence training

For multi-node training, ensure InfiniBand or NVLink interconnect and configure NCCL environment variables accordingly.

### 3.6 NeMo Microservices (Managed Fine-Tuning)

A separate managed service for production fine-tuning workflows. Key capabilities:
- REST API-driven fine-tuning job submission
- SFT and LoRA supported
- Integrated with ArgoCD/Argo Workflow on DGX Cloud
- Automated evaluation pipeline (NeMo Evaluator, regression tests, LLM-as-a-judge)
- Configuration via JSON API (not YAML)

Example SFT job config:
```json
{
  "base_model": "nvidia/Cosmos-Reason2-8B",
  "training_type": "sft",
  "finetuning_type": "lora",
  "precision": "bf16-mixed",
  "num_gpus": 8,
  "batch_size": 4,
  "epochs": 3,
  "learning_rate": 1e-5,
  "adapter_dim": 16,
  "adapter_dropout": 0.1
}
```

---

## 4. Cosmos-Reason2-8B Fine-Tuning

### 4.1 Model Architecture

Cosmos-Reason2-8B is based on **Qwen3-VL-8B-Instruct** architecture (released December 19, 2025). It is post-trained with:
- Supervised Fine-Tuning (SFT) on physical common sense and embodied reasoning data
- Reinforcement Learning for further alignment

Key capabilities relevant to CLASP:
- Spatio-temporal reasoning (space, time, physics understanding)
- Object detection with 2D/3D point localization and bounding-box coordinates
- Long-context processing: up to 256K input tokens
- Chain-of-thought reasoning for world dynamics
- Video and image inputs

**HuggingFace:** `nvidia/Cosmos-Reason2-8B`
**Inference VRAM requirement:** 32GB (BF16), or quantized for less

### 4.2 Official Fine-Tuning Frameworks

NVIDIA provides three official pathways for Cosmos-Reason2 post-training:

#### Path A: Cosmos-RL (Native, Recommended)
Located in `nvidia-cosmos/cosmos-reason2/examples/cosmos_rl/`

- **Framework:** Async post-training framework for SFT and RLHF, built for performance, scalability, and fault tolerance
- **Hardware minimum:** 4x GPUs with 80GB memory each (for 8B model)
- **Scalable to:** 8x GPUs by adjusting `dp_shard_size`
- **Data format:** LLaVA dataset format (JSON annotation file + media directory)
- **Config format:** TOML files (e.g., `llava_sft.toml`)

Annotation format:
```json
[
  {
    "id": "scene_001",
    "image": "path/to/image.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nIs it safe to hand off to the robot now?"},
      {"from": "gpt", "value": "<think>Analyzing the scene... human hand at position...</think>\nACT"}
    ]
  }
]
```

Training command:
```bash
uv run cosmos-rl --config configs/llava_sft.toml \
    --log-dir outputs/llava_sft \
    scripts/llava_sft.py
```

Dependencies:
- Redis (for job coordination): `conda install -c conda-forge redis-server`
- HuggingFace Transformers >= 4.57.0
- CUDA 12.8, NVIDIA Driver 570+

Storage note: Training downloads ~200GB of model and dataset files. Set `HF_HOME` and `COSMOS_CACHE` environment variables to a disk with sufficient space.

#### Path B: NVIDIA TAO Toolkit (Enterprise)
Located at `docs.nvidia.com/tao/tao-toolkit/latest/text/vlm_finetuning/cosmos_rl.html`

- **Hardware minimum:** 8x A100 GPUs with 80GB each (stricter than Cosmos-RL path)
- **Data format:** Same LLaVA format
- **Config format:** YAML via FTMS client
- **Quantization support:** FP8_DYNAMIC, W8A8, W8A16, W4A16
- **LoRA rank:** 8-256, targeting q_proj, v_proj, attention, MLP layers
- **FP8 training:** Supported on H100/H200 GPUs only
- **Gradient checkpointing:** Reduces memory ~40% at cost of compute

TAO Toolkit is primarily enterprise-focused; Cosmos-RL is more accessible for hackathon/research use.

#### Path C: NeMo AutoModel (HuggingFace-compatible)
As described in Section 3. This is the most flexible path and integrates best with the broader NeMo ecosystem for subsequent deployment via NIM.

### 4.3 Practical Fine-Tuning Results (Datature Tutorial)

The Datature team published a practical Cosmos-Reason2-8B fine-tuning guide using:
- **Hardware:** 4x NVIDIA A10G GPUs (24 GiB each)
- **Method:** LoRA with QLoRA (NF4 quantization + BFloat16)
- **Learning rate:** 5e-5
- **Batch size:** 4
- **Epochs:** 100
- **Train/validation split:** 80/20

**Results on warehouse spatial reasoning task (100 training scenes):**
- BERTScore F1: ~0.91
- ROUGE score: ~0.55
- METEOR score: ~0.60

**Key finding:** 4x A10G (96GB total VRAM) is sufficient for 8B model fine-tuning with QLoRA. This maps to ~$0.86 x 4 = $3.44/hr on RunPod (L40S is comparable VRAM).

**Data format used:**
```
Question: Spatial query with [Region N]<box>x1,y1,x2,y2</box> tokens
Answer: Response referencing regions in same format
(coordinates normalized 0-1024)
```

**Post-training performance gains (from NVIDIA technical blog):**
- SFT alone: >10% improvement over base model
- SFT + RL: additional ~5% gain
- Combined: 65.7 average score on robotics/AV benchmarks

### 4.4 CLASP-Specific Considerations

For fine-tuning on CLASP's SFT dataset (curated human-robot handoff prediction data):

1. **Local RTX 4060 Ti 16GB is insufficient** for 8B SFT. Options:
   - QLoRA on 4x A10G-equivalent or better in cloud
   - Use 2B model locally for experimentation
   - Run SFT in cloud, inference locally (model fits quantized)

2. **Data format:** Convert CLASP's POMDP decision logs to LLaVA conversation format, with video frames as images and THINK/ACT decisions as assistant responses

3. **SFT target:** Train on correct-ACT examples from high-performing agent episodes (high Life-Points, correct consensus with oracle)

4. **Evaluation:** Use CLASP's own safe-handoff prediction task, not generic benchmarks

---

## 5. NVIDIA NIM + Deployment Pipeline

### 5.1 NIM for VLMs

NVIDIA NIM for Vision Language Models is a separate product from NIM for LLMs, with dedicated containerization and API surface.

**Cosmos-Reason2 NIM support matrix (as of NIM VLM 1.6.0):**

| Model | Min VRAM | Notes |
|-------|---------|-------|
| Cosmos-Reason2-8B | >56GB (BF16) | Speculative decoding supported |
| Cosmos-Reason2-2B | >36GB (BF16) | Speculative decoding supported |
| Cosmos-Reason1-7B | 24GB (BF16) or 16GB (FP8) | — |

**API compatibility:** OpenAI-compatible API with custom NVIDIA extensions for additional multimodal functionality. Integrates with LangChain and LlamaIndex.

### 5.2 Complete Train-to-Deploy Pipeline

```
[CLASP SFT Data Collection]
        |
        v
[Cosmos-RL / NeMo AutoModel SFT]
  - 4x-8x 80GB GPUs (cloud)
  - LLaVA format input
  - HuggingFace checkpoint output
        |
        v
[Export / Convert Weights]
  - Safetensors format
  - Include config.json, tokenizer
        |
        v
[Deploy via NVIDIA NIM for VLMs]
  - Mount model dir to NIM container
  - Set NIM_FT_MODEL env variable
  - NIM auto-builds TensorRT-LLM engine
  - Serves OpenAI-compatible API
        |
        v
[Local CLASP Inference]
  - Query NIM endpoint from orchestrator
  - 2.6x throughput vs off-the-shelf H100
```

### 5.3 Fine-Tuned Model Deployment via NIM (LLM path, for reference)

For fine-tuned models deployed via NIM for LLMs (the documented path; VLM path similar):
- **Minimum VRAM:** 80GB GPU
- **Environment variables:**
  - `NIM_FT_MODEL=/path/to/fine-tuned-weights`
  - `NIM_SERVED_MODEL_NAME=clasp-cosmos-reason2-sft`
  - `NIM_USE_TRTLLM_LEGACY_BACKEND=1` (required for fine-tuned models — default backend does not support them)
  - `NIM_CUSTOM_MODEL_NAME=clasp-sft-v1` (for caching built engines)

**Critical note:** The **default PyTorch TRT-LLM backend does not support fine-tuned models.** The legacy backend must be explicitly enabled.

### 5.4 LoRA Adapter Deployment via NIM

For LoRA-based fine-tuning (lighter weight):
- Deploy base model via NIM with LoRA profile: `vllm-fp16-tp2-lora` or `vllm-fp16-tp1-lora`
- Adapters trained with HuggingFace PEFT or NVIDIA NeMo are both supported
- Multi-LoRA support: multiple adapters can be loaded per container

### 5.5 NIM on Brev (Serverless)

Brev provides one-click serverless NIM deployment. For production use:
1. Train/fine-tune on Brev instance
2. Export weights to persistent storage
3. Deploy as serverless NIM endpoint from the same Brev console
4. Auto-scaling handles load

---

## 6. Alternative GPU Cloud Providers

### 6.1 Nebius AI Cloud

**Why it matters for CLASP:** Nebius is a **Cosmos Cookoff competition sponsor** and judge. Deployment guides for Nebius AI Cloud are included in the Cosmos Cookbook. The Cookoff rules note "Nebius credits are exhausted" (for the competition), implying credits were distributed but may have run out.

**GPU Pricing (standard on-demand):**

| GPU | Region | Standard $/hr | Preemptible $/hr |
|-----|--------|--------------|-----------------|
| H100 NVLink (80GB) | eu-north1 | $2.95 | $1.25 |
| H200 NVLink (141GB) | eu-north1, us-central1 | $3.50 | $1.45 |
| B200 NVLink | us-central1, me-west1 | $5.50 | $2.90 |
| B300 NVLink | uk-south1 | $6.10 | $3.40 |
| L40S PCIe (48GB) | eu-north1 | $1.35 | — |

**Explorer Tier for new customers:** First 1,000 H100 GPU-hours at $1.50/hr (registered after Oct 1, 2024). H100 standard after that: $2.00/hr (single GPU) or $2.95/hr (HGX config with NVLink).

**Strengths:** Competitive pricing, NVIDIA partner for Cosmos ecosystem, official Cookoff sponsor, strong EU/US coverage, preemptible instances for budget training.

**Storage costs:** Network SSD at $0.071/GiB per 730 hours (~$0.071/GiB/month).

**Multi-month reserved discounts:** Up to 35% off on-demand rates for large cluster commitments.

### 6.2 Lambda Labs

A research-and-enterprise-focused GPU cloud known for reliability and pre-configured AI frameworks.

**GPU Pricing (individual instances, per GPU/hr):**

| GPU | 8x Config | 1x Config |
|-----|-----------|-----------|
| B200 SXM6 | $5.74 | $6.08 |
| H100 SXM | $3.44 | $3.78 |
| H100 PCIe | — | $2.86 |
| GH200 (96GB) | — | $1.99 |
| A100 SXM (80GB) | $2.06 | — |
| A100 SXM (40GB) | $1.48 | $1.48 |
| A10 | — | $0.86 |
| A6000 | — | $0.92 |

**1-Click Clusters:** H100 at $2.76/hr, B200 at $4.62/hr (on-demand, 2-week to 12-month reservation).

**Strengths:** Enterprise reliability, no egress fees, managed infrastructure, pre-configured frameworks, integrated monitoring. Good for research labs that need guaranteed uptime.

**Weaknesses:** Less flexible than RunPod, higher prices than bare-metal alternatives.

### 6.3 RunPod

A flexible hybrid cloud with community (cheap, third-party hosts) and secure (RunPod-managed data centers) tiers.

**GPU Pricing (on-demand):**

| GPU | Community $/hr | Secure $/hr |
|-----|---------------|------------|
| H100 SXM (80GB) | ~$1.99 | $2.69 |
| A100 SXM (80GB) | $1.39 | ~$1.64 |
| L40S (48GB) | $0.79 | ~$0.86 |
| RTX 4090 (24GB) | ~$0.49 | ~$0.59 |

**Committed pricing discounts:**
- 6-month: ~6-7% off
- 1-year: ~9-10% off

**Community vs Secure:**
- Community: Cheaper, more variety, vetted third-party hosts, less SLA guarantee
- Secure: RunPod-managed data centers, SOC2 compliance, persistent NVMe volumes, +$0.10-0.40/hr premium

**Strengths:** Lowest price-per-GPU among major platforms, custom container support, dynamic scaling, no egress fees, great for researchers.

**Weaknesses:** Community cloud has reliability variability; not ideal for production serving.

### 6.4 CoreWeave

An enterprise-focused HPC cloud built specifically for AI/ML workloads.

**GPU Pricing:**

| GPU | $/hr (on-demand) |
|-----|----------------|
| H100 PCIe (80GB) | $4.76 |
| H100 8x HGX (per GPU) | $6.15 |
| A100 80GB NVLink | $2.21 |

**Committed discount:** Up to 60% off for reserved capacity.

**Key differentiators:**
- InfiniBand networking (up to 400Gbps between nodes) for distributed training
- NVIDIA GPUDirect RDMA support
- Slurm on Kubernetes (SUNK) for >32K GPU clusters
- NVIDIA Quantum-X800 InfiniBand for GB200 NVL72 clusters
- Kubernetes-native deployment
- Topology-aware scheduling for optimal distributed training performance

**Strengths:** Best infrastructure for very large distributed training (>16 GPUs), enterprise SLAs, cutting-edge GPU access including Blackwell/GB200.

**Weaknesses:** Most expensive on-demand rates, complex setup, overkill for small fine-tuning jobs, enterprise-focused pricing.

### 6.5 Vast.ai

A marketplace-style platform connecting GPU providers with users.

**Pricing:** Current market floor at $1.49-1.87/hr for H100 (promotional/spot pricing). A100 80GB from ~$0.50/hr.

**Strengths:** Lowest possible prices.
**Weaknesses:** Variable reliability, no SLA, requires careful provider selection.

---

## 7. GPU Pricing Comparison Table

Pricing as of Q1 2026. All on-demand unless noted.

### H100 SXM/PCIe 80GB

| Provider | Type | $/GPU/hr | Notes |
|----------|------|---------|-------|
| Vast.ai | Marketplace | $1.49-1.87 | Spot/variable, no SLA |
| Nebius (Explorer Tier) | H100 NVLink | $1.50 | First 1,000 hrs, new customers |
| RunPod Community | H100 PCIe | $1.99 | Third-party hosted |
| Nebius | H100 NVLink | $2.00 | Single GPU on-demand |
| HPC-AI | H100 SXM | $1.99 | Boutique provider |
| TensorDock | H100 SXM5 | $2.25 | Dedicated |
| Lambda | H100 SXM (8x) | $2.76 | Cluster pricing |
| Lambda | H100 PCIe | $2.86 | Single |
| Nebius | H100 HGX NVLink | $2.95 | NVLink config |
| Lambda | H100 SXM | $3.78 | Single instance |
| Google Cloud | H100 (A3-highgpu) | $3.00 | Major cloud |
| AWS | H100 P5 | $3.90 | Post-44% cut (Jun 2025) |
| CoreWeave | H100 PCIe | $4.76 | Enterprise |
| Paperspace | H100 | $5.95 | Dedicated single |
| Azure | H100 | $6.98 | Most expensive |

### A100 80GB

| Provider | $/GPU/hr | Notes |
|----------|---------|-------|
| Vast.ai | ~$0.50 | Marketplace |
| RunPod Community | $1.39 | Community cloud |
| Lambda A100 SXM | $2.06 | 8x config |
| CoreWeave A100 NVLink | $2.21 | Enterprise |
| AWS A100 | ~$4.10 | P4de instances |

### L40S 48GB (good for VLM fine-tuning with QLoRA)

| Provider | $/GPU/hr | Notes |
|----------|---------|-------|
| RunPod Community | $0.79 | Community cloud |
| RunPod Secure | ~$0.86 | SOC2-compliant |
| Nebius L40S (Intel) | $1.35 | PCIe, eu-north1 |
| Nebius L40S (AMD) | $1.55+ | PCIe, eu-north1 |
| Modal | ~$1.00 | Serverless per-second |

### Cost Estimate: Cosmos-Reason2-8B SFT Run

Based on Datature tutorial (100 scenes, 100 epochs, 4x A10G/L40S-class GPUs):
- **RunPod 4x L40S:** 4 x $0.79/hr x ~8 hrs = **~$25**
- **Nebius 4x L40S:** 4 x $1.35/hr x ~8 hrs = **~$43**
- **Lambda 4x A100:** 4 x $1.48/hr x ~8 hrs = **~$47**
- **Nebius H100 NVLink (faster):** $2.95/hr x ~3 hrs = **~$9** (single H100, LoRA)

For the full CLASP SFT dataset (potentially hundreds to thousands of episodes), budget $50-500 depending on dataset size and number of training runs.

---

## 8. Cosmos Cookoff Infrastructure

### 8.1 Competition Overview

- **Timeline:** January 29 – March 5, 2026 (submissions close March 5, 5 PM PT)
- **Winners announced:** Week of March 30, 2026
- **Format:** Virtual, 4-week challenge

### 8.2 Prizes

| Place | Prize |
|-------|-------|
| 1st | $3,000 cash + NVIDIA DGX Spark (desktop AI supercomputer, 128GB unified memory) |
| 2nd | $2,000 cash + NVIDIA GeForce RTX 5090 GPU |
| 3rd | $500 Brev credits |

### 8.3 Compute Infrastructure Provided

**Nebius AI Cloud:** Competition sponsor. Deployment guides for Nebius are included in the official Cosmos Cookbook. The competition rules note that **"Brev and Nebius credits are exhausted"** — indicating credits were distributed (likely at registration) but are now used up for most participants. Any remaining training must use your own compute budget.

**Brev:** Provides the Launchables infrastructure for quick environment setup. Third-place prize is $500 in Brev credits.

### 8.4 Submission Requirements

1. **Text description:** Project features and functionality
2. **Demo video:** Under 3 minutes
3. **Public code repository URL:** GitHub/GitLab with full source
4. **README:** Deployment instructions

### 8.5 Judging Criteria

| Criterion | Description |
|-----------|-------------|
| Quality of Ideas | Compelling Cosmos Reason application for robotics/autonomous systems/video analytics |
| Technical Implementation | Code quality, reproducibility, clear documentation |
| Design | Intuitive UX, thoughtful project design |
| Impact | Real-world problem-solving, advancement of physical AI |

**Judges from:** Datature, Hugging Face, Nebius, Nexar, NVIDIA

### 8.6 Scope Restriction

The competition **explicitly excludes** generative media, art, gaming, and content creation. Projects must address physical AI reasoning: robotics, autonomous systems, video analytics. CLASP (safe human-robot handoff prediction) fits squarely within scope.

### 8.7 Available Resources

- Office hours with NVIDIA experts
- Discord community support
- Cosmos Cookbook recipes: `nvidia-cosmos.github.io/cosmos-cookbook/`
- Educational livestreams and AMAs throughout February

---

## 9. Recommendations for CLASP

### 9.1 For SFT Fine-Tuning Runs

**Recommended path:** RunPod Community Cloud or Nebius AI Cloud

- **Budget SFT run (~$25-50):** RunPod 4x L40S at $0.79/GPU/hr with QLoRA
  - Sufficient for Cosmos-Reason2-8B with NF4 quantization (matches Datature tutorial setup)
  - Use Cosmos-RL framework (`llava_sft.toml` with `dp_shard_size = 4`)

- **Quality SFT run (~$10-30):** Nebius single H100 NVLink at $2.95/hr
  - Faster training, no quantization needed
  - Nebius is the official Cookoff sponsor — familiarity useful for deployment

- **Avoid CoreWeave** for small SFT runs: overkill pricing, complex setup

### 9.2 For the CLASP SFT Dataset Format

Convert CLASP episode logs to LLaVA JSON format:
```json
{
  "id": "episode_001_frame_008",
  "image": "frames/ep001_frame008.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nYou are analyzing a human-robot handoff scene. Is it safe to release the object now?"
    },
    {
      "from": "gpt",
      "value": "<think>Examining human hand position, joint angles, and trajectory...</think>\nTHINK"
    }
  ]
}
```
Include only high-confidence, oracle-validated episodes where the agent made the correct decision.

### 9.3 For Post-SFT Deployment

- **Local inference (RTX 4060 Ti 16GB):** Use quantized weights (GPTQ/AWQ 4-bit) via vLLM or llmcompressor (see `cosmos-reason2/docs/llmcompressor.md`). The 8B model quantized to 4-bit fits in ~8GB VRAM.
- **Cloud inference during competition:** Deploy via NIM on Nebius (sponsor alignment) or RunPod Secure for reliability
- **NIM deployment:** After fine-tuning, export HF-format weights and deploy with `NIM_USE_TRTLLM_LEGACY_BACKEND=1`

### 9.4 Using Brev for Competition

Even if Brev credits are exhausted, Brev's Launchables are useful for:
- Quickly spinning up consistent environments
- NGC catalog one-click deploys for NeMo containers
- Sharing reproducible environments in the README submission

### 9.5 Framework Choice Decision Tree

```
Have 4+ 80GB GPUs available?
  YES -> Use Cosmos-RL directly (most native, TOML config)
  NO  -> Use QLoRA path:
           Cosmos-Reason2-8B (Qwen3-VL) compatible with:
           - NeMo AutoModel VLM fine-tuning (YAML config, FSDP2)
           - TRL (HuggingFace training, notebooks)
           Both work with 4x 24GB GPUs via NF4 quantization
```

---

## 10. Sources and References

### NVIDIA Official Documentation
- [NVIDIA Brev Developer Page](https://developer.nvidia.com/brev)
- [Brev Documentation - About](https://docs.nvidia.com/brev/latest/about-brev.html)
- [Brev Documentation - Launchables](https://docs.nvidia.com/brev/latest/launchables.html)
- [DGX Cloud for Developers](https://developer.nvidia.com/dgx-cloud)
- [DGX Cloud Lepton Announcement](https://developer.nvidia.com/blog/introducing-nvidia-dgx-cloud-lepton-a-unified-ai-platform-built-for-developers/)
- [VLM Fine-tuning Playbook (DGX Spark)](https://build.nvidia.com/spark/vlm-finetuning)
- [NeMo Framework Overview](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html)
- [NeMo AutoModel VLM Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/automodel/vlm.html)
- [NeMo AutoModel SFT/PEFT Guide](https://docs.nvidia.com/nemo/automodel/latest/guides/llm/finetune.html)
- [NIM for VLMs - Introduction](https://docs.nvidia.com/nim/vision-language-models/latest/introduction.html)
- [NIM for VLMs - Support Matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html)
- [NIM Fine-Tuned Model Support](https://docs.nvidia.com/nim/large-language-models/latest/ft-support.html)
- [Deploying Fine-Tuned Models with NIM (blog)](https://developer.nvidia.com/blog/deploying-fine-tuned-ai-models-with-nvidia-nim/)
- [TAO Toolkit - Cosmos-RL Fine-tuning](https://docs.nvidia.com/tao/tao-toolkit/latest/text/vlm_finetuning/cosmos_rl.html)
- [Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/index.html)
- [Cosmos-Reason2 Documentation](https://docs.nvidia.com/cosmos/latest/reason2/index.html)
- [Post-training Cosmos Reason (Technical Blog)](https://developer.nvidia.com/blog/maximize-robotics-performance-by-post-training-nvidia-cosmos-reason/)
- [Cosmos-Reason2 Post-training Guide](https://docs.nvidia.com/cosmos/2.0.0/reason1/post-training_guide.html)
- [NGC Fine-tune Mistral 7B via Brev](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/csp_launcher/resources/finetune_mistral7b)
- [Deploy GPU-Optimized AI via Brev + NGC (blog)](https://developer.nvidia.com/blog/deploy-gpu-optimized-ai-software-with-one-click-using-brev-dev-and-nvidia-ngc-catalog/)

### GitHub Repositories
- [nvidia-cosmos/cosmos-reason2](https://github.com/nvidia-cosmos/cosmos-reason2)
- [nvidia-cosmos/cosmos-reason2 - Cosmos-RL README](https://github.com/nvidia-cosmos/cosmos-reason2/blob/main/examples/cosmos_rl/README.md)
- [NVIDIA-NeMo/Automodel](https://github.com/NVIDIA-NeMo/Automodel)
- [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL)

### Competition Pages
- [NVIDIA Cosmos Cookoff (Luma)](https://luma.com/nvidia-cosmos-cookoff)
- [Cosmos Cookoff Launch Announcement (NVIDIA Forums)](https://forums.developer.nvidia.com/t/the-nvidia-cosmos-cookoff-is-here/359090)

### Cloud Provider Pricing
- [Nebius AI Cloud Pricing](https://nebius.com/prices)
- [Nebius Compute Pricing Documentation](https://docs.nebius.com/compute/resources/pricing)
- [Lambda Labs Pricing](https://lambda.ai/pricing)
- [RunPod Pricing](https://www.runpod.io/pricing)
- [RunPod L40S](https://www.runpod.io/gpu-models/l40s)
- [H100 Rental Prices Comparison 2026 (IntuitionLabs)](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
- [CoreWeave GPU Pricing](https://www.coreweave.com/pricing)

### Third-Party Analysis
- [Finetuning Cosmos-Reason2 Model (Datature)](https://datature.io/blog/finetuning-your-own-cosmos-reason2-model)
- [NVIDIA Brev Acquisition Coverage (AsiaTechDaily)](https://asiatechdaily.com/nvidia-expands-ai-portfolio-with-brev-dev-acquisition/)
- [Top Cloud GPU Providers 2026 (RunPod)](https://www.runpod.io/articles/guides/top-cloud-gpu-providers)
- [Nvidia Cosmos Cookoff prize coverage](https://roboticsandautomationnews.com/2026/02/08/nvidia-launches-robotics-hackathon-with-5000-top-prize-to-spur-physical-ai-development/)

---

## 11. Methodology

**Research approach:** Three-phase web research with parallel query execution.

**Phase 1 - Web searches (parallel):**
- NVIDIA Brev acquisition status and platform overview
- DGX Cloud VLM fine-tuning and NeMo integration
- NeMo framework Cosmos/VLM SFT LoRA 2025
- NIM training-to-deployment pipeline
- GPU cloud provider pricing comparison (Lambda, RunPod, CoreWeave)
- Cosmos Cookoff infrastructure and credits
- Nebius AI Cloud pricing and Cookoff sponsorship
- Cosmos-Reason2-8B fine-tuning documentation and hardware requirements
- NIM VLM multimodal fine-tuned model deployment

**Phase 2 - Documentation fetches (targeted):**
- `docs.nvidia.com/brev/latest/about-brev.html` and `/launchables.html`
- `docs.nvidia.com/nemo-framework/.../automodel/vlm.html`
- `docs.nvidia.com/nemo/automodel/latest/guides/llm/finetune.html`
- `developer.nvidia.com/blog/introducing-nvidia-dgx-cloud-lepton/`
- `build.nvidia.com/spark/vlm-finetuning`
- `developer.nvidia.com/dgx-cloud`
- `docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html`
- `docs.nvidia.com/nim/large-language-models/latest/ft-support.html`
- `developer.nvidia.com/blog/deploying-fine-tuned-ai-models-with-nvidia-nim/`
- `docs.nvidia.com/tao/tao-toolkit/.../cosmos_rl.html`
- `github.com/nvidia-cosmos/cosmos-reason2` (repo and cosmos_rl README)
- `nvidia-cosmos.github.io/cosmos-cookbook/`
- `docs.nvidia.com/cosmos/latest/reason2/index.html`
- `developer.nvidia.com/blog/maximize-robotics-performance-by-post-training-nvidia-cosmos-reason/`
- `nebius.com/prices` and `docs.nebius.com/compute/resources/pricing`
- `lambda.ai/pricing`
- `intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison`
- `luma.com/nvidia-cosmos-cookoff`
- `datature.io/blog/finetuning-your-own-cosmos-reason2-model`

**Sources consulted:** 30+ web sources including official NVIDIA documentation, competition pages, provider pricing pages, and independent analysis.

**Information currency:** All pricing and status information current as of March 2026 research date. GPU pricing is volatile; verify before making compute purchasing decisions.

**Limitations:**
- DGX Cloud enterprise pricing is not publicly available; contact NVIDIA sales for quotes
- Brev GPU catalog and exact pricing only visible in console at time of instance creation
- Cosmos Cookoff compute credit allocation details not fully documented publicly
- NIM fine-tuned VLM deployment specifics (as opposed to base model) are not fully documented for the VLM NIM variant (LLM NIM documentation used as reference)
