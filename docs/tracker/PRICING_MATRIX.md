# ABEE Cloud Pricing Matrix

Last updated: 2026-03-05

## GPU Cloud Providers

### H100 80GB (Recommended for GRPO)

| Provider | On-Demand | Spot/Preemptible | Min Config | Notes |
|----------|-----------|------------------|-----------|-------|
| **Vast.ai** | $1.49/hr | — | 1 GPU | Cheapest. Community GPUs, variable availability. |
| **Nebius (Explorer)** | $1.50/hr | — | 1 GPU | First 1K hrs only. Cosmos Cookoff sponsor. |
| **RunPod (Community)** | $1.99/hr | — | 1 GPU | Serverless option. Datature confirmed QLoRA works. |
| **Lambda Labs** | $2.76-3.78/hr | — | 1 GPU | Dev-friendly, SSH access. |
| **Nebius (Standard)** | $2.95/hr | $1.25/hr | 1 GPU | HGX cluster. Credits exhausted for Cookoff. |
| **NVIDIA Brev** | Pass-through | — | 1 GPU | Aggregator — prices vary by underlying provider. |
| **Google Cloud (GCE)** | ~$6.98/hr | ~$2.10/hr | 8-GPU min (a3) | Expensive but reliable. DWS FLEX_START recommended. |
| **CoreWeave** | $4.76-6.15/hr | — | 1 GPU | K8s-native, best for large distributed runs. |

### A100 80GB (Good for SFT)

| Provider | On-Demand | Spot/Preemptible | Notes |
|----------|-----------|------------------|-------|
| **Vast.ai** | ~$0.80/hr | — | Community, variable |
| **Lambda Labs** | $1.10/hr | — | Reliable |
| **RunPod** | $1.64/hr | — | Good availability |
| **Nebius** | N/A | N/A | H100/H200 only |
| **Google Cloud** | ~$3.67/hr (40GB) | ~$1.10/hr | A2 instances |
| **CoreWeave** | $2.06/hr | — | Enterprise |

### L40S 48GB (Budget SFT / QLoRA)

| Provider | On-Demand | Notes |
|----------|-----------|-------|
| **Nebius** | $1.35/hr | Good VRAM/price ratio |
| **RunPod** | ~$0.99/hr | Community cloud |

### Budget GPUs (QLoRA / Evaluation Only)

| Provider | GPU | Rate | Notes |
|----------|-----|------|-------|
| **Google Cloud** | L4 24GB | ~$0.70/hr (spot $0.24) | Good for eval, light QLoRA |
| **Google Cloud** | T4 16GB | ~$0.35/hr (spot $0.11) | Inference only |

## Estimated Training Costs for ABEE

### Scenario 1: Budget QLoRA ($25-50)

| Step | Platform | Config | Time | Cost |
|------|----------|--------|------|------|
| SFT QLoRA (8B) | RunPod | 4x L40S | ~2hr | ~$8 |
| GRPO QLoRA (8B) | RunPod | 4x L40S | ~4hr | ~$16 |
| Evaluation | Local | RTX 4060 Ti | ~2hr | $0 |
| **Total** | | | | **~$24** |

### Scenario 2: Standard LoRA ($100-200)

| Step | Platform | Config | Time | Cost |
|------|----------|--------|------|------|
| SFT LoRA (8B) | Lambda | 4x A100 80GB | ~4hr | ~$18 |
| GRPO LoRA (8B) | Lambda | 4x A100 80GB | ~8hr | ~$35 |
| Iteration (3x) | Lambda | 4x A100 80GB | ~12hr | ~$53 |
| Distillation | Vertex AI | Claude 3.5 Sonnet | ~100K tok | ~$15 |
| **Total** | | | | **~$121** |

### Scenario 3: Full Fine-Tuning ($500-1500)

| Step | Platform | Config | Time | Cost |
|------|----------|--------|------|------|
| SFT Full (8B) | Nebius | 8x H100 | ~10hr | ~$100 |
| GRPO Full (8B) | Nebius | 8x H100 | ~20hr | ~$200 |
| Iteration (5x) | Nebius | 8x H100 | ~40hr | ~$400 |
| Distillation | Vertex AI | Claude 3.5 Sonnet | ~500K tok | ~$75 |
| Experiment tracking | Vertex AI | TensorBoard | — | ~$20 |
| **Total** | | | | **~$795** |

## Cloud Distillation (Vertex AI)

| Model | Input Cost | Output Cost | Notes |
|-------|-----------|-------------|-------|
| Claude 3.5 Sonnet | $3/M tokens | $15/M tokens | Vision-capable. Use for reasoning trace generation. |
| Gemini 2.0 Flash | $0.10/M tokens | $0.40/M tokens | Cheaper but less capable for detailed reasoning. |

## Hardware Budget (Sensors — Post-Training Phase)

| Item | Cost | Priority | Value |
|------|------|----------|-------|
| FLIR Lepton XDS (RGB+thermal USB) | $239 | P0 | 500-2000ms pre-release signal. Novel contribution. |
| Intel RealSense D435i (depth camera) | $200 | P0 | Ground-truth depth at grip range. |
| Orbbec Femto Bolt (alt depth) | $400 | P1 | Azure Kinect replacement. |
| ESP32 WiFi CSI x3 | $30 | DEFERRED | 12.5cm resolution insufficient. |
| **P0 Total** | **$439** | | |

## NIM Deployment Costs

| Option | Cost | Notes |
|--------|------|-------|
| build.nvidia.com API | Free tier available | Best for prototyping/distillation |
| Self-hosted NIM (cloud) | GPU cost + NIM license | Requires >56GB VRAM (BF16) |
| Local Ollama (4-bit) | $0 | Already running. 16GB VRAM. |

## Cost Optimization Tips

1. **Start with QLoRA on RunPod** ($25/run) — iterate on dataset quality before scaling
2. **Use spot/preemptible** on GCP (70% savings) with DWS FLEX_START
3. **Nebius Explorer tier** — $1.50/hr H100 for first 1K hours
4. **Local RTX 4060 Ti** — QLoRA on 2B model, all evaluation, data preprocessing
5. **Distill with Gemini Flash first** ($0.10/M) for bulk, Claude for high-quality traces
