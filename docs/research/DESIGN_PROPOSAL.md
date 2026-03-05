# CLASP Next Stages: Design Proposal

## Table of Contents
1. [Training Pipeline Design](#1-training-pipeline-design)
2. [Cloud Infrastructure Options](#2-cloud-infrastructure-options)
3. [Advanced Sensing Modalities](#3-advanced-sensing-modalities)
4. [Implementation Roadmap](#4-implementation-roadmap)

---

## 1. Training Pipeline Design

### 1.1 Cosmos-RL: The Official Training Framework

NVIDIA provides **Cosmos-RL**, an async post-training framework purpose-built for Cosmos models. This is the primary tool for both SFT and RL (GRPO) fine-tuning.

**Supported modes:**
- Supervised Fine-Tuning (SFT) — available now
- GRPO (Group Relative Policy Optimization) — available for Reason1, coming for Reason2
- LoRA parameter-efficient fine-tuning

**Key reference:** The [Physical Plausibility Prediction recipe](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason1/physical-plausibility-check/post_training.html) from the Cosmos Cookbook is the closest analog to CLASP's task. It trains Cosmos-Reason to evaluate video for physical plausibility using SFT + GRPO.

#### Dataset Format (LLaVA-style)

```json
{
  "conversations": [
    {
      "role": "system",
      "content": [{"type": "text", "text": "You are a handoff safety evaluator..."}]
    },
    {
      "role": "user",
      "content": [
        {"type": "video", "video": "file:///path/to/trajectory.mp4", "fps": 4},
        {"type": "text", "text": "Should the robot release the object now? Evaluate grip stability, hand position, and transfer readiness."}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "<think>\nAnalyzing frame...\n</think>\n<answer>THINK</answer>"}]
    }
  ]
}
```

**Critical: fps=4** — Cosmos-Reason2 was trained at 4 FPS. All video inputs must match this.

#### CLASP SFT Dataset → Cosmos-RL Format

Our existing `SFTSerializer` outputs JSONL. The conversion pipeline:

```
CLASP dry-run/real runs
  → SFT records (data/sft_dataset.jsonl)
    → Convert to LLaVA conversations format
      → Split train/val/test
        → Upload to training infra (GCS/Brev)
```

**Data volume estimate:** At 50 trajectories × 20 frames × 4 agents = ~4,000 decision records per run. Need ~10-20 runs for a viable SFT dataset (~40K-80K records). Golden-only filtering reduces this to ~25% (correct ACTs).

#### TOML Configuration for CLASP SFT

```toml
[custom.dataset]
path = "data/clasp_sft_train"

[train]
epoch = 5
output_dir = "outputs/clasp_sft"
train_batch_per_replica = 16

[policy]
model_name_or_path = "nvidia/Cosmos-Reason2-8B"
model_max_length = 4096

[train.train_policy]
type = "sft"
conversation_column_name = "conversations"
mini_batch = 2

[train.ckpt]
enable_checkpoint = true

[policy.parallelism]
tp_size = 2
dp_shard_size = 4
pp_size = 1
```

#### GRPO Reward Design

Standard binary rewards cause sparsity. Use multi-component rewards:
- **Format reward**: Valid `<think>` tags present (+0.1)
- **Decision reward**: Correct ACT = +1.0, wrong ACT = -0.66 (early) / -0.33 (late)
- **Temporal proximity**: Partial credit for near-miss ACTs (+0.3 if within 2 frames of safe window)
- **THINK reward**: +0.05 per correct THINK (small, prevents degenerate all-THINK)

This mirrors the Life-Points system directly. Reference: VLA-R1 (arXiv:2510.01623) — GRPO on VLA with verifiable rewards.

**Two-stage pipeline:** SFT cold-start → GRPO. NVIDIA used this exact approach: +10% from SFT, +5% from RL.

#### GRPO Configuration for CLASP RL

```toml
[custom.dataset]
path = "data/clasp_grpo_train"

[train]
epoch = 3
output_dir = "outputs/clasp_grpo"

[policy]
model_name_or_path = "outputs/clasp_sft/checkpoints/latest/policy"
model_max_length = 6144

[train.train_policy]
type = "grpo"
kl_beta = 0.05
n_generation = 8
temperature = 0.9
max_response_length = 6144
mini_batch = 2

[policy.parallelism]
tp_size = 2
dp_shard_size = 4
pp_size = 1
```

### 1.2 Train/Test/Validation Split Strategy

For temporal trajectory data, standard random splits risk **temporal leakage**. Frame-level splitting is the most common mistake in video ML pipelines — it produces falsely optimistic validation numbers.

| Split | Allocation | Purpose |
|-------|-----------|---------|
| Train | 70% of trajectories | SFT + GRPO training |
| Validation | 15% of trajectories | Hyperparameter tuning, early stopping |
| Test | 15% of trajectories | Final accuracy reporting only |

**Key rules:**
- Split at the **trajectory/episode level**, never frame level (temporal leakage)
- Stratify by trajectory difficulty (early/mid/late safe windows)
- Hold out test set completely — never touch until final evaluation
- For k-fold: use `GroupKFold(groups=episode_ids)` from sklearn, NOT standard KFold
- Expanding-window splits for temporal generalization testing

**Minimum viable dataset:** 200 episodes (~4,000 labeled frames). Functional target: 500-1,000 episodes.

### 1.3 Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Accuracy** | % trajectories with correct release | >85% |
| **Premature Rate** | % trajectories with early release | <2% |
| **Late Rate** | % trajectories with late release | <10% |
| **No-Release Rate** | % trajectories with no release (too conservative) | <15% |
| **Temporal MAE** | Mean abs error: predicted vs true release frame | <3 frames |
| **Precision (ACT)** | Of all ACT decisions, % that were safe | >95% |
| **Recall (ACT)** | Of all safe frames, % where ACT was chosen | >60% |
| **ACT Precision** | Of all ACT decisions, % that were safe | **>0.95 REQUIRED before any deployment** (safety gate) |
| **Temporal Offset Error** | When predictions made vs ground-truth release frame | Positive bias OK (waits), negative bias NOT OK (premature) |
| **Calibration** | Reliability diagram of confidence vs accuracy | ECE < 0.05 (use Smooth ECE + temperature scaling post-training) |

### 1.4 Cloud Distillation Pipeline

Use Claude 3.5 Sonnet (via Vertex AI) to generate high-quality training data:

```
Video frames → Claude 3.5 Sonnet (vision) → rich reasoning traces
  → Filter by ground truth → Golden SFT records
    → Fine-tune Cosmos-Reason2-8B (student)
```

**Distillation strategy (EvoKD pattern):**
1. Identify current student (Cosmos-Reason2) failure mode categories
2. Generate targeted synthetic samples for those failure cases specifically
3. Include physics oracle label in prompt to ground teacher reasoning
4. Filter by teacher-oracle agreement (discard contradictions)

**Vertex AI Claude integration:**
```python
import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project="clasp-project", location="us-central1")
model = GenerativeModel("claude-3-5-sonnet@20240620")

response = model.generate_content([
    video_part,  # trajectory video
    "Analyze this human-robot handoff. For each frame, determine if release is safe..."
])
```

**NIM deployment of fine-tuned model:**
NIM officially supports Cosmos-Reason2-8B (>56GB VRAM, BF16). For fine-tuned models, requires `NIM_USE_TRTLLM_LEGACY_BACKEND=1` — default backend doesn't support custom weights. LoRA adapters from both HuggingFace PEFT and NeMo are supported.

---

## 2. Cloud Infrastructure Options

### 2.1 NVIDIA Brev (Recommended Primary)

**What it is:** NVIDIA-acquired GPU cloud platform that aggregates GPUs from multiple providers. One-click deployment with pre-configured environments.

**Key advantages for CLASP:**
- Direct integration with NVIDIA NGC catalog (NIM, NeMo, Cosmos models)
- Pre-configured CUDA, PyTorch, Jupyter environments
- CLI-based workflow: `brev shell <instance>` or `brev open <instance>` (VS Code)
- Launchables: one-click optimized environments for specific workloads

**Available GPUs:**
| GPU | VRAM | Typical Rate |
|-----|------|-------------|
| L40S | 48GB | ~$1.35/hr |
| H100 80GB | 80GB | ~$2.95-3.50/hr |
| H200 | 141GB | ~$3.50/hr |
| A100 80GB | 80GB | ~$1.50-2.50/hr |

**Key insight:** Cosmos-Reason2-8B is based on **Qwen3-VL-8B-Instruct**, so any Qwen3-VL-compatible training framework works (TRL SFTTrainer, cosmos-rl, NeMo).

**For CLASP SFT (8B model):**
- Budget: 1× A100 40GB QLoRA (4-bit, r=16) — ~$1.10-3.67/hr spot — estimated $3.50-4.50 per 3-epoch run
- Minimum: 4× A100 80GB (~$6-10/hr) — matches Cosmos-RL requirements
- Optimal: 8× H100 80GB (~$24-28/hr) — faster training, matches cookbook config

**Setup:**
```bash
# Install Brev CLI
curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install.sh | sh

# Create GPU instance
brev create clasp-training --gpu h100:4

# Connect
brev shell clasp-training

# Inside instance: install Cosmos-RL
git clone https://github.com/nvidia-cosmos/cosmos-reason2
cd cosmos-reason2/examples/cosmos_rl
uv sync
```

### 2.2 Google Vertex AI (Recommended for Distillation + Experiment Tracking)

**Use for:**
- Claude 3.5 Sonnet distillation (via Vertex AI Model Garden)
- Experiment tracking (Vertex AI Experiments / Vertex AI TensorBoard)
- Dataset management (GCS buckets)
- Evaluation pipelines

**NOT recommended for:** Managed SFT (only supports Gemini + a few Model Garden entries). CAN be used for training via Custom Training Jobs (Docker container → Artifact Registry → Python SDK), but Brev/DGX Cloud have better NeMo/Cosmos-RL integration out of the box.

**GPU pricing (custom training jobs):**
| GPU | On-Demand | Spot |
|-----|-----------|------|
| T4 16GB | ~$0.35/hr | ~$0.11/hr |
| L4 24GB | ~$0.70/hr | ~$0.24/hr |
| A100 40GB | ~$2.50/hr | ~$0.75/hr |
| A100 80GB | ~$3.80/hr | ~$1.14/hr |
| H100 80GB | ~$11.06/hr (8-GPU min) | N/A |

**Vertex AI training pipeline (for custom containers):**
```python
from google.cloud import aiplatform

aiplatform.init(project="clasp-cookoff", location="us-central1")

job = aiplatform.CustomContainerTrainingJob(
    display_name="clasp-sft-run",
    container_uri="nvcr.io/nvidia/nemo:24.12",
    model_serving_container_image_uri="nvcr.io/nvidia/tritonserver:24.12",
)

model = job.run(
    replica_count=1,
    machine_type="a2-ultragpu-4g",  # 4x A100 80GB
    accelerator_type="NVIDIA_A100_80GB",
    accelerator_count=4,
    args=["--config", "configs/clasp_sft.toml"],
)
```

### 2.3 Alternative GPU Providers (Budget Options)

| Provider | H100 Rate | A100 Rate | Key Feature |
|----------|-----------|-----------|-------------|
| **Vast.ai** | $1.49/hr | ~$0.80/hr | Cheapest, community GPUs |
| **Nebius** (Cosmos partner) | $2.95/hr | N/A | Cookoff sponsor, Explorer tier $1.50/hr first 1K hrs |
| **RunPod** | $1.99/hr (community) | $1.64/hr | Serverless GPUs, Datature confirmed 4×L40S QLoRA ~$25/run |
| **Lambda Labs** | $2.76/hr | $1.10/hr | Dev-friendly, SSH access |
| **CoreWeave** | $4.76/hr | $2.06/hr | K8s-native, best for large distributed runs |

**Key finding:** Datature tutorial confirms 4× A10G/L40S (24GB each) works with QLoRA for Cosmos-Reason2-8B at roughly **$25 per training run** on RunPod.

**Cookoff note:** 3rd-place Cookoff prize includes $500 in Brev credits. Nebius credits were distributed at registration but now exhausted.

### 2.4 Local Training (RTX 4060 Ti 16GB)

Our hardware can handle:
- **QLoRA on 2B model**: Cosmos-Reason2-2B with 4-bit quantization fits in 16GB
- **LoRA on 8B model**: NOT feasible locally (needs ~24GB minimum for inference alone)
- **Inference/evaluation**: 4-bit quantized 8B works (already running via Ollama)
- **Data preprocessing**: SAM2, MiDaS, embedding generation all local

**Local QLoRA stack (confirmed feasible on 16GB):**
- 4-bit NF4 quantization + gradient checkpointing + paged_adamw_8bit + Flash Attention 2
- VRAM: 12-14GB for 8B model with QLoRA (rank=32)
- LR: 1e-5 with cosine decay (official Cosmos-Reason2 full-model LR is 2e-7)
- Effective batch=32 via batch=2, grad_accum=16

```bash
python train_local.py \
  --model nvidia/Cosmos-Reason2-2B \
  --lora_rank 32 \
  --quantize nf4 \
  --batch_size 2 \
  --gradient_accumulation 16 \
  --lr 1e-5 \
  --epochs 3 \
  --data data/clasp_sft_train.jsonl
```

### 2.5 Recommended Hybrid Pipeline

```
┌──────────────────────────────────────────────────┐
│                 CLASP Training Pipeline            │
├──────────────────────────────────────────────────┤
│                                                   │
│  LOCAL (RTX 4060 Ti 16GB)                        │
│  ├─ Data collection (CLASP dry-run + real runs)   │
│  ├─ SAM2 + MiDaS physics oracle                 │
│  ├─ SFT dataset curation                        │
│  ├─ QLoRA on 2B model (rapid iteration)          │
│  └─ Evaluation / inference                       │
│                                                   │
│  VERTEX AI (Claude 3.5 Sonnet)                   │
│  ├─ Cloud distillation (rich reasoning traces)   │
│  ├─ Experiment tracking                          │
│  └─ Dataset versioning (GCS)                     │
│                                                   │
│  BREV / NEBIUS (4-8× H100)                       │
│  ├─ Full SFT on Cosmos-Reason2-8B               │
│  ├─ GRPO reinforcement learning                  │
│  ├─ Full LoRA fine-tuning                        │
│  └─ Model checkpointing                          │
│                                                   │
│  DEPLOYMENT (Local / NIM)                        │
│  ├─ Pull fine-tuned model → Ollama               │
│  ├─ Or NIM endpoint (cloud)                      │
│  └─ Run CLASP with fine-tuned backbone            │
│                                                   │
└──────────────────────────────────────────────────┘
```

### 2.6 Cosmos Cookoff Submission Requirements

**Deadline: March 5, 2026** (TODAY)

**Required deliverables:**
1. Finalized repository with clean README
2. Working demo video showing system in action
3. Clear problem statement (physical AI use case)
4. Integration and explanation of Cosmos Reason 2
5. Post to nvidia-cosmos/discussions/4

**Judging criteria:** Technical implementation, design, potential impact, quality of ideas
**Judges:** Experts from Datature, Hugging Face, Nebius, Nexar, NVIDIA
**Prizes:** $5,000, NVIDIA DGX Spark, and more
**Winners announced:** Week of March 16 (GTC San Jose)

---

## 3. Advanced Sensing Modalities (Post-Training Phase)

These additions are for AFTER smooth training/testing, to enhance the physical understanding of handoff scenes.

### 3.1 Gaussian Splatting (3DGS)

**What:** 3D scene reconstruction from images/video using differentiable Gaussian primitives. Creates photorealistic 3D representations in real-time.

**Value for CLASP:**

| Aspect | Current (SAM2+MiDaS) | With 3DGS |
|--------|----------------------|-----------|
| Depth | Monocular estimate (noisy) | True 3D geometry |
| Object pose | 2D bounding box | Full 6-DOF pose |
| Grip analysis | Segmentation mask only | 3D contact surface |
| Occlusion | Lost information | View synthesis around occlusions |
| Real-time | Yes (~30ms) | Yes (~33ms at 30Hz) |

**Key papers for robotics:**
- **POGS (ICRA 2025, Berkeley)**: Directly tracks human-robot object handoffs via 3DGS pose tracking. 12 consecutive successful handoffs, recovers from 80% of in-grasp perturbations. Single stereo camera. **Most relevant paper for CLASP.**
- **RoboSplat** (RSS 2025): One-shot manipulation via 3DGS demo generation. 87.8% success vs 57.2% with 2D augmentation.
- **GaussianGrasper**: Language-guided robotic grasping using 3DGS scene representation.
- **Splat-MOVER**: Multi-stage open-vocabulary manipulation via editable Gaussian Splatting. Real-time 30Hz operation.
- **PUGS (ICRA 2025)**: Predicts physical properties (mass, friction, hardness) from Gaussian geometry zero-shot — directly useful for grip stability estimation.
- **GaussianVLM**: Compresses ~40K Gaussians to 132 tokens for VLM input. 5x improvement over prior 3D VLMs.
- **SplatTalk (ICCV 2025)**: Posed RGB → tokens compatible with standard LLMs. Zero-shot 3D VQA.

**NVIDIA integration:** NVIDIA stated (Aug 2025) that "Cosmos models use Gaussian Splat scenes as the geometric backbone." NuRec library makes 3DGS first-class in Isaac Sim. Warp + gsplat runs physics correction at 33ms — matching CLASP's frame budget.

**Feasibility on RTX 4060 Ti 16GB:**
- Inference: YES — 3DGS rendering is very GPU-efficient (~10-30ms per frame)
- Training/reconstruction: LIMITED — initial scene reconstruction takes 5-30 minutes for a new scene, but can be done offline
- Real-time update: CHALLENGING — incremental updates possible but may compete with VLM for VRAM

**Integration approach:**
```
Camera feed → 3DGS reconstruction (offline/periodic)
  → Per-frame: render depth + normal maps from 3DGS
    → Replace MiDaS depth with 3DGS depth
      → Feed into Physics Oracle as higher-quality depth
```

**NVIDIA ecosystem:** NVIDIA has invested heavily in 3DGS via Instant NGP, NeuralVDB, and Kaolin. The Cosmos ecosystem doesn't yet directly integrate 3DGS, but the depth/normal outputs are compatible with Cosmos-Reason2 video input.

**Recommendation: HIGH VALUE, MEDIUM EFFORT**
- Phase 1: Swap MiDaS for Depth Anything V2 (quick win, better monocular depth)
- Phase 2: Add 3DGS depth refinement — metric depth + normals from Gaussians
- Phase 3: POGS-style object pose tracker as oracle veto signal
- Phase 4: 3DGS scene tokens injected into Cosmos-Reason2 (via GaussianVLM approach)
- RTX 4060 Ti 16GB: tight but viable for small handoff scenes (~50-200K Gaussians, 40-70 FPS)

### 3.2 Infrared (IR) Sensing

**What:** Thermal imaging detects heat signatures from human hands during object handoff — grip contact, thermal transfer, and release dynamics.

**Value for CLASP:**

| Signal | What IR Reveals | Impact on Handoff |
|--------|----------------|-------------------|
| Contact heat | Thermal fingerprint where hand touches object | Confirms grip contact area |
| Heat decay | Cooling rate after hand lifts | Detects release timing within ~100ms |
| Grip pressure proxy | Higher contact temperature = tighter grip | Predicts grip stability |
| Pre-release signals | Subtle temperature changes before release | Early warning (~200ms) |

**Key research:**
- PMC study: "IR cameras detect areas where subjects' fingers have touched objects" — heat signatures differentiate grip types that look visually identical in RGB
- MOTIF Hand (USC): Embedded IR camera in robotic palm for contactless temperature sensing
- Cornell: Wristband with 4 thermal cameras for 3D hand pose tracking
- VideoPhy2: Thermal features improve physical plausibility prediction

**The physics:** When a hand grips an object, conductive heat transfer creates measurable thermal elevation at contact zones. As grip loosens (500-2000ms before release), contact area decreases and a **micro-cooling signal** appears — a leading indicator with **no RGB equivalent**. A 2025 study in Advanced Engineering Materials measured gripper finger temperature profiles during contact/release cycles confirming this gradient.

**Novel contribution:** No study has yet used thermal grip dynamics as VLM input for handoff safety — this would be a first.

**Hardware options (user has IR electronics):**
| Device | Resolution | FPS | Price | Interface |
|--------|-----------|-----|-------|-----------|
| **FLIR Lepton XDS** (Feb 2026) | 160×120 + 5MP RGB | 9Hz | ~$239 | USB (factory-aligned!) |
| FLIR Lepton 3.5 | 160×120 | 9Hz | ~$250 | SPI/I2C |
| MLX90640 | 32×24 | 16Hz/32Hz | ~$70 | I2C |
| Seek Thermal CompactPRO | 320×240 | 15Hz | ~$500 | USB |

**Integration approach (zero model modification):**
```
FLIR Lepton XDS → thermal frame (factory-aligned with RGB)
  → Render as false-color → compose 2-panel: [RGB | Thermal]
    → Feed to Cosmos-Reason2 as single image with panel-description prompt
  → Also: threshold(T > T_ambient + ΔT) → contact mask
    → "grip_thermal_confidence" score → Physics Oracle veto signal
```

For CLASP's modality_mask system, IR becomes a 4th mask option: `["full", "gripper", "velocity", "thermal"]`

ThermEval benchmark (arXiv Feb 2026) confirms VLMs can reason about thermal images with appropriate prompting.

**Recommendation: HIGHEST VALUE, LOW EFFORT**
- Micro-cooling signal 500-2000ms before visible release — the single best early warning for handoff safety
- Nearly deterministic — no ML needed for basic grip/no-grip contact detection
- FLIR Lepton XDS ($239) gives factory-aligned RGB+thermal in one USB module
- Novel contribution to the field (no prior work on thermal VLM handoff)
- ~1 week for basic integration

### 3.3 WiFi CSI (Channel State Information) as Omnidirectional 3D Gauge

**What:** WiFi signals penetrate objects and reflect off surfaces. By analyzing Channel State Information (CSI), you can create 3D spatial representations — essentially WiFi-based radar.

**Value for CLASP:**

| Capability | Current State (2025-2026) | Relevance |
|-----------|---------------------------|-----------|
| Human pose estimation | ~85% accuracy (room-level) | LOW — camera is better for close-range |
| Object detection | Proof-of-concept stage | LOW — resolution too coarse |
| 3D point clouds from CSI | ~10mm RMSE, ICP fitness ~0.64 | MEDIUM — complementary to vision |
| Through-occlusion sensing | Strong — WiFi penetrates objects | HIGH — sees behind hand/object |
| Room-scale geometry | Good for static environments | LOW — we're close-range only |

**Key research:**
- **CSI2PC** (2024): WiFi CSI → 3D point clouds via transformer. 2.31ms GPU inference, 10mm RMSE. However, "limitations in depth estimation and vertical angle coverage."
- **RoboMNIST** (Nature 2025): Multi-robot activity recognition using WiFi CSI + video + audio. Demonstrates multi-modal fusion feasibility.
- **MobileCSI**: Mobile WiFi-based object detection using CSI sniffers on robotic platforms.

**Physics limit:** At 2.4GHz, spatial resolution is ~12.5cm — fine for detecting a person approaching or handoff posture, but **cannot resolve grip state or finger contact**. 802.11bf-2025 standardizes sensing at 60GHz (5mm resolution — genuinely useful for hand gestures) but 60GHz hardware is not commercially available yet.

**Hardware requirements:**
- ESP32 with CSI extraction firmware (~$10/board) × 2-3
- Or Intel 5300 NIC / Atheros CSI Tool-compatible chipset
- 2-3 WiFi access points for triangulation

**Feasibility assessment:**
- **Resolution**: 12.5cm at 2.4GHz — cannot detect grip state
- **Latency**: 2-30ms — acceptable
- **Setup complexity**: HIGH — calibration, multiple APs, firmware
- **Best use**: Coarse body-pose / approach detection, through-wall sensing

**Recommendation: LOW VALUE FOR HANDOFF, DEFER**
- 12.5cm resolution fundamentally insufficient for grip/release detection
- Useful only as safety perimeter sensor (detect human approaching robot)
- 60GHz would change this equation but hardware not available yet
- ~3-4 weeks implementation, high uncertainty
- **Defer entirely — invest that time in thermal + 3DGS instead**

### 3.4 LiDAR

**What:** Laser-based depth sensing with millimeter accuracy. Unlike monocular depth (MiDaS), LiDAR provides ground-truth 3D point clouds.

**Value for CLASP:**

| Aspect | MiDaS Depth | LiDAR |
|--------|-------------|-------|
| Accuracy | ~10-20% error at 1m | ~1mm at 1m |
| Range | Any (monocular estimation) | 0.1-10m typical |
| Resolution | Per-pixel (image res) | 640-1024 points per scan |
| Real-time | ~50ms | ~10ms |
| Cost | Free (software) | $100-$500 |

**Important:** Spinning LiDAR (RPLiDAR etc.) is the wrong class — it produces 2D horizontal scans, useless for 3D grip geometry. The correct tool is a **depth camera**.

**Hardware options:**
| Device | Type | Range | Price | Interface |
|--------|------|-------|-------|-----------|
| **Orbbec Femto Bolt** | ToF depth | 0.25-5m | ~$400 | USB (Azure Kinect replacement) |
| **Intel RealSense D435i** | Stereo depth | 0.15-3m | ~$200 | USB-C (best for close-range grip zone) |
| Intel RealSense D455 | Stereo depth | 0.6-6m | ~$300 | USB-C |
| Livox Mid-360 | 3D LiDAR | 0.1-70m | ~$500 | Ethernet (overkill for desktop) |

**Integration approach:**
```
LiDAR → point cloud (aligned with RGB via extrinsic calibration)
  → Extract hand/object depth at segmented regions (SAM2 masks)
    → Compare with MiDaS estimates → weighted fusion
      → Physics Oracle: ground-truth depth for velocity/acceleration calculation
```

**Recommendation: MEDIUM VALUE, LOW EFFORT**
- Ground-truth depth dramatically improves velocity estimation (key for handoff timing)
- Intel RealSense D455 gives both stereo depth AND RGB — could replace current camera
- ~1 week integration if using ROS2 driver
- **Best bang-for-buck of all additional sensors**

### 3.5 Layered Visual Feed Architecture

Combining all modalities into a unified "layered visual feed":

```
Layer 0: RGB video (existing) ─────────── [H×W×3]
Layer 1: Depth map (3DGS or LiDAR) ───── [H×W×1]
Layer 2: Segmentation masks (SAM2) ────── [H×W×N_classes]
Layer 3: Thermal overlay (IR) ─────────── [H×W×1]
Layer 4: Normal maps (from 3DGS) ──────── [H×W×3]
Layer 5: Velocity field (optical flow) ── [H×W×2]
Layer 6: WiFi heatmap (if available) ──── [H×W×1] (low-res, upscaled)
                                           ─────────
                                           Total: H×W×(3+1+N+1+3+2+1) channels
```

**Feeding to VLM:**
Cosmos-Reason2 accepts video (RGB). For multi-modal input:

**Option A: Channel stacking** — Encode layers as false-color channels in video frames. Simple but lossy.

**Option B: Multi-view rendering** — Render each layer as a separate "camera view" tiled in a single frame. VLM sees a 2×3 grid of perspectives.

**Option C: Text description** — Convert non-visual layers to text descriptions injected into the prompt. "Thermal: contact detected at 34.2°C, area=12cm². Depth: hand at 0.42m, object at 0.38m."

**Option D: Hybrid** — RGB video for VLM + structured text for non-visual sensors. This is the most practical approach:

```python
prompt = f"""
You are evaluating a human-robot handoff.

Visual: [video frames at 4 FPS]
Depth: hand={depth_hand:.3f}m, object={depth_obj:.3f}m, gap={gap:.3f}m
Thermal: contact_area={thermal_area:.1f}cm², peak_temp={peak_temp:.1f}°C, grip_confidence={grip_conf:.2f}
Velocity: hand_vel={hand_vel:.3f}m/s, object_vel={obj_vel:.3f}m/s, relative={rel_vel:.3f}m/s
LiDAR: point_density_hand={pts_hand}, point_density_object={pts_obj}

Should the robot release now? Think step by step.
"""
```

**Recommendation: Option D (Hybrid) is optimal** — visual understanding from RGB video, precise numerics from structured text. Avoids the information loss of trying to encode everything visually.

---

## 4. Implementation Roadmap

### Phase 1: Cookoff Submission (TODAY — March 5, 2026)
- [x] CLASP system running with dry-run
- [x] Life-Points, GRPO, dynamic consensus implemented
- [ ] Record demo video
- [ ] Finalize README
- [ ] Submit to nvidia-cosmos/discussions/4

### Phase 2: Real Inference + Data Collection (Week 1-2 post-submission)
- [ ] Run CLASP with real Cosmos-Reason2-8B inference (local 4-bit)
- [ ] Collect 500+ trajectory SFT records
- [ ] Validate ArchiveKV population with real embeddings
- [ ] Run cloud distillation via Vertex AI (Claude 3.5 Sonnet)
- [ ] Build train/val/test splits

### Phase 3: SFT Fine-Tuning (Week 2-3)
- [ ] Local: QLoRA on Cosmos-Reason2-2B (rapid iteration)
- [ ] Cloud: Full SFT on Cosmos-Reason2-8B via Brev (4× A100)
- [ ] Evaluate: premature rate, accuracy, temporal MAE
- [ ] Iterate on dataset filtering and prompt engineering

### Phase 4: GRPO Reinforcement Learning (Week 3-4)
- [ ] Define reward function (maps to CLASP's Life-Points scoring)
- [ ] Run GRPO on SFT checkpoint (8× H100 via Brev/Nebius)
- [ ] Evaluate improvement over SFT baseline
- [ ] Final model selection

### Phase 5: Advanced Sensing Integration (Week 4-8, if all goes well)

**Priority order (by value/effort ratio):**

1. **Infrared (FLIR Lepton XDS $239)** — 1 week, 500-2000ms early warning before visible release, novel contribution
2. **Depth Camera (RealSense D435i $200)** — 1 week, ground-truth depth at grip range (15-50cm)
3. **Gaussian Splatting** — 2 weeks, 3D scene understanding, POGS-style handoff tracking
4. **WiFi CSI** — DEFER (12.5cm resolution insufficient for grip detection)

Each modality integrates via the **Hybrid prompt approach** (Option D), feeding structured numeric data alongside RGB video to the VLM.

---

## Appendix A: Cost Estimates

### Training Budget (Minimum Viable — RunPod QLoRA)

| Item | Hours | Rate | Cost |
|------|-------|------|------|
| SFT QLoRA on 8B (RunPod, 4×L40S) | ~2hr/run × 5 runs | ~$5/hr | $50 |
| GRPO on 8B (RunPod, 4×L40S) | ~4hr/run × 3 runs | ~$5/hr | $60 |
| Distillation (Vertex Claude) | ~100K tokens | $3/M in | $15 |
| GCS storage | 50GB/mo | $0.02/GB | $1 |
| **Total** | | | **~$126** |

### Training Budget (Optimal)

| Item | Hours | Rate | Cost |
|------|-------|------|------|
| SFT on 8B (Nebius, 8×H100) | 20hr | $24/hr | $480 |
| GRPO on 8B (Nebius, 8×H100) | 40hr | $24/hr | $960 |
| Distillation (Vertex Claude) | ~500K tokens | $3/M in | $75 |
| Experiment tracking (Vertex) | | | $20 |
| **Total** | | | **~$1,535** |

### Hardware Budget (Sensing)

| Item | Cost | Priority |
|------|------|----------|
| FLIR Lepton XDS (RGB+thermal) | ~$239 | P0 — novel, highest value |
| Intel RealSense D435i (depth) | ~$200 | P0 — ground-truth depth |
| Orbbec Femto Bolt (alt depth) | ~$400 | P1 — if D435i insufficient |
| ESP32 WiFi CSI ×3 | ~$30 | DEFERRED |
| **Total (P0)** | **~$439** | |

## Appendix B: Key Sources

- [Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/index.html) — Official post-training recipes
- [Cosmos-RL SFT Guide](https://docs.nvidia.com/nemo/rl/latest/guides/sft.html) — NeMo RL SFT documentation
- [Cosmos-Reason2 GitHub](https://github.com/nvidia-cosmos/cosmos-reason2) — Model repo with training examples
- [NVIDIA Brev](https://docs.nvidia.com/brev/latest/about-brev.html) — GPU cloud platform
- [Nebius Pricing](https://nebius.com/pricing) — Cosmos Cookoff partner pricing
- [Cosmos Cookoff Forum](https://forums.developer.nvidia.com/t/cosmos-cookoff-final-stretch-before-submissions-close/361850) — Competition announcements
- [Post-Training Cosmos Reason Blog](https://developer.nvidia.com/blog/maximize-robotics-performance-by-post-training-nvidia-cosmos-reason/) — NVIDIA technical blog
- [RoboSplat (RSS 2025)](https://github.com/InternRobotics/RoboSplat) — 3DGS for robotic manipulation
- [GaussianGrasper](https://arxiv.org/html/2403.09637v1) — Language-guided grasping with 3DGS
- [WiFi CSI 3D Point Clouds](https://arxiv.org/html/2410.16303v1) — CSI2PC transformer
- [IR Contact Detection (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8883210/) — Thermal hand contact measurement
- [POGS (ICRA 2025)](https://arxiv.org/abs/2503.05189) — 3DGS handoff tracking (12 consecutive handoffs)
- [GaussianVLM](https://arxiv.org/abs/2507.00886) — 3D scene compressed to 132 VLM tokens
- [SplatTalk (ICCV 2025)](https://arxiv.org/html/2503.06271v1) — Zero-shot 3D VQA from posed RGB
- [VLA-R1](https://arxiv.org/abs/2510.01623) — GRPO on VLA with verifiable rewards
- [Datature Cosmos-Reason2 Tutorial](https://datature.io/blog/finetuning-your-own-cosmos-reason2-model) — QLoRA on 4xL40S confirmed ~$25/run
- [NIM Fine-tuned Model Deployment](https://docs.nvidia.com/nim/) — Legacy backend flag required
