# Training Pipeline Architecture

## Document Purpose
Defines the end-to-end training pipeline for ABEE, spanning local infrastructure,
Google Cloud Vertex AI, and NVIDIA Brev. This is the authoritative reference for
how data flows from collection through training to deployment.

---

## 1. Pipeline Overview

```
                        ABEE TRAINING PIPELINE
                        ======================

  ┌─────────────────────────────────────────────────────────────┐
  │                    LOCAL (RTX 4060 Ti 16GB)                 │
  │                                                             │
  │  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
  │  │  Video    │───→│  ABEE    │───→│   SFT    │             │
  │  │  Source   │    │ Orchest. │    │  JSONL   │──┐          │
  │  └──────────┘    └──────────┘    └──────────┘  │          │
  │       │               │               │        │          │
  │       │          ┌────┴────┐          │        │          │
  │       │          │ Physics │          │        │          │
  │       │          │ Oracle  │          │        │          │
  │       │          │SAM2+MiDaS│         │        │          │
  │       │          └─────────┘          │        │          │
  │       │                               │        │          │
  │  ┌────┴─────┐    ┌──────────┐        │        │          │
  │  │ArchiveKV │←───│  Golden  │←───────┘        │          │
  │  │  (FAISS) │    │ Memories │                  │          │
  │  └──────────┘    └──────────┘                  │          │
  └────────────────────────────────────────────────┼──────────┘
                                                   │
                          ┌────────────────────────┘
                          │  Upload
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │              GOOGLE CLOUD VERTEX AI                         │
  │                                                             │
  │  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
  │  │ Datasets │───→│ Colab    │───→│  GCS     │             │
  │  │(versions)│    │Enterprise│    │ Bucket   │             │
  │  └──────────┘    │(convert) │    │(LLaVA)   │             │
  │                  └──────────┘    └────┬─────┘             │
  │                                       │                    │
  │  ┌──────────┐    ┌──────────┐        │                    │
  │  │ Distill  │───→│ Enriched │───→────┘                    │
  │  │ (Claude) │    │ Traces   │                              │
  │  └──────────┘    └──────────┘                              │
  │                                                             │
  │  ┌──────────────────────────────────────────┐              │
  │  │         Custom Training Job              │              │
  │  │  ┌────────────────────────────────┐      │              │
  │  │  │  Container: cosmos-rl          │      │              │
  │  │  │  GPU: 4x A100 40GB (spot)     │      │              │
  │  │  │  Config: abee_sft.toml        │      │              │
  │  │  │  Output: checkpoint → GCS     │      │              │
  │  │  └────────────────────────────────┘      │              │
  │  └──────────────────────────────────────────┘              │
  │                       │                                     │
  │  ┌──────────┐        │        ┌──────────┐                │
  │  │Experiments│←───────┘       │  Model   │                │
  │  │(tracking)│                 │ Registry │                │
  │  └──────────┘                 └────┬─────┘                │
  └────────────────────────────────────┼───────────────────────┘
                                       │
                          ┌────────────┘
                          │  Download checkpoint
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │              LOCAL DEPLOYMENT                                │
  │                                                             │
  │  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
  │  │Fine-tuned│───→│  Ollama  │───→│   ABEE   │             │
  │  │Checkpoint│    │  / vLLM  │    │(improved)│             │
  │  └──────────┘    └──────────┘    └──────────┘             │
  └─────────────────────────────────────────────────────────────┘
```

## 2. Data Flow Specification

### 2.1 Stage 1: Data Collection (Local)

**Input:** Video trajectories (real or synthetic)
**Process:** ABEE orchestrator runs with Cosmos-Reason2-8B (4-bit, Ollama)
**Output:** `data/sft_dataset.jsonl` — one record per agent-frame decision

Record schema:
```json
{
  "trajectory_id": "string",
  "frame_idx": "int",
  "agent_name": "string",
  "agent_bias": "string (first 80 chars of prompt)",
  "temporal_stride": "int",
  "modality_mask": "string (full|gripper|velocity)",
  "decision": "THINK|ACT",
  "confidence": "float [0,1]",
  "think_trace": "string (chain-of-thought)",
  "is_correct": "bool",
  "ground_truth_t_release": "int",
  "embedding_snippet": "[float] (first 16 dims)",
  "golden_rule": "string|null"
}
```

### 2.2 Stage 2: Dataset Preparation (Vertex AI — Colab Enterprise)

**Input:** ABEE JSONL from Stage 1
**Process:** Colab notebook converts to LLaVA conversation format
**Output:** GCS bucket with train/val/test splits

Conversion: ABEE JSONL → LLaVA conversations:
```json
{
  "conversations": [
    {
      "role": "system",
      "content": [{"type": "text", "text": "<agent_prompt_bias>"}]
    },
    {
      "role": "user",
      "content": [
        {"type": "video", "video": "gs://abee-data/trajectories/<id>.mp4", "fps": 4},
        {"type": "text", "text": "Frame <N>: Should the robot release? Evaluate grip stability, velocity, and transfer readiness.\nAnswer format:\n<think>\nYour reasoning\n</think>\n<answer>THINK or ACT</answer>"}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "<think>\n<think_trace>\n</think>\n<answer><decision></answer>"}]
    }
  ]
}
```

Split strategy:
- GroupKFold by trajectory_id (never split frames across sets)
- 70% train / 15% validation / 15% test
- Stratify by trajectory difficulty

### 2.3 Stage 3: Cloud Distillation (Vertex AI — Claude 3.5 Sonnet)

**Input:** Video trajectories + ground-truth labels
**Process:** Claude generates rich reasoning traces (teacher model)
**Output:** Enriched SFT records with detailed chain-of-thought

Pipeline:
```
For each trajectory:
  1. Send video frames + ground truth to Claude 3.5 Sonnet
  2. Claude generates detailed reasoning per frame
  3. Filter by oracle-teacher agreement
  4. Merge enriched traces into SFT dataset
```

EvoKD strategy:
- Identify student failure modes (premature ACTs, missed safe windows)
- Generate targeted synthetic samples for failure categories
- Progressively harden the training set

### 2.4 Stage 4: SFT Training (Vertex AI — Custom Training Job)

**Input:** GCS bucket with LLaVA dataset
**Compute:** Custom Training Job, 4x A100 40GB (spot)
**Framework:** cosmos-rl (Docker container from NVIDIA NGC)
**Output:** SFT checkpoint → GCS

Training configuration (TOML):
```toml
[custom.dataset]
path = "gs://abee-data/sft_train"

[train]
epoch = 5
output_dir = "gs://abee-checkpoints/sft"
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
dp_shard_size = 2
pp_size = 1

[policy.dtensor_cfg.lora_cfg]
enabled = true
dim = 32
alpha = 32
match_all_linear = true
```

### 2.5 Stage 5: GRPO Training (Vertex AI or Brev — Sequential after SFT)

**Input:** SFT checkpoint + reward function
**Compute:** Custom Training Job, 4-8x A100/H100
**Output:** GRPO-refined checkpoint

Reward function:
```
R(decision, ground_truth, frame_idx) =
  +1.0   if correct ACT (in safe window)
  -0.66  if premature ACT (before safe window)
  -0.33  if late ACT (after safe window)
  +0.05  if correct THINK (before safe window)
  -0.02  if unnecessary THINK (in safe window, but recoverable)
  +0.1   if valid <think> tags present (format reward)
  +0.3   if ACT within 2 frames of safe window (partial credit)
```

### 2.6 Stage 6: Evaluation + Deployment (Local)

**Input:** Fine-tuned checkpoint from GCS
**Process:** Load into Ollama/vLLM, run ABEE evaluation suite
**Safety gate:** ACT Precision > 0.95 required before deployment

Metrics tracked:
- Accuracy, premature rate, late rate, no-release rate
- ACT Precision, ACT Recall
- Temporal Offset Error (positive bias acceptable, negative NOT)
- Expected Calibration Error (ECE < 0.05)

---

## 3. Parallelization Strategy

### What CAN Run in Parallel

```
PARALLEL TRACK A              PARALLEL TRACK B
(Data Quality)                (Infrastructure)
──────────────                ──────────────────
Collect more trajectories     Build Docker container for Vertex AI
Curate golden memories        Set up GCS buckets + IAM
Run cloud distillation        Write Colab conversion notebook
                              Configure Vertex Experiments

PARALLEL TRACK C              PARALLEL TRACK D
(Local Experiments)           (Sensor Prep)
──────────────                ──────────────────
QLoRA on 2B model locally     Order FLIR Lepton XDS
Test prompt variations        Order RealSense D435i
Ablation studies              Prototype integration code
```

### What MUST Be Sequential

```
SFT Dataset Ready ──→ SFT Training ──→ SFT Checkpoint
                                            │
                                            ▼
                                      GRPO Training ──→ GRPO Checkpoint
                                                            │
                                                            ▼
                                                      Evaluation
                                                            │
                                                   ┌────────┴────────┐
                                                   │                 │
                                              PASS (deploy)    FAIL (iterate)
                                                              └──→ back to SFT
```

**Why sequential:** GRPO needs the SFT checkpoint as its starting point. The SFT model
provides the "cold start" — without it, GRPO has no policy to refine. And each
evaluation informs the next iteration's data curation.

### What We CAN Parallelize Within Training

- **Multiple LoRA rank experiments** (r=8, r=16, r=32) — submit 3 Vertex jobs at once
- **Hyperparameter sweeps** via Vertex AI Vizier (learning rate, batch size)
- **Different data subsets** — train on golden-only vs all data simultaneously
- **2B vs 8B model** — QLoRA on 2B locally while 8B trains in cloud

---

## 4. Vertex AI Service Mapping

| Vertex AI Service | ABEE Use | Stage |
|---|---|---|
| **Datasets** | Version SFT JSONL, manage splits | 2 |
| **Colab Enterprise** | Interactive data prep, format conversion, quick experiments | 2 |
| **Custom Training (Model dev > Training)** | SFT + GRPO jobs on A100s | 4, 5 |
| **Experiments** | Track all runs: loss, accuracy, ACT precision per epoch | 4, 5 |
| **Model Registry** | Store fine-tuned checkpoints with metadata | 5, 6 |
| **GenAI Evaluation** | Evaluate model outputs against ground truth | 6 |
| **Vector Search** | (Future) Replace local FAISS ArchiveKV | Later |
| **RAG Engine** | (Future) Replace manual RAG retrieval | Later |
| **Provisioned Throughput** | (Future) Production inference if needed | Later |

### NOT Used

| Service | Reason |
|---|---|
| Tuning (managed) | Only supports Gemini/Llama, not Cosmos-Reason2 |
| Agent Builder/Designer | Conversational agents, not our use case |
| Agent Garden samples | Pre-built ADK agents (RAG, Data Science, etc.) — interesting templates but not directly applicable to VLM training. See Section 6. |
| Vertex AI Search | Enterprise search, not relevant |
| Feature Store | We use Redis LiveKV + FAISS ArchiveKV instead |

---

## 5. Brev Role (Fallback + Scale-Up)

Brev serves as:
1. **Fallback** if Vertex AI Custom Training has queue delays or GPU unavailability
2. **Scale-up** for large GRPO runs needing 8x H100 (if Vertex doesn't have them)
3. **Quick iteration** for interactive debugging (SSH shell into GPU instance)

Brev is NOT the primary — Vertex AI is, because of credits and integrated tooling.

---

## 6. Agent Garden Assessment

The Vertex AI Agent Garden pre-built samples are ADK (Agent Development Kit) templates.
None directly apply to VLM training, but two patterns are relevant:

### RAG Agent Pattern
The RAG sample shows retrieval-augmented generation with grounding. This maps to:
- Our ArchiveKV retrieval (golden memories inform agent decisions)
- Could use Vertex AI RAG Engine to replace local FAISS in production
- Not needed now, but the pattern validates our architecture

### Data Science Agent Pattern
The Data Science sample does NL queries, predictive modeling, trend visualization.
Could be adapted for:
- Automated training run analysis ("which hyperparams correlated with highest ACT precision?")
- Dashboard generation from experiment logs
- Not a priority, but interesting for later automation

### Bottom Line
Agent Garden is for building LLM-powered conversational agents, not for training
vision-language models. The tooling we need is in **Model development > Training**
and **Colab Enterprise**, not Agent Builder.

---

## 7. KV Cache Architecture (Memory System)

```
                    ABEE MEMORY ARCHITECTURE
                    ========================

  Per-Trajectory (ephemeral)           Cross-Trajectory (persistent)
  ──────────────────────────           ────────────────────────────

  ┌──────────────────────┐            ┌──────────────────────┐
  │     LiveKV (Redis)   │            │  ArchiveKV (FAISS)   │
  │                      │            │                      │
  │  FIFO window per     │            │  Golden memories     │
  │  trajectory.         │            │  from correct ACTs.  │
  │                      │            │                      │
  │  Stores: frame       │            │  Stores: trajectory  │
  │  summaries, recent   │            │  ID, frame, agent,   │
  │  context for agents. │            │  golden rule,        │
  │                      │            │  embedding vector.   │
  │  Cleared after each  │            │                      │
  │  trajectory.         │            │  Persisted to disk.  │
  │                      │            │  Loaded on startup.  │
  │  Window size: W_i    │            │  Cosine similarity   │
  │  (dynamic, per       │            │  retrieval (top-K).  │
  │  agent).             │            │                      │
  │                      │            │  Burn-in threshold:  │
  │  Backend: Redis      │            │  50 memories before  │
  │  (Docker container). │            │  retrieval activates.│
  └──────────────────────┘            └──────────────────────┘
         │                                    │
         └─────────────┬──────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   DualCache     │
              │   (unified API) │
              │                 │
              │ .store_frame()  │
              │ .get_live_window│
              │ .retrieve_archive│
              │ .add_golden_memory│
              │ .clear_trajectory│
              └─────────────────┘

  Future (Vertex AI managed):
  ┌──────────────────────┐     ┌──────────────────────┐
  │  Vertex AI Vector    │     │  Vertex AI RAG       │
  │  Search              │     │  Engine               │
  │  (replaces FAISS)    │     │  (replaces manual    │
  │                      │     │   retrieval pipeline) │
  └──────────────────────┘     └──────────────────────┘
```

---

## 8. Network Topology

```
  LOCAL MACHINE (192.168.x.x)
  ├── Docker: abee-redis (port 6379)
  ├── Ollama: cosmos-reason2-8b (port 11434)
  ├── ABEE Dashboard (port 8050)
  └── ABEE CLI (run_abee.py)
        │
        │ HTTPS (authenticated)
        ├──→ Google Cloud (Vertex AI)
        │      ├── GCS: gs://abee-data/
        │      ├── Custom Training Jobs (A100 spot)
        │      ├── Experiments API
        │      └── Claude 3.5 Sonnet (Model Garden)
        │
        ├──→ NVIDIA Brev (fallback)
        │      └── GPU instances (H100/A100)
        │
        └──→ NVIDIA NIM API (build.nvidia.com)
               └── Cosmos-Reason2-8B inference (cloud)
```
