# VLM Training Best Practices for Robotics
## Research Synthesis for CLASP / Cosmos-Reason2-8B

**Research Date:** March 5, 2026
**Target System:** CLASP — Adversarial Blind Epistemic Ensemble
**Model:** NVIDIA Cosmos-Reason2-8B (Qwen3-VL-Instruct base)
**Task:** Stopping-time POMDP — predicting safe human-robot handoff release windows

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [SFT Dataset Design for VLMs](#2-sft-dataset-design-for-vlms)
   - 2.1 Dataset Structure and Annotation Schema
   - 2.2 Train/Validation/Test Split Strategies for Temporal Data
   - 2.3 K-Fold Cross-Validation for Video Data
   - 2.4 Data Augmentation for Robotic Manipulation Video
   - 2.5 Minimum Data Requirements for 8B VLM SFT
3. [Fine-Tuning Strategies for Cosmos-Reason2-8B](#3-fine-tuning-strategies-for-cosmos-reason2-8b)
   - 3.1 LoRA vs QLoRA vs Full Fine-Tuning
   - 3.2 GRPO for VLM Training
   - 3.3 Hyperparameters: Learning Rate, Batch Size, Gradient Accumulation
   - 3.4 Mixed Precision and Memory Optimization
   - 3.5 VRAM Requirements Reference Table
4. [Evaluation Metrics for Handoff Prediction](#4-evaluation-metrics-for-handoff-prediction)
   - 4.1 Safety-Critical Classification Metrics
   - 4.2 Temporal Accuracy Metrics
   - 4.3 Calibration Metrics
   - 4.4 CLASP-Specific Metric Recommendations
5. [Training Infrastructure Patterns](#5-training-infrastructure-patterns)
   - 5.1 Reproducible Pipeline Architecture
   - 5.2 Experiment Tracking
   - 5.3 Checkpointing and Early Stopping
   - 5.4 Distributed Training for 8B Models
6. [Cloud Distillation](#6-cloud-distillation)
   - 6.1 Response Distillation vs Feature Distillation
   - 6.2 Distillation Pipeline for CLASP
   - 6.3 Quality Filtering and Data Curation
7. [CLASP-Specific Recommendations](#7-clasp-specific-recommendations)
8. [Sources and References](#8-sources-and-references)
9. [Methodology](#9-methodology)

---

## 1. Executive Summary

This document synthesizes best practices for training, validating, and evaluating Vision-Language Models (VLMs) for robotics applications, with specific focus on the CLASP system using NVIDIA Cosmos-Reason2-8B. Key findings:

**Dataset Design:** For an 8B VLM with a safety-critical binary decision task (THINK vs ACT), effective SFT is achievable with 500–2,000 high-quality labeled trajectory episodes. Temporal data requires episode-level (not frame-level) train/test splitting to prevent leakage. Standard k-fold is inappropriate; use expanding-window or group k-fold on episode IDs.

**Fine-Tuning:** QLoRA (4-bit, rank 32–64) is the recommended approach for the RTX 4060 Ti 16GB constraint. The official Cosmos-Reason2 post-training pipeline uses LR 2e-7, cosine decay, and per-GPU batch size 32 with gradient accumulation 4. For CLASP's hardware (16GB VRAM), per-GPU batch size must be reduced to 1–2 with gradient accumulation 16–32.

**GRPO:** Cosmos-Reason2 uses cosmos-rl, NVIDIA's GRPO implementation for Physical AI. Binary verifiable rewards (correct ACT/THINK label) can be augmented with format rewards and partial credit for timing proximity. The SFT-then-GRPO two-stage pipeline (used by the base model) is the recommended approach.

**Evaluation:** For safety-critical handoff prediction, prioritize recall over precision (false negatives — missed "safe" windows — are less catastrophic than false positives — releasing unsafely). Add temporal offset error (frame distance from ground-truth release frame) and Expected Calibration Error (ECE) to the metric suite.

**Distillation:** Claude 3.5 Sonnet (or Claude Opus 4) can generate chain-of-thought reasoning traces for unlabeled video segments, bootstrapping a synthetic SFT dataset. Use EvoKD-style adaptive generation: identify student model failure modes, then generate targeted synthetic samples for those cases.

**Infrastructure:** Use W&B for experiment tracking (superior VLM artifact management vs MLflow), checkpoint every 50 steps, implement patience-based early stopping on validation recall.

---

## 2. SFT Dataset Design for VLMs

### 2.1 Dataset Structure and Annotation Schema

VLM SFT datasets for robotics follow a **multimodal conversation format**. Each sample is a JSON object containing:

```json
{
  "id": "trajectory_042_frame_017",
  "episode_id": "ep_042",
  "frame_index": 17,
  "total_frames": 24,
  "video_path": "trajectories/ep_042/clip.mp4",
  "keyframes": ["frame_015.jpg", "frame_016.jpg", "frame_017.jpg"],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nFrame 17 of 24. The robot arm is approaching the human hand. Assess whether this frame represents a safe release window. Output THINK or ACT with reasoning."
    },
    {
      "from": "gpt",
      "value": "<think>The human hand trajectory shows deceleration over frames 14-17. Grip posture is open. Object velocity is within 0.3 m/s. No sudden motion detected by depth estimation. Safety margin is adequate.</think>\nACT"
    }
  ],
  "ground_truth_label": "ACT",
  "ground_truth_release_frame": 18,
  "annotator": "physics_oracle",
  "split": "train"
}
```

Key annotation fields for CLASP:
- **`episode_id`**: Critical for temporal split integrity — all frames from one episode must stay in one split
- **`frame_index` and `total_frames`**: Enables temporal position encoding in prompts
- **`ground_truth_release_frame`**: Required for temporal offset error metric
- **`annotator`**: Tag whether labeled by physics oracle, human annotator, or cloud distillation (Claude)

The OpenVLA paper (Kim et al., 2024) and its OFT follow-up (Kim, Finn & Liang, 2025) demonstrate that **chain-of-thought reasoning in the "gpt" turn** significantly improves performance over raw label-only SFT. The Cosmos-Reason2 model was itself post-trained on 3.7M VQA samples with explicit reasoning traces. CLASP's SFT data should include the full `<think>...</think>` reasoning chain, not just the final THINK/ACT label.

### 2.2 Train/Validation/Test Split Strategies for Temporal Data

**The fundamental constraint:** Frames within a single trajectory episode are temporally correlated. Splitting at the frame level would allow the model to see frames from the same physical handoff in both train and test sets, which constitutes temporal leakage.

**Recommended split strategy: Episode-level stratified splitting**

```
Episode IDs → Shuffle (with fixed seed) → Split:
  Train:      70%  of episodes
  Validation: 15%  of episodes
  Test:       15%  of episodes
```

Additional stratification axes to balance across splits:
- Human operator (different grip styles, approach velocities)
- Object type (weight, size, fragility)
- Lighting condition
- Trajectory difficulty (annotated by oracle safety margin)

**For production deployment:** Maintain a **held-out temporal test set** — episodes collected after training data cutoff — to evaluate temporal distribution shift. This is distinct from the random 15% test split.

**Buffer zones:** When episodes share a physical setup session, apply a temporal buffer: do not put episode N in train and episode N+1 in validation from the same day's recording session. This prevents implicit label leakage through environmental correlations (same lighting, same robot state).

The 2025 literature on time series cross-validation (Analytics Vidhya, 2026) confirms that standard random k-fold causes RMSE error gains of up to 20.5% in time series contexts due to leakage, versus below 5% for time-aware splits.

### 2.3 K-Fold Cross-Validation for Video/Trajectory Data

**Standard k-fold is inappropriate** for trajectory data because it violates the i.i.d. (independent and identically distributed) assumption. Standard shuffled k-fold will produce optimistically biased evaluation results.

**Recommended approaches:**

**Option A — Group K-Fold (preferred for CLASP)**
Use `sklearn.model_selection.GroupKFold` with `groups=episode_ids`. This ensures all frames from the same episode stay in one fold, preventing within-episode leakage. With 5 folds, each fold uses 80% of episodes for training and 20% for validation.

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=episode_ids)):
    # train_idx and val_idx are episode-level, not frame-level
    ...
```

**Option B — Expanding Window Split (for temporal generalization testing)**
Train on episodes 1 through N, validate on episodes N+1 through N+K, then expand. This tests whether the model generalizes forward in time — most relevant for deployment robustness.

**Is k-fold worth the compute cost for CLASP?**
Given that full Cosmos-Reason2-8B fine-tuning is expensive (even with QLoRA), running 5 folds is costly. Recommendation: Use a single fixed episode-level split for iterative development, and run 3-fold group cross-validation only for final model selection before the test set evaluation. The CLASP validation set serves primarily as early stopping signal, not as a hyperparameter tuning oracle.

### 2.4 Data Augmentation Strategies for Robotic Manipulation Videos

Augmentation falls into three categories for VLM training:

**Visual/Spatial Augmentations (applied to frames before tokenization):**
| Technique | Benefit | CLASP Relevance | Notes |
|-----------|---------|----------------|-------|
| Random crop + resize | Viewpoint robustness | High | Preserves relative hand position |
| Color jitter (brightness, contrast) | Lighting invariance | High | Different lab conditions |
| Horizontal flip | Handedness generalization | Medium | Flip labels for left/right hand |
| Gaussian blur | Focus robustness | Low | Can obscure finger state |
| Random erasing | Occlusion robustness | High | Object partially blocks view |
| Temporal frame jitter | FPS robustness | Medium | Sample ±1 frame from nominal |

**Temporal Augmentations:**
- **Frame subsampling:** Train at 1 FPS, 2 FPS, and 0.5 FPS variants of the same trajectory — models the uncertainty of temporal resolution
- **Clip trimming:** Start the observation window at different lead-in frames (not always frame 1)
- **Speed perturbation:** Time-stretch/compress video by ±20% using interpolation

**Generative Augmentations (emerging, 2025):**
The paper "Generative Spatiotemporal Data Augmentation" (arXiv:2512.12508, 2025) demonstrates that video diffusion models can generate realistic viewpoint and scene variations from a single trajectory. H2R (ICLR 2025) converts human-hand operation videos into robot-centric data. For CLASP, these are worth investigating when the labeled dataset is small (<200 episodes).

**Augmentations to AVOID for CLASP:**
- Any augmentation that destroys depth cues (depth information is the physics oracle's primary signal)
- Temporal reversal (backward trajectories are physically meaningless for handoff prediction)
- Aggressive color shifts that alter skin tone discrimination (relevant for hand detection)

**Text-side augmentations:**
Vary the prompt template across training samples (see the CLASP 4-prompt x 3-temporal x 3-modality asymmetry matrix). This is already encoded in the Hyper-GRPO identity design and should be reflected in the SFT dataset — each sample should use one of the prompt variants, so the model becomes robust to all of them.

### 2.5 Minimum Data Requirements for 8B VLM SFT

**General guidance for 8B models (from literature):**

| Task Type | Minimum | Practical | Large-scale |
|-----------|---------|-----------|-------------|
| Simple binary classification | 100–300 samples | 500–1,000 | 5,000+ |
| Temporal sequence prediction | 500–2,000 episodes | 2,000–10,000 | 50,000+ |
| Complex domain adaptation | 1,000–5,000 | 5,000–20,000 | 100,000+ |

The particula.tech 2026 analysis of LLM data scaling finds that "LoRA 8B still performs better than SFT 2B at adapting to the target dataset despite requiring less GPU resources" — meaning the 8B model parameter count partially compensates for limited data through stronger priors.

**For CLASP's specific task (binary stop/go prediction from video):**

The task is structurally similar to a binary classification with temporal context. Based on OpenVLA-OFT (Kim, Finn & Liang, 2025) which fine-tuned a 7B VLA from 76.5% to 97.1% success on LIBERO using task-specific data, the critical factor is **quality over quantity**.

Recommended dataset targets:
- **Minimum viable:** 200 episodes × average 20 frames = 4,000 labeled samples (enables initial training, expect ~70% accuracy)
- **Functional target:** 500–1,000 episodes × 20 frames = 10,000–20,000 labeled samples
- **High-performance target:** 2,000+ episodes covering diverse operators, conditions, and object types

**Quality factors that reduce data requirements:**
1. Chain-of-thought reasoning traces in labels (vs. bare THINK/ACT labels)
2. Balanced class ratio (aim for 40–60% ACT labels; pure THINK-heavy data starves the model of positive examples)
3. Hard negative examples: frames that look safe but are not (teaches caution)
4. Diverse episode conditions (multiple operators, lighting, object types)

The Cosmos-Reason2 base model was post-trained on 3.7M VQA samples. CLASP's domain-specific SFT operates on top of this foundation, so the base model already has strong physical common sense. The SFT layer only needs to teach the specific THINK/ACT decision grammar and handoff-specific safety reasoning.

---

## 3. Fine-Tuning Strategies for Cosmos-Reason2-8B

### 3.1 LoRA vs QLoRA vs Full Fine-Tuning

**Hardware constraint for CLASP:** RTX 4060 Ti 16GB

| Method | VRAM (8B model) | Quality | Speed | Recommendation |
|--------|-----------------|---------|-------|----------------|
| Full fine-tuning (bf16) | ~80GB | Highest | Slowest | Not feasible on 16GB |
| LoRA (bf16, r=32) | ~24–32GB | High | Fast | Not feasible on 16GB |
| QLoRA (4-bit NF4, r=32) | ~10–14GB | Good (80–90% of full FT) | Moderate | **Recommended for CLASP** |
| QLoRA (4-bit NF4, r=16) | ~8–10GB | Acceptable | Fastest | Minimum viable |

**QLoRA configuration for CLASP (RTX 4060 Ti 16GB):**

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=32,                          # Rank — higher = more capacity, more VRAM
    lora_alpha=64,                 # Scaling factor (2x rank is standard)
    target_modules=[               # Qwen3-VL attention + MLP layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

**Key finding from literature:** OpenVLA (Kim et al., 2024) demonstrated that LoRA with r=32 "matches full fine-tuning performance while training only 1.4% of the model parameters." The arXiv:2512.11921 paper (December 2025) shows LoRA rank=8 with 4-bit quantization enables training on an RTX 4060 8GB — meaning rank=16 or 32 should be achievable on 16GB.

**Gradient checkpointing** is mandatory at 16GB VRAM. Enable via `model.gradient_checkpointing_enable()`. This trades ~20% compute speed for ~30% VRAM reduction.

**Recommended tooling:**
- **Unsloth** — fastest QLoRA implementation, natively supports Qwen3-VL, achieves 2–4x speedup over HuggingFace PEFT alone
- **LLaMA-Factory** — high-level training wrapper with built-in VLM support and W&B integration
- **TRL (HuggingFace)** — preferred for GRPO training via `GRPOTrainer`

### 3.2 GRPO for VLM Training

**What GRPO does:** Rather than requiring a separate critic model (as PPO does), GRPO samples G responses from the current policy for each input prompt, then computes the advantage for each response as its normalized deviation from the group mean reward. This eliminates the critic model, halving VRAM requirements compared to PPO.

**GRPO training objective:**
```
For each prompt x, sample G outputs {o_1, ..., o_G}
Compute rewards {r_1, ..., r_G}
Advantage: A_i = (r_i - mean(r)) / std(r)
Policy gradient update with KL penalty against reference policy
```

**Two-stage pipeline (standard for Cosmos-Reason2):**

```
Stage 1: SFT on curated labeled dataset
  → Teaches the model correct answer format and domain vocabulary
  → Cold-starts the reasoning capability

Stage 2: GRPO with verifiable rewards
  → Fine-tunes reasoning quality and calibration
  → SFT-trained model serves as the reference policy for KL penalty
```

The NVIDIA Cosmos-Reason2 technical blog confirms: "Fine-tuning on physical AI tasks boosts Cosmos Reason's base model performance by over 10%, with reinforcement learning adding another 5% gain."

**Reward function design for CLASP:**

Binary rewards alone cause reward sparsity — groups where all G samples get the same reward produce zero advantage, wasting compute. The 2025 literature on GRPO reward design recommends a composite reward:

```python
def clasp_reward(response: str, ground_truth: str, frame_idx: int,
                release_frame: int) -> float:
    reward = 0.0

    # Primary correctness reward (binary verifiable)
    predicted_action = parse_action(response)  # "ACT" or "THINK"
    if predicted_action == ground_truth:
        reward += 1.0

    # Format reward (ensures parseable output)
    if has_valid_think_tags(response) and has_clear_action(response):
        reward += 0.1

    # Temporal proximity reward (partial credit for near-misses)
    # Only applied when prediction is wrong
    if predicted_action != ground_truth and ground_truth == "ACT":
        temporal_distance = abs(frame_idx - release_frame)
        if temporal_distance <= 2:
            reward += 0.3  # Partial credit for being close to release window

    # Safety penalty (asymmetric: false ACT is worse than false THINK)
    if predicted_action == "ACT" and ground_truth == "THINK":
        # Premature release — dangerous
        # Apply additional penalty proportional to how early
        frames_too_early = release_frame - frame_idx
        if frames_too_early > 5:
            reward -= 0.5  # Very early false positive — penalize harder

    return reward
```

**Key design principle:** The asymmetric penalty aligns the GRPO reward with the CLASP Life-Points system — false ACT early in a trajectory is more dangerous than false THINK.

**Practical GRPO setup with TRL:**
```python
from trl import GRPOConfig, GRPOTrainer

grpo_config = GRPOConfig(
    num_generations=8,           # G — number of samples per prompt
    max_new_tokens=512,
    temperature=0.7,
    learning_rate=5e-7,          # Lower than SFT LR
    kl_coeff=0.1,               # KL penalty weight
    cliprange=0.2,               # PPO-style clipping
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
)
```

**VLA-R1 paper (Ye et al., 2025, arXiv:2510.01623)** is directly relevant — they apply GRPO to a Vision-Language-Action model with verifiable rewards for trajectory consistency and output formatting, achieving strong generalization on both in-domain and out-of-domain robot platforms.

**TON (Wang et al., 2025, arXiv:2505.16854)** introduces "thought dropout" during SFT as a cold-start for selective reasoning — randomly replace reasoning traces with empty thoughts during Stage 1. This teaches the model to choose when deep reasoning is worth the token cost. For CLASP's THINK decision (which requires reasoning) vs ACT (which requires confidence), this is directly applicable.

### 3.3 Hyperparameters: Learning Rate, Batch Size, Gradient Accumulation

**Official Cosmos-Reason2 post-training hyperparameters** (from the AV captioning cookbook recipe):

| Parameter | Official Value | CLASP Adaptation (16GB) |
|-----------|---------------|------------------------|
| Learning rate | 2e-7 | 1e-6 to 2e-6 (QLoRA adapters train faster) |
| LR schedule | Cosine annealing | Cosine annealing |
| Warmup | 3% of total steps | 5–10% (smaller dataset needs more warmup) |
| Weight decay | 0.01 | 0.01 |
| Gradient clipping | 1.0 | 1.0 |
| Per-GPU batch size | 32 | 1–2 (VRAM constraint) |
| Gradient accumulation | 4 | 16–32 (compensate for small batch) |
| Effective batch size | 256 (8 GPUs × 32) | 16–32 (1 GPU × 1–2 × 16–32) |
| Epochs | 3 | 3–5 (smaller dataset needs more passes) |
| Max sequence length | 4,096 tokens | 4,096 tokens |

**Learning rate guidance:**
- QLoRA adapter LR should be 1–2 orders of magnitude higher than full fine-tuning LR because only adapter parameters are updated
- Start with 1e-5 for LoRA adapters on Qwen3-VL architecture and decay to 1e-7
- Monitor validation loss — if it rises after 1 epoch, LR is too high

**Batch size guidance:**
- Effective batch size (batch_size × grad_accum) should be at least 32 for stable gradients
- For CLASP at 16GB: 2 (batch) × 16 (accum) = 32 effective samples minimum
- Frame-packing (combining multiple short context windows into one batch item) can increase throughput

### 3.4 Mixed Precision and Memory Optimization

**bf16 (BFloat16) vs fp16:**
- **Use bf16** for Cosmos-Reason2 (Qwen3-VL base). bf16 has wider dynamic range than fp16, reducing gradient underflow on large models. The official pipeline specifies bf16.
- fp16 requires loss scaling to prevent underflow; bf16 does not. At 16GB VRAM on consumer hardware, bf16 is preferred.

**Memory optimization stack for CLASP (in order of impact):**

1. **4-bit NF4 quantization** (QLoRA) — reduces model size from ~16GB to ~4GB
2. **Gradient checkpointing** — trades 20% compute for ~30% activation memory reduction
3. **Paged AdamW** optimizer — offloads optimizer states to CPU when GPU memory spikes
4. **Flash Attention 2** — reduces attention memory from O(n²) to O(n), enables longer sequences
5. **Activation offloading** — moves unused activations to CPU RAM (96GB available on CLASP hardware)

**Recommended optimizer:** `paged_adamw_8bit` from bitsandbytes for QLoRA training. Reduces optimizer state VRAM by 75% while maintaining numerical stability.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    bf16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
)
```

### 3.5 VRAM Requirements Reference Table

| Configuration | VRAM Required | Feasible on 4060 Ti 16GB |
|---------------|---------------|--------------------------|
| Full inference (bf16) | ~16GB | Barely (batch=1) |
| Full inference (4-bit) | ~5GB | Yes (batch=8+) |
| QLoRA SFT r=16 (4-bit) | ~10–12GB | Yes (batch=1–2) |
| QLoRA SFT r=32 (4-bit) | ~12–14GB | Yes (batch=1) |
| QLoRA SFT r=64 (4-bit) | ~14–16GB | Marginal (may OOM) |
| GRPO (G=8, QLoRA r=32) | ~14–16GB | Marginal — reduce G to 4 |
| Full SFT (bf16) | ~80GB+ | No |

**Note:** The NVIDIA official docs state the 8B model requires "at least 32GB of GPU memory for training" — this refers to full fine-tuning in bf16. QLoRA with 4-bit quantization brings this into 16GB range. Datature's 2025 fine-tuning guide confirms QLoRA feasibility for Cosmos-Reason2-8B on consumer hardware.

---

## 4. Evaluation Metrics for Handoff Prediction

### 4.1 Safety-Critical Classification Metrics

For CLASP's binary ACT/THINK prediction, raw accuracy is insufficient. The full metric suite:

**Confusion Matrix Breakdown:**
```
               Predicted THINK  |  Predicted ACT
Actual THINK:      TN           |      FP  (premature release — DANGEROUS)
Actual ACT:        FN           |      TP  (correct release)
```

**Asymmetric risk analysis:**
- **FP (premature ACT):** Robot releases object when human is not ready — injury risk. This is the catastrophic failure mode.
- **FN (late THINK / missed release window):** Robot waits when it could have released — efficiency loss, no injury risk.

**Consequence:** The system should be tuned to **minimize FP rate** (maximize specificity/precision for ACT class). A conservative model that is biased toward THINK is safer.

**Primary metrics:**

| Metric | Formula | Target | CLASP Priority |
|--------|---------|--------|---------------|
| **ACT Precision** | TP / (TP + FP) | > 0.95 | Critical — measures premature release rate |
| **ACT Recall** | TP / (TP + FN) | > 0.80 | Important — measures window utilization |
| **F-beta (β=0.5)** | Precision-weighted F | > 0.90 | Balances with precision emphasis |
| **Specificity** | TN / (TN + FP) | > 0.97 | Safety-critical — false ACT rate |
| **Balanced Accuracy** | (Sensitivity + Specificity) / 2 | > 0.85 | Overall performance |
| **MCC** | Matthews Correlation Coefficient | > 0.70 | Robust to class imbalance |

**Why F-beta with β=0.5 (not F1)?**
F-beta with β < 1 weights precision more heavily than recall. Given that premature release is dangerous, we prioritize precision. F0.5 is `(1 + 0.25) * precision * recall / (0.25 * precision + recall)`.

**Calibrated Safe Prediction (from arXiv:2508.09346, 2025):** Recent work specifically addresses calibration guarantees for image-controlled autonomous systems, noting that distribution shift over prediction horizons causes safety evaluators to produce overconfident outputs. This is directly applicable to CLASP — the model should know when it doesn't know.

### 4.2 Temporal Accuracy Metrics

Because CLASP is a **stopping-time prediction** problem, correctness is not binary — it matters **when** the model correctly predicts ACT relative to the ground-truth release frame.

**Temporal Offset Error (TOE):**
```
TOE = predicted_ACT_frame - ground_truth_release_frame

TOE < 0: Premature release (dangerous — negative)
TOE = 0: Perfect timing
TOE > 0: Late release (safe but inefficient — positive)
```

**Temporal metrics:**
| Metric | Formula | Notes |
|--------|---------|-------|
| **Mean TOE** | mean(TOE) | Should be positive (systematic caution bias is acceptable) |
| **Median TOE** | median(TOE) | Robust to outlier late releases |
| **TOE within ±2 frames** | % of samples | Acceptable timing window |
| **Premature rate** | % where TOE < 0 | Must be minimized — safety critical |
| **Mean early error** | mean(TOE | TOE < 0) | Severity of premature releases |

**Episode-level temporal analysis:**
For each test episode, compute the full ACT probability curve across all frames and measure:
- **First ACT frame:** When does the model first predict ACT?
- **Confidence ramp:** How quickly does ACT confidence rise toward the release window?
- **False alarm rate:** Number of false ACT predictions before the release window

### 4.3 Calibration Metrics

**Why calibration matters for CLASP:** The agent outputs THINK or ACT, but the underlying model produces a probability. If the model says "90% confident this is ACT" but is actually right only 60% of the time at that confidence level, it is miscalibrated — and a dangerous partner for a safety-critical system.

**Expected Calibration Error (ECE):**
```
ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

Where B_b is the set of samples in confidence bin b,
acc is the bin's accuracy, conf is the bin's mean confidence
```

Lower ECE = better calibrated. A perfectly calibrated model has ECE = 0.

**ICLR 2025 calibration review** (arXiv:2501.19047) notes that standard ECE has known weaknesses (discontinuous, bin-sensitive). Use **Smooth ECE** (ICLR 2024, proceedings) or **adaptive binning ECE** for more reliable estimates.

**Calibration tools:**
- `torchmetrics.classification.CalibrationError` (TorchMetrics)
- Temperature scaling post-hoc calibration (apply after training)
- `sklearn.calibration.calibration_curve` for reliability diagrams

**Reliability diagram:** Plot mean predicted probability (x-axis) vs observed accuracy (y-axis) per bin. A perfectly calibrated model produces a diagonal line. Deviations above the diagonal = overconfident; below = underconfident.

**For CLASP:** A slightly underconfident model (biased toward THINK) is preferable to an overconfident one (biased toward ACT). Consider applying **temperature scaling** post-training with T > 1 to reduce overconfidence.

**Additional calibration metrics:**
- **Maximum Calibration Error (MCE):** max over bins of |acc - conf| — captures worst-case miscalibration
- **Brier Score:** mean((p_predicted - y_actual)²) — proper scoring rule that jointly measures accuracy and calibration

### 4.4 CLASP-Specific Metric Recommendations

**Primary evaluation dashboard (in priority order):**

1. ACT Precision (safety gate — must be > 0.95 to proceed to deployment)
2. Premature Release Rate (% of TOE < 0)
3. Balanced Accuracy or MCC
4. Expected Calibration Error (ECE)
5. ACT Recall (efficiency metric)
6. Mean TOE (bias measurement)
7. F-beta (β=0.5) score

**Per-agent metrics:** Because CLASP has 4 agents with different asymmetry parameters, evaluate each agent's metric suite independently. An agent with SAFETY-FIRST temporal weighting should have near-zero premature release rate at the cost of higher FN rate.

**Ensemble metrics:** Track the consensus agreement rate and the correlation between individual agent confidence and ensemble correctness.

---

## 5. Training Infrastructure Patterns

### 5.1 Reproducible Pipeline Architecture

**Minimum reproducibility requirements:**

```
project/
├── configs/
│   ├── sft_config.yaml          # Full hyperparameter specification
│   └── grpo_config.yaml
├── data/
│   ├── splits/
│   │   ├── train_episodes.txt   # Episode IDs only — no raw data
│   │   ├── val_episodes.txt
│   │   └── test_episodes.txt
│   └── preprocessing/
│       └── augmentation_config.yaml
├── training/
│   ├── sft_train.py
│   ├── grpo_train.py
│   └── evaluate.py
├── checkpoints/
│   └── run_{timestamp}/
│       ├── checkpoint-{step}/
│       └── best_val_recall/
└── experiments/
    └── {run_id}/
        ├── config_snapshot.yaml  # Exact config used for this run
        ├── git_hash.txt
        └── metrics.jsonl
```

**Seed management:**
```python
import torch, random, numpy as np

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For Qwen3-VL tokenizer sampling
    transformers.set_seed(seed)
```

**Environment pinning:** Use `pip freeze > requirements.txt` and commit with each experiment. Cosmos-Reason2 is based on Qwen3-VL — pin the exact `transformers` version as Qwen3 architecture changes have historically broken fine-tuned checkpoint compatibility.

### 5.2 Experiment Tracking

**W&B (Weights & Biases) — recommended for CLASP:**
- Superior artifact management for large model checkpoints vs MLflow
- Native VLM image logging (log sample predictions with ground truth during validation)
- Built-in sweep functionality for hyperparameter search
- Free tier sufficient for solo projects

**Key metrics to log per training step:**
- Training loss (total, language model head only)
- Learning rate (track warmup and decay)
- Gradient norm (diagnose exploding gradients)
- Per-class validation metrics (precision, recall per ACT/THINK)
- Sample validation predictions (log 10–20 example images with model output and ground truth)
- VRAM usage (torch.cuda.memory_allocated())

**MLflow alternative:** More appropriate if integrating with Vertex AI or MLOps pipelines. Use MLflow 2.10+ for enhanced checkpoint management (import_checkpoints API). The March 2026 MLflow mastery guide confirms strong support for large model artifact tracking.

**Logging example:**
```python
import wandb

wandb.init(
    project="clasp-cosmos-reason2",
    config={
        "model": "cosmos-reason2-8b",
        "lora_rank": 32,
        "learning_rate": 1e-5,
        "effective_batch_size": 32,
        "dataset_size": len(train_dataset),
        "git_hash": get_git_hash(),
    }
)

# Log per-step
wandb.log({
    "train/loss": loss.item(),
    "train/lr": scheduler.get_last_lr()[0],
    "train/grad_norm": grad_norm,
    "train/vram_gb": torch.cuda.memory_allocated() / 1e9,
})

# Log validation predictions as W&B Table
predictions_table = wandb.Table(
    columns=["frame_id", "image", "ground_truth", "prediction", "correct"]
)
```

### 5.3 Checkpointing and Early Stopping

**Checkpoint strategy:**

```python
# Save every N steps
save_steps = 50  # More frequent for small dataset where epochs are short

# Save best-by-metric checkpoints separately
best_val_precision = 0.0
if val_precision > best_val_precision:
    model.save_pretrained(f"checkpoints/best_precision/")
    best_val_precision = val_precision

# Save best-by-safety (prioritize ACT precision over overall metrics)
best_premature_rate = 1.0
if premature_rate < best_premature_rate:
    model.save_pretrained(f"checkpoints/best_safety/")
    best_premature_rate = premature_rate
```

**Early stopping for CLASP:**

Standard early stopping on validation loss is not optimal for safety-critical systems. Use patience-based stopping on **ACT precision**:

```python
class SafetyAwareEarlyStopping:
    def __init__(self, patience=5, min_delta=0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_precision = 0.0

    def __call__(self, val_act_precision: float) -> bool:
        if val_act_precision > self.best_precision + self.min_delta:
            self.best_precision = val_act_precision
            self.counter = 0
            return False  # Continue training
        else:
            self.counter += 1
            return self.counter >= self.patience  # Stop
```

**Checkpoint selection at deployment:** Do NOT use the final checkpoint. Use the checkpoint with the best validation ACT precision (safety metric), not best overall loss or accuracy.

The Official Cosmos-Reason2 cookbook saves every 20 steps. For CLASP with smaller datasets, save every 50 steps with the best-metric checkpoint tracked separately.

### 5.4 Distributed Training Considerations for 8B Models

**For CLASP's single-GPU 16GB constraint:** Full distributed training is not feasible. However, CPU offloading via DeepSpeed ZeRO-3 or FSDP can leverage CLASP's 96GB DDR5 RAM:

```python
# DeepSpeed ZeRO-Infinity config for CPU offloading
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": true},
        "offload_param": {"device": "cpu", "pin_memory": true},
    },
    "bf16": {"enabled": true}
}
```

ZeRO-3 with CPU offload can reduce GPU VRAM usage to ~4–6GB for 8B model parameters at the cost of 3–5x slower training (CPU-GPU data transfer bottleneck). For CLASP's 16GB GPU, this enables slightly larger batch sizes but is likely slower than QLoRA which avoids CPU offload entirely.

**Recommended for CLASP:** Stick with QLoRA on the single RTX 4060 Ti 16GB. If training time becomes a bottleneck, consider renting a cloud GPU (A100 40GB or H100 80GB) for the GRPO stage which is more memory-intensive.

---

## 6. Cloud Distillation

### 6.1 Response Distillation vs Feature Distillation

**Two distinct approaches:**

**Response Distillation (Black-Box):**
The teacher model (Claude 3.5 Sonnet / Opus 4) receives video frames and a prompt, then generates full reasoning traces + labels. These teacher outputs become the student (Cosmos-Reason2-8B) SFT labels. No access to teacher model internals required.

**Advantages for CLASP:**
- Works with API-only teacher models (Claude via Anthropic API)
- Generates rich CoT reasoning traces that teach the student how to reason
- Can target specific failure modes (see EvoKD below)

**Feature Distillation (White-Box):**
Match intermediate layer activations between teacher and student. Requires access to teacher model weights. Not applicable when teacher is accessed via API.

**Recommendation for CLASP:** Use **response distillation** with Claude as teacher. This is the same approach used in Cosmos-Reason2's own post-training — the model's 24.7K video VQA samples had "reasoning traces distilled from DeepSeek-R1."

### 6.2 Distillation Pipeline for CLASP

**Pipeline architecture:**

```
Phase 1: Seed Dataset
├── Collect raw video trajectories (unlabeled or lightly labeled)
├── Run physics oracle (SAM2 + MiDaS) to get hard safety labels
└── Store: {video, oracle_label, no_reasoning_trace}

Phase 2: Teacher Annotation (Claude API)
├── For each video segment:
│   ├── Extract 3 keyframes at temporal positions [t-2, t-1, t]
│   ├── Send to Claude with prompt:
│   │   "You are analyzing a human-robot handoff. The robot arm is transferring
│   │    an object to the human. Based on these frames, provide:
│   │    1. <think> detailed safety analysis of hand position, approach velocity,
│   │       grip readiness, and any risk factors </think>
│   │    2. Final judgment: ACT (safe to release) or THINK (wait)"
│   └── Store: {video, claude_label, claude_reasoning_trace}
├── Filter: Keep samples where claude_label == oracle_label (agreement filter)
└── Output: synthetic_sft_dataset.jsonl

Phase 3: Student SFT on Synthetic Data
├── Initial fine-tuning on filtered synthetic dataset
├── Evaluate on held-out validation set
└── Identify failure mode categories

Phase 4: EvoKD Adaptive Generation (Liu et al., 2024)
├── For each failure mode category:
│   ├── Generate targeted prompts asking Claude for harder examples
│   └── Store: {video_description, claude_label, claude_reasoning_trace}
├── Mix targeted synthetic samples with original dataset
└── Re-train student
```

**Claude prompt template for handoff annotation:**
```
You are a physical safety expert analyzing a human-robot object handoff.

Context: The robot arm is approaching the human to transfer [OBJECT_DESCRIPTION].
Frame sequence: [KEYFRAMES]
Frame index: [FRAME_N] of estimated [TOTAL_FRAMES] total frames.
Prior frames summary: [BRIEF_PRIOR_CONTEXT]

Analyze the following safety dimensions:
1. Human hand readiness (position, grip posture, gaze direction)
2. Object velocity and trajectory stability
3. Spatial clearance and collision risk
4. Anomaly detection (sudden motion, environmental hazard)

Provide your analysis in this format:
<think>
[Detailed multi-factor safety analysis]
</think>
[ACT if safe to release now, THINK if robot should wait]

Ground truth physics oracle says: [ORACLE_LABEL]
If you disagree, explain why in one sentence.
```

**Including the oracle label in the prompt** serves two purposes: it grounds the teacher's reasoning in physical ground truth, and it causes the teacher to generate higher-quality reasoning traces that reconcile its visual assessment with the physics-based label.

### 6.3 Quality Filtering and Data Curation

**Agreement filter:** Retain only samples where the teacher label matches the oracle label. In the Cosmos-Reason2 base model training, human-annotated and auto-labeled samples were weighted 50%/30% respectively — quality labeling is worth more than volume.

**Reasoning quality filter:**
```python
def is_high_quality_trace(trace: str) -> bool:
    # Must reference at least 2 safety dimensions
    safety_keywords = ["velocity", "grip", "position", "hand", "clearance",
                       "distance", "approach", "stable", "motion"]
    keyword_count = sum(1 for kw in safety_keywords if kw in trace.lower())

    # Must be substantive (not a one-liner)
    word_count = len(trace.split())

    return keyword_count >= 2 and word_count >= 30
```

**Dataset weight mixing:** The official Cosmos-Reason2 pipeline uses `[0.3, 0.5, 0.2]` weights for cv_annotated, human_annotated, and reasoning_sft respectively. For CLASP:
- Human-verified hard cases: weight 0.5
- Claude-distilled synthetic samples: weight 0.3
- Physics-oracle-only (no reasoning trace): weight 0.2

**Contamination check:** Before using distilled samples in training, ensure no visual scene overlap with the test set (same room, same operator). Use perceptual hashing on keyframes.

---

## 7. CLASP-Specific Recommendations

### Immediate Priorities (P0 alignment with current implementation status)

**1. SFT Dataset Schema**
Implement the annotation schema from Section 2.1 immediately. Key additions beyond current `clasp_pkg/local_inference.py` output:
- Add `ground_truth_release_frame` field to all samples
- Add `annotator` provenance field
- Store full `<think>` reasoning traces, not just final labels

**2. Episode-level data splitting**
Current `clasp_pkg` has data_loader — ensure it splits by `episode_id` not by frame. Temporal leakage will cause falsely optimistic validation metrics.

**3. QLoRA training script**
For the RTX 4060 Ti 16GB:
```
Model: nvidia/Cosmos-Reason2-8B (4-bit NF4)
LoRA rank: 32
LoRA target: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
LR: 1e-5 (LoRA adapters) with cosine decay to 1e-7
Effective batch size: 32 (batch=2, accum=16)
Epochs: 3-5
Optimizer: paged_adamw_8bit
Precision: bf16 (adapter weights)
Flash Attention 2: enabled
```

**4. Safety-first evaluation**
Primary early stopping criterion: validation ACT Precision > 0.95. Never deploy a checkpoint where premature release rate exceeds 5%.

### Integration with CLASP's Hyper-GRPO System

The 36-identity asymmetry matrix (4 prompts × 3 temporal × 3 modality) creates natural **multi-prompt diversity** in the SFT dataset. Include samples from all 36 identity variants to ensure the SFT model generalizes across prompt styles before GRPO training begins.

During GRPO: the Life-Points system provides natural reward signal structure. The 33-point penalty for wrong ACT maps to a -0.33 reward signal; the 66-point catastrophic early penalty maps to -0.66 (or -1.0 for simplicity). This directly encodes the asymmetric risk into the GRPO reward function.

### Cosmos-Reason2 Specific Notes

- The model is based on **Qwen3-VL-Instruct** — all Qwen3-VL fine-tuning recipes apply
- The cosmos-rl repository (github.com/nvidia-cosmos/cosmos-reason2) provides the official GRPO implementation
- Maximum pixels per frame is set to 360,000 (~600×600) in the official pipeline; CLASP can use up to 81,920 per the dataset config, which reduces VRAM at the cost of visual detail
- Frame rate of 1 FPS is used in the official pipeline — CLASP's 30FPS video should be subsampled to 1–2 FPS for training to match the pretraining distribution

---

## 8. Sources and References

### Academic Papers

- **OpenVLA** — Kim et al. (2024). "OpenVLA: An Open-Source Vision-Language-Action Model." arXiv:2406.09246. [1,134 citations] https://consensus.app/papers/openvla-an-opensource-visionlanguageaction-model-kim-pertsch/
- **OpenVLA-OFT** — Kim, Finn & Liang (2025). "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success." arXiv:2502.19645. [162 citations] https://openvla-oft.github.io/
- **RoboFlamingo** — Li et al. (2023). "Vision-Language Foundation Models as Effective Robot Imitators." arXiv:2311.01378. [272 citations] https://consensus.app/papers/visionlanguage-foundation-models-as-effective-robot-li-liu/
- **VLA-R1** — Ye et al. (2025). "VLA-R1: Enhancing Reasoning in Vision-Language-Action Models." arXiv:2510.01623. https://github.com/GigaAI-research/VLA-R1
- **TON (Think or Not)** — Wang et al. (2025). "Think or Not? Selective Reasoning via Reinforcement Learning for Vision-Language Models." arXiv:2505.16854. https://github.com/kokolerk/TON
- **Turn-Level Reward Design** — Wei et al. (2025). "Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Reward Design." IJCNN 2025.
- **GRPO Dynamics** — (2025). "Reinforcement Learning with Verifiable Rewards: GRPO's Effective Loss, Dynamics, and Success Amplification." arXiv:2503.06639. https://arxiv.org/abs/2503.06639
- **Smooth ECE** — ICLR 2024 proceedings. "Smooth ECE: Principled Reliability Diagrams." https://proceedings.iclr.cc/paper_files/paper/2024/file/06cf4bae7ccb6ea37b968a394edc2e33-Paper-Conference.pdf
- **ECE Introduction** — (2025). "Understanding Model Calibration." arXiv:2501.19047. ICLR Blogposts 2025. https://iclr-blogposts.github.io/2025/blog/calibration/
- **Safe Autonomous Prediction** — (2025). "How Safe Will I Be Given What I Saw? Calibrated Prediction of Safety Chances for Image-Controlled Autonomy." arXiv:2508.09346
- **Generative Spatiotemporal Augmentation** — (2025). arXiv:2512.12508. https://arxiv.org/abs/2512.12508
- **H2R Augmentation** — ICLR 2025. "H2R: A Human-to-Robot Data Augmentation for Robot Pre-training from Videos." https://openreview.net/forum?id=meY9nInitM
- **LoRA Consumer GPU** — (2025). "Towards Accessible Physical AI: LoRA-Based Fine-Tuning of VLA Models for Real-World Robot Control." arXiv:2512.11921
- **LoRA/QLoRA Profiling** — (2025). "Profiling LoRA/QLoRA Fine-Tuning Efficiency on Consumer GPUs: An RTX 4060 Case Study." arXiv:2509.12229
- **EvoKD** — Liu et al. (2024). "Evolving Knowledge Distillation with Large Language Models and Active Learning." EMNLP 2024.
- **KD Survey** — (2025). "Knowledge Distillation and Dataset Distillation of Large Language Models: Emerging Trends, Challenges, and Future Directions." arXiv:2504.14772. https://arxiv.org/abs/2504.14772
- **VL2Lite** — Jang et al. (2025). "VL2Lite: Task-Specific Knowledge Distillation from Large Vision-Language Models to Lightweight Models." CVPR 2025.
- **Embodied AI Survey** — Yifan et al. (2025). "Embodied AI: A Survey on the Evolution from Perceptive to Behavioral Intelligence." SmartBot 1(3). https://doi.org/10.1002/smb2.70003
- **Robot Feedback** — Biyik (2026). "Training robots with natural and lightweight human feedback." AI Magazine 47(1). https://doi.org/10.1002/aaai.70037
- **GRPO Illustrated** — Pichka (2025). https://epichka.com/blog/2025/grpo/
- **GRPO Cameron Wolfe** — Wolfe (2025). "Group Relative Policy Optimization." https://cameronrwolfe.substack.com/p/grpo
- **DeepSeekMath / GRPO Origin** — (2024). arXiv:2402.03300. https://arxiv.org/abs/2402.03300

### Technical Documentation

- **Cosmos-Reason2 GitHub** — https://github.com/nvidia-cosmos/cosmos-reason2
- **Cosmos Cookbook** — https://nvidia-cosmos.github.io/cosmos-cookbook/
- **Cosmos-Reason2 Post-Training Recipe** — https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason2/video_caption_vqa/post_training.html
- **NVIDIA Blog: Cosmos Reason Post-Training** — https://developer.nvidia.com/blog/maximize-robotics-performance-by-post-training-nvidia-cosmos-reason/
- **Datature: Finetuning Cosmos-Reason2** — https://datature.io/blog/finetuning-your-own-cosmos-reason2-model
- **GRPO Trainer (TRL)** — https://huggingface.co/docs/trl/en/grpo_trainer
- **HuggingFace VLM Fine-Tuning Guide** — https://www.philschmid.de/fine-tune-multimodal-llms-with-trl
- **HuggingFace Smol Course: Fine-Tuning VLMs** — https://huggingface.co/learn/smol-course/en/unit4/3
- **Unsloth RL Guide** — https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide

### Analysis and Tutorials

- **VRAM for Fine-Tuning** — https://modal.com/blog/how-much-vram-need-fine-tuning
- **LoRA vs QLoRA** — https://modal.com/blog/lora-qlora
- **LoRA Fine-Tuning Infrastructure** — https://introl.com/blog/fine-tuning-infrastructure-lora-qlora-peft-scale-guide-2025
- **Data Requirements for LLM Fine-Tuning** — https://particula.tech/blog/how-much-data-fine-tune-llm
- **Time Series Cross-Validation** — https://www.analyticsvidhya.com/blog/2026/03/time-series-cross-validation/
- **Temporal Data Leakage** — https://hectorv.com/2023/07/06/data-leakage-in-time-series-data-cross-validations-in-machine-learning/
- **LSTM Temporal Leakage Study** — https://arxiv.org/html/2512.06932v1
- **MLflow Experiment Tracking (2026)** — https://dasroot.net/posts/2026/02/ml-model-versioning-experiment-tracking-mlflow/
- **W&B vs MLflow vs ZenML** — https://www.zenml.io/blog/mlflow-vs-weights-and-biases

---

## 9. Methodology

**Research conducted:** March 5, 2026

**Sources consulted:**
1. **Brave Web Search** — 12 targeted queries across all 5 research areas
2. **Consensus Academic Search API** — 4 queries on VLA fine-tuning, GRPO, and knowledge distillation (200M+ papers database)
3. **Scholar Gateway Semantic Search** — 1 query on VLM training for robotic manipulation (Wiley/peer-reviewed literature)
4. **Direct URL fetch** — NVIDIA developer blog on Cosmos-Reason2 post-training and official cookbook post-training recipe
5. **Claude Sonnet 4.6 internal knowledge** — synthesized with external sources; flagged where internal knowledge fills gaps not covered by search results

**Research approach:**
- Phase 1: Parallel web searches on all 5 topic areas simultaneously
- Phase 2: Academic database queries for peer-reviewed support
- Phase 3: Direct technical documentation retrieval for Cosmos-Reason2 specific details
- Phase 4: Synthesis and cross-referencing of findings, resolving conflicts where sources disagreed

**Areas of high confidence (multiple concordant sources):**
- QLoRA feasibility for 8B VLMs at 16GB VRAM
- GRPO two-stage (SFT then RL) pipeline superiority
- Episode-level splitting necessity for temporal data
- ECE as calibration standard for safety-critical systems

**Areas of uncertainty / insufficient data:**
- Exact VRAM consumption of Cosmos-Reason2-8B in QLoRA training mode (varies by sequence length, frame resolution)
- Optimal LoRA rank for this specific task (empirical tuning required; r=32 is a well-supported starting point)
- Minimum viable dataset size for CLASP's specific stopping-time task (robotics literature suggests 500+ episodes; the exact number requires empirical validation)
- Claude API throughput and cost for large-scale distillation (rate limits not characterized for batch annotation jobs)

**Conflicts noted:**
- Official NVIDIA documentation states 32GB minimum VRAM for 8B training (referring to full fine-tuning); consumer GPU literature (arXiv:2509.12229, Datature blog) confirms 4-bit QLoRA brings this within 16GB range
- General LLM data requirements literature suggests 100–300 samples suffice for simple tasks; robotics-specific sources emphasize that embodied AI is "far more data hungry" — both are correct at different task complexities
