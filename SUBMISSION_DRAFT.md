# NVIDIA Cosmos Cookoff Submission

## Personal Fields
- First Name: Ameer
- Last Name: Ibrahim O.
- Email: ameer.ibrahimosman@gmail.com
- Organization / University: Vierla
- Job Role: Developer/Engineer
- Location: Canada
- Age 18+: Yes

## Team
- Team Name: J&J
- Team Members: Nada Osman

## Project

### Code Link
https://github.com/AmeerJ97/cosmos-cookoff

### Demo Video Link (< 3 min, upload to YouTube)
<!-- TODO: replace with final YouTube URL -->
https://youtu.be/PLACEHOLDER

### Nebius Credits
No, I used my own GPUs and compute.

---

## Project Description (< 200 words)

CLASP (Cosmos Learning Agent Safety Protocol) is a multi-agent stopping-time
system for human-robot handoff safety, built on NVIDIA Cosmos.

Three blind agents independently query Cosmos Reason 2 per video frame, each
receiving a structurally different projection via a P x T x M asymmetry matrix
(36 possible identities). No agent knows the others exist. Before going live,
new agents undergo spectating burn-in — observing only failure cases to learn
what kills, never what succeeds. Correct behavior must be discovered through
survival pressure alone.

A Life-Points survival game penalizes mistakes: premature ACT costs 66 points
(near-fatal), wrong ACT costs 33, while THINK drains 2 per frame. Dead agents
are instantly replaced by Hyper-GRPO, a gradient-free bandit over the 36-identity
space. The orchestrator enforces dynamic consensus — unanimous agreement at early
frames, relaxing to 66% as evidence accumulates — invisible to agents.

Cosmos Embed 1.0 powers dual-cache memory (Redis LiveKV + FAISS ArchiveKV) for
temporal context and golden-rule retrieval. SAM2 + MiDaS vision bridge provides
physics-grounded hard-veto. A Cosmos-powered data factory generates enriched SFT
training data with synthetic sensor overlays.

**Result: zero premature releases across 1,437 trajectories.**

---

## Cosmos Model Usage

| Model | Role |
|-------|------|
| `nvidia/cosmos-reason2-8b` | Primary agent reasoning — `<think>` CoT + JSON decision per frame |
| `nvidia/cosmos-predict2-14b` | Tie-breaker when agents disagree near consensus threshold |
| `nvidia/cosmos-embed-1.0` | 768-dim frame embeddings for LiveKV and ArchiveKV FAISS retrieval |
| `Qwen/Qwen3-VL-8B-Instruct` | QLoRA fine-tuning base (Cosmos Reason 2 backbone) |

---

## Judging Criteria Alignment

- **Quality of Ideas**: Novel stopping-time POMDP with blind epistemic agents,
  failure-only spectating burn-in (no success examples — agents discover correct
  behavior through survival), Life-Points mechanics, and Hyper-GRPO identity
  evolution over a 36-configuration space
- **Technical Implementation**: Full async Python orchestrator, 5 Cosmos models
  integrated (Reason, Predict, Embed + QLoRA fine-tuning), Redis + FAISS
  dual-cache memory, Pydantic-validated state, Vertex AI serverless training
- **Design**: Information-asymmetric agents prevent correlated failure; dynamic
  consensus is invisible to agents; per-agent modality masking isolates context;
  vision bridge provides physics grounding; SFT exhaust enables continuous learning
- **Impact**: Zero premature releases across 1,437 trajectories — a structural
  safety guarantee emerging from survival pressure, not hard-coded rules. The
  system preferentially fails safe (44.6% no-release) rather than risking a
  single premature drop
