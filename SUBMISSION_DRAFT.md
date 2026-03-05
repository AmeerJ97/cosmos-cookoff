# NVIDIA Cosmos Cookoff Submission

## Personal Fields (FILL IN)
- First Name: ___
- Last Name: ___
- Email: ___
- Organization / University: ___
- Job Role: Developer/Engineer
- Location: ___
- Age 18+: Yes

## Team
- Team Name: ABEE
- Team Members: (your name & email)

## Project

### Code Link
https://github.com/AmeerJ97/cosmos-cookoff

### Demo Video Link (< 3 min, upload to YouTube)
https://youtu.be/FILL_ME

### Nebius Credits
No, I used my own GPUs and compute.

---

## Project Description (< 200 words)

ABEE (Adversarial Blind Epistemic Ensemble) solves the stopping-time problem in
human-robot object handoffs using NVIDIA Cosmos Reason 2.

Four blind agents run concurrently on every video frame via NIM API, each seeing
a structurally different projection through a P x T x M asymmetry matrix (4
prompt biases x 3 temporal strides x 3 sensor modalities = 36 possible
identities). No agent knows the others exist.

Agents play a Life-Points survival game: wrong ACT decisions deal near-fatal
damage, THINK decisions drain life slowly. Dead agents are replaced by a
Hyper-GRPO bandit that exploits the 36-identity space. Dynamic consensus
requires unanimity at early frames, relaxing to 66% as trajectories mature.

A dual-cache memory layer provides temporal context (Redis LiveKV, 5-30 frame
sliding window) and semantic retrieval from 722 distilled golden rules (FAISS
ArchiveKV with cosmos-embed-1.0 embeddings). A physics oracle (SAM2 + MiDaS)
can hard-veto unsafe releases before agents are even consulted.

Result: zero premature releases across 300 evaluation trajectories. The system
also continuously curates SFT training data from winning agent reasoning traces
for downstream Cosmos Reason 2 fine-tuning.

---

## Judging Criteria Alignment

- Quality of Ideas: Novel POMDP formulation with blind multi-agent consensus,
  Life-Points survival mechanics, and Hyper-GRPO identity evolution
- Technical Implementation: Full async Python system, Pydantic validation,
  Redis + FAISS dual-cache, comprehensive Mermaid architecture diagrams
- Design: Information-asymmetric agents prevent correlated failure; oracle
  hard-veto provides physics grounding; SFT exhaust enables continuous learning
- Impact: Zero premature release rate demonstrates structural safety guarantees
  for physical AI without explicit hard-coded rules
