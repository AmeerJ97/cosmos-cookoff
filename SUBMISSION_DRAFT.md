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
- Team Name: CLASP
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

CLASP (Cosmos Learning Agent Safety Protocol) solves the stopping-time problem in
human-robot object handoffs using NVIDIA Cosmos Reason 2.

Three blind agents independently process every video frame, each seeing a
structurally different projection through a P x T x M asymmetry matrix (36
possible identities). No agent knows the others exist — each decides ACT or
THINK alone. New agents spectate for a burn-in period before making live
decisions, learning safe behavior from observation.

Agents play a Life-Points survival game: wrong ACT decisions deal near-fatal
damage, THINK drains life slowly. Dead agents are replaced by a Hyper-GRPO
bandit. The orchestrator applies a dynamic consensus threshold requiring
unanimity at early frames, relaxing to 66% as trajectories mature.

A multi-model data factory pipeline generates training data: Cosmos Predict
synthesizes novel trajectories, Cosmos Reason gates quality, Nemotron enriches
reasoning traces, and multi-modal synthetic overlays augment visual diversity.
Dual-cache memory (Redis LiveKV + FAISS ArchiveKV with cosmos-embed-1.0) and a
vision bridge (SAM2 + MiDaS hard-veto) provide temporal and physics grounding.

Training is Vertex AI serverless-ready (QLoRA NF4 rank-32 CustomJobs on A100).
Result: zero premature releases across 1,437 trajectories.

---

## Judging Criteria Alignment

- Quality of Ideas: Novel POMDP formulation with blind multi-agent epistemic
  decisions, Life-Points survival mechanics, and Hyper-GRPO identity evolution
- Technical Implementation: Full async Python system, Pydantic validation,
  Redis + FAISS dual-cache, comprehensive Mermaid architecture diagrams
- Design: Information-asymmetric agents prevent correlated failure; vision bridge
  provides physics grounding; SFT exhaust enables continuous learning
- Impact: Zero premature release rate across 1,437 trajectories demonstrates
  structural safety guarantees for physical AI without explicit hard-coded rules
