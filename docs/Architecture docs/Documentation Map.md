# ABEE Documentation Map

## Conventions
- **Architecture docs/** — System design, pipeline architecture, diagrams, networking, KV cache design.
  All structural "how it works" documents live here.
- **research/** — Raw research findings, literature reviews, technology assessments.
  Reference material that informed design decisions.
- **tracker/** — Living documents: training logs, price tracking, research index.
  Updated continuously as work progresses.
- **Research - ABEE/** — Earlier ABEE-specific research (pre-implementation).
- **Research - General/**, **REsearch - Vision/**, **Other ML research/** — Background research.
- **resources/** — Models, tooling inventory, external references.

## Architecture Docs

| File | Description |
|------|-------------|
| [ABEE System Documentation.md](ABEE%20System%20Documentation.md) | **Complete reference** — 8 Mermaid diagrams covering every subsystem |
| [System Architecture - Adversarial Blind Epistemic Ensemble.md](System%20Architecture%20-%20Adversarial%20Blind%20Epistemic%20Ensemble.md) | Detailed prose architecture (958 lines) |
| [Training Pipeline Architecture.md](Training%20Pipeline%20Architecture.md) | SFT → GRPO training plan |
| [PROPOSAL - Multi-Modal 3D Scene Fusion & Physics Oracle Upgrade.md](PROPOSAL%20-%20Multi-Modal%203D%20Scene%20Fusion%20%26%20Physics%20Oracle%20Upgrade.md) | Phase 2 sensor fusion proposal |

---

## Architecture Docs (docs/Architecture docs/) (docs/Architecture docs/)

| Document | Purpose | Status |
|---|---|---|
| System Architecture - ABEE.md | Master system design: POMDP, agents, scoring, oracle, memory | Current (Mar 5 2026) |
| Training Pipeline Architecture.md | End-to-end training: data flow, Vertex AI mapping, parallelization, KV cache, networking | Current (Mar 5 2026) |
| PROPOSAL - Multi-Modal 3D Scene Fusion & Physics Oracle Upgrade.md | Proposal: SHARP 3DGS, Glinty NDFs, RTX Neural Materials, IR/ultrasonic/WiFi sensor layer, oracle upgrade | DRAFT — Open for Discussion (Mar 5 2026) |
| Documentation Map.md | This file — index of all documentation | Current |

## Research (docs/research/)

| Document | Lines | Topics |
|---|---|---|
| DESIGN_PROPOSAL.md | 700+ | Master synthesis: training pipeline, cloud infra, sensors, roadmap, costs |
| vertex_ai_pipeline.md | 1217 | Vertex AI Custom Training, KFP v2, Vizier HPO, pricing, code examples |
| nvidia_brev_training.md | 855 | Brev platform, DGX Cloud, NeMo AutoModel, 3 fine-tuning paths, NIM deploy |
| gaussian_splatting.md | 686 | 3DGS for robotics: POGS, GaussianVLM, RTX 4060 Ti feasibility |
| multimodal_sensing.md | 780 | Thermal IR micro-cooling, WiFi CSI limits, depth cameras, sensor fusion |
| vlm_training_practices.md | 921 | QLoRA config, GRPO rewards, GroupKFold, EvoKD distillation, safety gates |

## Tracker (docs/tracker/)

| Document | Purpose | Updated |
|---|---|---|
| TRAINING_LOG.md | All training runs (dry-run, real inference, cloud) with metrics | Living |
| PRICING_MATRIX.md | Cloud provider comparison, cost scenarios, hardware budget | Living |
| RESEARCH_INDEX.md | All papers, sources, official NVIDIA resources | Living |
| ISAAC_SIM_ASSESSMENT.md | Isaac Sim role analysis, feasibility, integration plan | Mar 5 2026 |
