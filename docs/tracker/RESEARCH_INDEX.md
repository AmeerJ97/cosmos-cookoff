# CLASP Research Index

Master index of all research documents, papers, and sources gathered for the CLASP project.

## Research Documents (docs/research/)

| Document | Size | Lines | Topics |
|----------|------|-------|--------|
| `DESIGN_PROPOSAL.md` | 32KB | 692 | Master synthesis: training pipeline, cloud infra, sensors, roadmap |
| `vertex_ai_pipeline.md` | 47KB | 1217 | Vertex AI Custom Training Jobs, KFP v2, Vizier HPO, pricing |
| `nvidia_brev_training.md` | 39KB | 855 | Brev platform, DGX Cloud Lepton, NeMo AutoModel, 3 fine-tuning paths |
| `gaussian_splatting.md` | 40KB | 686 | 3DGS for robotics, POGS handoff tracking, GaussianVLM, RTX feasibility |
| `multimodal_sensing.md` | 54KB | 780 | Thermal IR micro-cooling, WiFi CSI limits, depth cameras, sensor fusion |
| `vlm_training_practices.md` | 50KB | 921 | QLoRA config, GRPO reward design, GroupKFold, EvoKD distillation |

## Key Papers

### Gaussian Splatting for Robotics
- **POGS** (ICRA 2025, Berkeley) — 3DGS handoff tracking, 12 consecutive handoffs. Most relevant to CLASP. [arXiv:2503.05189](https://arxiv.org/abs/2503.05189)
- **RoboSplat** (RSS 2025) — One-shot manipulation via 3DGS. 87.8% vs 57.2% with 2D. [arXiv:2504.13175](https://arxiv.org/abs/2504.13175)
- **GaussianGrasper** — Language-guided grasping with 3DGS. [arXiv:2403.09637](https://arxiv.org/abs/2403.09637)
- **GaussianVLM** — 40K Gaussians → 132 VLM tokens. [arXiv:2507.00886](https://arxiv.org/abs/2507.00886)
- **SplatTalk** (ICCV 2025) — Zero-shot 3D VQA from posed RGB. [arXiv:2503.06271](https://arxiv.org/abs/2503.06271)
- **PUGS** (ICRA 2025) — Physical properties from Gaussian geometry. [arXiv:2502.12231](https://arxiv.org/abs/2502.12231)
- **Splat-MOVER** — Multi-stage open-vocabulary manipulation. [arXiv:2405.04378](https://arxiv.org/abs/2405.04378)
- **Interaction-Aware 4DGS** — Hand-object reconstruction from monocular video. [arXiv:2511.14540](https://arxiv.org/abs/2511.14540)

### VLM Training & GRPO
- **VLA-R1** — GRPO on VLA with verifiable rewards. [arXiv:2510.01623](https://arxiv.org/abs/2510.01623)
- **Cosmos-Reason2 Model Card** — [HuggingFace](https://huggingface.co/nvidia/Cosmos-Reason2-8B)
- **Physical Plausibility Cookbook** — SFT + GRPO recipe for Cosmos-Reason. [Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason1/physical-plausibility-check/post_training.html)
- **Datature Fine-tuning Tutorial** — QLoRA on 4xL40S confirmed ~$25/run. [Datature Blog](https://datature.io/blog/finetuning-your-own-cosmos-reason2-model)

### Thermal/IR Sensing
- **IR Contact Detection** — Thermal fingerprints of hand-object contact. [PMC:8883210](https://pmc.ncbi.nlm.nih.gov/articles/PMC8883210/)
- **MOTIF Hand** (USC) — Robotic hand with embedded IR camera for contactless temperature sensing.
- **ThermEval** (Feb 2026) — Benchmark confirming VLMs reason about thermal images.

### WiFi CSI Sensing
- **CSI2PC** — WiFi CSI → 3D point clouds, 2.31ms inference, 10mm RMSE. [arXiv:2410.16303](https://arxiv.org/abs/2410.16303)
- **RoboMNIST** (Nature 2025) — Multi-robot activity recognition with WiFi + video + audio. [Nature](https://www.nature.com/articles/s41597-025-04636-2)

### NVIDIA Ecosystem
- **NVIDIA Warp + 3DGS** — Physics correction at 33ms. [NVIDIA Blog](https://developer.nvidia.com/blog/building-robotic-mental-models-with-nvidia-warp-and-gaussian-splatting/)
- **Isaac Sim CosmosWriter** — Synthetic data generation with Cosmos Transfer. [Docs](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_cosmos.html)
- **NeMo RL SFT Guide** — [Docs](https://docs.nvidia.com/nemo/rl/latest/guides/sft.html)
- **Cosmos-RL GitHub** — [GitHub](https://github.com/nvidia-cosmos/cosmos-reason2/blob/main/examples/cosmos_rl/README.md)

## Official NVIDIA Resources
- [Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/index.html)
- [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/)
- [NVIDIA Brev Quickstart](https://docs.nvidia.com/brev/latest/quick-start.html)
- [NIM Documentation](https://docs.nvidia.com/nim/)
- [Cosmos Cookoff Forum](https://forums.developer.nvidia.com/t/cosmos-cookoff-final-stretch-before-submissions-close/361850)
