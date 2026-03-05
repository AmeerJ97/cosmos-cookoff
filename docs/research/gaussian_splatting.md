# Gaussian Splatting for Robotic Manipulation and Human-Robot Handoff
## Research Synthesis for the ABEE Project

**Date:** 2026-03-05
**Author:** Research Synthesizer (Claude Sonnet 4.6)
**Relevance:** ABEE Physics Oracle enhancement, oracle pre-processing pipeline, potential replacement or augmentation of SAM2 + MiDaS

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [3D Gaussian Splatting Fundamentals](#3d-gaussian-splatting-fundamentals)
3. [3DGS for Robotic Manipulation](#3dgs-for-robotic-manipulation)
4. [Hand-Object Interaction with 3DGS](#hand-object-interaction-with-3dgs)
5. [Integration with Vision-Language Models](#integration-with-vision-language-models)
6. [NVIDIA Ecosystem: Cosmos, NuRec, and 3DGUT](#nvidia-ecosystem-cosmos-nurec-and-3dgut)
7. [3DGS vs MiDaS for Depth Estimation](#3dgs-vs-midas-for-depth-estimation)
8. [Feasibility Analysis for ABEE](#feasibility-analysis-for-abee)
9. [RTX 4060 Ti 16GB Compute Assessment](#rtx-4060-ti-16gb-compute-assessment)
10. [Integration Recommendations](#integration-recommendations)
11. [Research Gaps and Open Questions](#research-gaps-and-open-questions)
12. [References and Sources](#references-and-sources)
13. [Methodology](#methodology)

---

## Executive Summary

3D Gaussian Splatting (3DGS) has rapidly matured from a novel view synthesis technique (Kerbl et al., SIGGRAPH 2023) into a foundational representation for physical AI and robotics as of 2025-2026. The key value proposition for ABEE is threefold:

1. **Richer 3D geometry than MiDaS**: 3DGS produces explicit, metric-scale point clouds with per-Gaussian attributes, while MiDaS produces relative (unitless) depth maps. When combined with even minimal multi-view input (2-4 cameras or sequential frames), 3DGS depth rendering demonstrably outperforms monocular depth estimation.

2. **Semantic + geometric representation in one structure**: Language features, grasp affordances, physical properties (mass, friction), and depth can all be embedded directly into the Gaussian primitives. This is directly applicable to estimating grip stability, object pose, and handoff safety.

3. **NVIDIA has made 3DGS a first-class citizen of the Cosmos / Isaac Sim ecosystem**: The NuRec library (August 2025) integrates RTX-accelerated Gaussian splatting into Omniverse, and the Cosmos world foundation models use Gaussian splat scenes as their geometric backbone. This is highly aligned with the ABEE project's NVIDIA Cosmos Cookoff context.

**For ABEE specifically**, the most actionable near-term integration is replacing or augmenting the MiDaS depth component in the Physics Oracle with a lightweight 3DGS-based depth renderer initialized from monocular video. This is feasible on the RTX 4060 Ti 16GB, though with important caveats around latency and reconstruction quality documented below.

A more ambitious path - using a 4D Gaussian representation of the hand-object interaction scene as an additional input channel to the Cosmos-Reason2-8B VLMs - is technically grounded by the GaussianVLM and SplatTalk papers (ICCV 2025) but would require significant engineering and may exceed real-time requirements for the current ABEE architecture.

---

## 3D Gaussian Splatting Fundamentals

### What Is 3DGS?

3D Gaussian Splatting, introduced by Kerbl, Kopanas, Leimkuhler, and Drettakis (SIGGRAPH 2023, arXiv 2308.04079), represents a 3D scene as a set of anisotropic 3D Gaussian primitives. Each Gaussian has:

- **Position**: 3D mean (x, y, z)
- **Covariance**: 3x3 matrix (encoded as a scale vector + quaternion rotation) representing the shape and orientation of the ellipsoid
- **Opacity**: alpha transparency
- **Color/appearance**: spherical harmonic coefficients (up to degree 3 = 48 parameters per Gaussian)

Rendering uses a differentiable tile-based rasterizer that projects the 3D Gaussians onto the image plane and alpha-composites them front-to-back. Training uses photometric loss against multi-view images with an adaptive densification/pruning strategy.

**Distinguishing features over NeRF:**

| Property | NeRF | 3DGS |
|---|---|---|
| Representation | Implicit (neural network) | Explicit (point cloud of Gaussians) |
| Rendering | Volume ray marching | Tile-based rasterization |
| Training speed | Hours to days | Minutes (30 min standard; <1 min fast variants) |
| Inference/rendering | 0.1-5 FPS (MLP forward passes) | 30-160+ FPS (GPU rasterization) |
| Scene editing | Difficult | Direct - Gaussians can be moved, removed, cloned |
| Depth extraction | Via integration | Direct from Gaussian z-values |
| Semantic extension | Feature distillation required | Feature embedding per-Gaussian |

### Real-Time Performance (2025 State of the Art)

The original Kerbl et al. implementation achieves **>=30 FPS at 1080p** on an RTX 3090. As of 2025, the performance landscape is:

- **Static scene rendering**: 100-160 FPS at 1080p on RTX 3090/4090 class GPUs. RTX 3060 (laptop) achieves ~60 FPS.
- **Dynamic scene rendering (4DGS)**: 82 FPS at 800x800 on high-end hardware (CVPR 2024).
- **Online incremental reconstruction**: Stereo-GS (Electronics, 2025) achieves real-time streaming reconstruction from stereo pairs, outperforming baselines on EuRoC (+0.22 dB PSNR) and TartanAir (+2.45 dB PSNR).
- **Fast training**: "Fast Converging 3DGS for 1-Minute Reconstruction" (arXiv 2601.19489, 2025) achieves full scene reconstruction in under 60 seconds.
- **NVIDIA DesignWorks vk_gaussian_splatting**: Real-time GPU-accelerated rendering with Vulkan backend.

**Key architectural evolution in 2025-2026:**
- **3DGUT (3D Gaussian Unscented Transform)**: NVIDIA's improved variant supporting distorted camera models (fisheye, rolling shutter), secondary ray tracing, and reflections/refractions within a unified framework.
- **Physically Embodied Gaussians**: Dual particle+Gaussian representation that closes the loop between physics simulation and visual rendering at ~30 Hz.
- **Gaussian World Models (GWM)**: Action-conditioned 3D video prediction for robot manipulation (ICCV 2025).

### How 3DGS Creates 3D Representations from Video

The standard pipeline for video input:
1. **Structure-from-Motion (SfM)**: COLMAP or equivalent extracts camera poses and a sparse point cloud from keyframes
2. **Gaussian initialization**: Sparse SfM points serve as initial Gaussian centers
3. **Differentiable optimization**: Gaussians are iteratively densified and pruned while minimizing photometric loss
4. **Depth extraction**: Depth maps rendered from any viewpoint by projecting Gaussian means through the camera model, sorted by depth

For **monocular video** (single camera, no stereo), modern approaches (DepthSplat, Mode-GS) use a monocular depth prior (e.g., Depth Anything V2) to bootstrap geometric initialization before Gaussian optimization - this is the most relevant configuration for ABEE's single-camera setup.

---

## 3DGS for Robotic Manipulation

### Key Papers (2024-2025)

#### GaussianGrasper (IEEE RA-L 2024)
**Citation count: 82** — Among the most-cited 3DGS robotics papers.

- **Method**: RGB-D input from limited viewpoints; Efficient Feature Distillation (EFD) module uses contrastive learning to embed CLIP/language features per-Gaussian; pre-trained grasp model generates collision-free candidates; normal-guided filtering selects best grasp pose
- **Key insight**: Explicit Gaussian representation provides direct surface normals - a computationally cheap feature critical for grasp quality
- **Relevance to ABEE**: The normal-guided grasp module could estimate whether the robot's grip on the handoff object is stable (wrist perpendicular to object surface)

#### GraspSplats (arXiv 2409.02084, 2024)
- Depth-supervised 3DGS for high-quality scene representations in **under 60 seconds**
- Real-time tracking algorithm for displaced objects in dynamic scenes
- Includes a grasping latency breakdown (quantitative, scene-dependent)
- **Key capability**: Editing reconstructed representations without compromising geometric/semantic features - useful for handoff scene state updates

#### SparseGrasp (arXiv 2412.02140, 2024)
- Open-vocabulary grasping from **sparse-view RGB** (no depth required)
- Uses DUSt3R for dense point cloud initialization
- Render-and-compare strategy for rapid scene updates
- Trained in 7,000 iterations (~4 minutes)
- Significantly outperforms baselines in speed and adaptability

#### Persistent Object Gaussian Splat / POGS (ICRA 2025)
**Directly relevant to human-robot handoff.**

- **Method**: Embeds semantics, self-supervised DINO features, and object grouping into a compact Gaussian representation; continuously updated pose estimation via feature + depth loss minimization
- **Hardware**: Single stereo camera only (no depth sensor required)
- **Tracks both human AND robot manipulation** - explicitly designed for handoff-like scenarios
- **Results**: 12 consecutive successful object resets; 80% recovery from 30-degree tool perturbations during grasp
- **Tracking mechanism**: Loss minimization between rendered DINO features and observed frames - runs without expensive rescanning
- **Relevance to ABEE**: POGS directly models the scenario ABEE reasons about. It could provide object pose estimates (stability, position, orientation) as oracle features.

#### Splat-MOVER (CoRL 2024, 44 citations)
- Three-module stack: ASK-Splat (semantic + affordance), SEE-Splat (real-time scene editing / digital twin), Grasp-Splat (grasp generation)
- ASK-Splat trains in real-time from RGB images in a brief scanning phase
- SEE-Splat and Grasp-Splat operate in real-time during manipulation
- Demonstrated on physical Kinova robot
- **Relevance to ABEE**: The "digital twin" concept (SEE-Splat) provides a continuously updated 3D model of the handoff scene that could feed into the physics oracle

#### PUGS - Zero-shot Physical Understanding (ICRA 2025)
- Reconstructs 3D objects via Gaussian splatting then **predicts physical properties zero-shot**: mass, friction, hardness
- Geometry-aware regularization loss for improved shape quality
- Feature-based property propagation module
- State-of-the-art on ABO-500 mass prediction benchmark
- **Relevance to ABEE**: Physical property prediction from visual geometry could inform grip stability and safe release window estimation (heavy objects need more support; slippery objects are higher risk)

#### RoboSplat (RSS 2025)
- Generates novel demonstration data for policy learning using 3DGS
- One-shot manipulation with 87.8% success vs 57.2% for hundreds of real-world demonstrations
- Generalizes across 6 axes: object poses, types, camera views, scene appearance, lighting, embodiments
- **Relevance to ABEE**: Could generate synthetic training data for SFT dataset across diverse handoff scenarios

#### GWM - Gaussian World Models (ICCV 2025)
- Action-conditioned 3D video prediction using Gaussian Splatting as the world state
- Enhances visual representation learning for imitation learning
- **Relevance to ABEE**: Frame-level world model that could predict how the handoff scene evolves under robot actions

### Architectural Patterns in Robotic 3DGS

Three recurring integration patterns are visible across papers:

**Pattern 1: Reconstruct-then-Query**
- One-time scene reconstruction (30 sec - 4 min)
- Query the Gaussian field at runtime for features, depths, normals
- Examples: GaussianGrasper, GraspSplats
- Latency during operation: real-time (rendering = <10ms per frame)
- **ABEE suitability**: High for static workspace, moderate for dynamic handoff

**Pattern 2: Online Incremental Update**
- Streaming input; Gaussians updated frame-by-frame
- Examples: POGS (pose tracking), Stereo-GS (mapping)
- Latency: variable; pose refinement ~100ms/update with POGS
- **ABEE suitability**: High - matches the ABEE streaming video paradigm

**Pattern 3: Physics-Coupled Representation**
- Dual particle + Gaussian representation with 33ms physics cycle
- Examples: Physically Embodied Gaussians (NVIDIA Warp + gsplat)
- Provides predictive simulation with visual correction
- **ABEE suitability**: Aspirational - high value but significant engineering effort

---

## Hand-Object Interaction with 3DGS

This category is the most directly relevant to ABEE's human-robot handoff scenario.

### Interaction-Aware 4D Gaussian Splatting (arXiv 2511.14540, 2025)

The most relevant paper for ABEE's core problem.

**What it does:**
- Simultaneously models geometry and appearance of **hand-object interaction** from monocular RGB egocentric video
- Separates hand Gaussians from object Gaussians with an interaction-aware deformation field
- Uses MANO hand parameters from any off-the-shelf tracker as coarse 3D guidance
- Requires zero object shape priors - category-agnostic
- Piecewise linear deformation hypothesis with Weight (motion smoothness) and Radius (edge sharpness) parameters

**Performance:**
- State-of-the-art on HOI4D and HO3D benchmarks
- Training: ~80 minutes on RTX 3090 (offline, not suitable for real-time)
- Handles mutual occlusion and edge blur - critical for handoff scenes where hand and object overlap
- Limitation: struggles with "exceedingly rapid motion / complex trajectories"

**ABEE relevance**: The ability to reconstruct the full hand-object interaction geometry from monocular video - without object priors - maps directly to the ABEE scenario. The trained model could serve as a feature extractor during inference, even if training is offline.

### EgoGaussian (ResearchGate 2025)
- Dynamic scene understanding from egocentric (first-person) video with 3DGS
- Separates dynamic foreground (hands, objects) from static background
- **Relevance**: Egocentric viewpoint matches robot-mounted cameras in handoff scenarios

### 6DOPE-GS (ICCV 2025)
- Online 6D object pose estimation and reconstruction using Gaussian Splatting
- Live pose tracking and reconstruction of dynamic objects
- **Relevance**: Provides 6DoF object pose during handoff - position + orientation of the object being transferred

### Object and Contact Point Tracking (arXiv 2411.03555)
- Tracks objects AND contact points in demonstrations using 3DGS
- Contact point tracking directly relevant to grip stability assessment
- **Relevance to ABEE**: Contact geometry is a key physical cue for safe release window prediction

---

## Integration with Vision-Language Models

This section addresses whether Gaussian splat representations can be fed directly into VLMs like Cosmos-Reason2-8B.

### GaussianVLM (arXiv 2507.00886, IEEE 2025)
**The first 3D VLM operating natively on Gaussian splats.**

**Architecture:**
- SceneSplat-based 3D vision module predicts a SigLIP2 language feature for each Gaussian primitive end-to-end
- Dual sparsifier compresses ~40,000 language-augmented Gaussians to **132 tokens**
  - Task-guided pathway: retains task-relevant Gaussians
  - Location-guided pathway: retains spatially distributed Gaussians
- LLM processes 132 scene tokens + text query

**Performance:**
- 5x improvement over prior 3D VLM (LL3DA) in out-of-domain settings
- CIDEr scores: embodied dialogue 145.9 -> 270.1; planning 65.1 -> 220.4
- Designed for embodied reasoning tasks including spatial QA and planning

**ABEE relevance**: The 132-token compression scheme is key. It means a full 3D Gaussian scene representation can be fed to a VLM within a reasonable token budget. This is the path to feeding Cosmos-Reason2-8B a 3D scene representation directly.

### SplatTalk (ICCV 2025, arXiv 2503.06271)
**3D VQA with Gaussian Splatting - zero-shot capable.**

**Architecture:**
- Feed-forward 3DGS model processes posed RGB images
- LLaVA-OV encodes 2D images; multimodal projector creates visual-language feature maps
- Features embedded per-Gaussian into a "3D-Language Gaussian Field"
- At inference: language features extracted from Gaussians at their 3D positions
- 3,584-dim features compressed to 256-dim via autoencoder
- **32,076 tokens** fed to Qwen2 LLM using entropy-adaptive sampling (selects highest-entropy Gaussians first)

**Key implication for ABEE**: SplatTalk takes posed RGB images as input (exactly what ABEE has) and produces tokens directly compatible with a standard LLM interface. No depth sensor required. The entropy-adaptive sampling ensures semantically informative regions (hands, objects in transition) are prioritized.

### SceneSplat + 3D Vision-Language Gaussian Splatting (ICLR 2025)
- Scene understanding via pretrained VLP (Vision-Language Pretraining) on Gaussian representations
- 3D VLG model emphasizes language modality representation learning
- **Relevance**: Establishes the training methodology pipeline for scene-level Gaussian VLMs

### Practical Integration Path for Cosmos-Reason2-8B

Cosmos-Reason2-8B uses a standard VLM architecture (vision encoder + LLM). Direct feeding of Gaussian splat tokens would require either:

**Option A - Render-to-Image (No Architecture Change):**
- Render the Gaussian splat from a novel viewpoint (e.g., bird's eye, side view)
- Feed rendered depth map + RGB as additional image channels to existing VLM
- Zero architectural change to Cosmos-Reason2-8B
- Latency: ~10ms for rendering; compatible with current pipeline

**Option B - GaussianVLM Token Injection:**
- Use SceneSplat to extract 132 Gaussian scene tokens
- Prepend to text/image tokens before Cosmos-Reason2-8B attention layers
- Requires architectural modification or fine-tuning
- High value but significant engineering investment

**Option C - Feature Map Augmentation:**
- Extract per-Gaussian features (depth, normal, semantic) and render as additional feature channels
- Feed as extra "views" alongside standard RGB frames
- Moderate engineering effort; potentially achievable in the ABEE SFT dataset pipeline

---

## NVIDIA Ecosystem: Cosmos, NuRec, and 3DGUT

### NVIDIA NuRec (Announced August 2025)
NuRec (Neural Reconstruction) is NVIDIA's production Gaussian splatting library for Omniverse.

**Technical stack:**
- **Input**: Multi-view images, LiDAR, stereo cameras, or smartphone footage
- **Processing**: COLMAP (SfM) -> 3DGUT training -> USDZ export
- **Rendering**: Omniverse RTX ray-traced Gaussian splatting
- **Output**: USD assets loadable in Isaac Sim as standard scene objects

**Key capabilities:**
- Supports 3DGS, NeRF, and 3DGUT (best of all three)
- 3DGUT adds: fisheye/rolling shutter camera support, reflections, refractions
- FiftyOne (Voxel51) integration for data preparation
- Nova Carter (NVIDIA's reference robot) datasets built with NuRec stereo workflow

**NuRec stereo workflow for robotics:**
```
Stereo RGB -> Isaac ROS -> cuSFM (pose estimation) -> FoundationStereo (depth)
-> nvblox (mesh) -> 3DGURT (neural reconstruction) -> Isaac Sim USD asset
```

### Cosmos Integration with Gaussian Splatting

The NVIDIA announcement (August 2025, GTC DC 2025) explicitly states:

> "Cosmos models learn from — and generate — worlds and use Gaussian Splat scenes as the geometric backbone."

This is architecturally significant: Cosmos world foundation models use 3DGS as their scene representation internally. The implication is that **feeding Gaussian splat representations to Cosmos-class models is the intended design direction** for NVIDIA's physical AI stack.

**Cosmos Transfer-1** (related product): Generates photorealistic controllable video in under 30 seconds using Gaussian splat scenes as geometric control signals.

### NVIDIA WARP + Gaussian Splatting: Physically Embodied Gaussians
A production-feasible architecture described in NVIDIA's technical blog:

- **NVIDIA Warp** (extended position-based dynamics / XPBD) handles physics at ~30 Hz
- **gsplat** (differentiable rendering) provides visual supervision
- Physics cycle: "only needs to remain accurate for 33ms before visual correction"
- Fewer cameras than traditional 3DGS: leverages known robot geometry, poses, and physical constraints
- Creates live digital twins for downstream robot tasks

**This 33ms physics cycle matches ABEE's frame rate requirements** for a real-time oracle.

### 3DGUT vs Standard 3DGS

| Feature | Standard 3DGS | 3DGUT (NVIDIA) |
|---|---|---|
| Camera models | Pinhole only | Fisheye, rolling shutter, any distortion |
| Ray tracing | Rasterization only | Secondary ray tracing supported |
| Reflections/refractions | No | Yes |
| Isaac Sim integration | Via conversion | Native USD asset |
| Open source | Yes (graphdeco-inria) | 3DGRUT on GitHub (nv-tlabs) |

---

## 3DGS vs MiDaS for Depth Estimation

### MiDaS Limitations (Current ABEE Oracle)

MiDaS (and its successor Depth Anything V2) produces **relative depth** - values have no absolute scale. Key limitations for ABEE:

- Scale is unknown and varies between frames
- Cannot directly compute metric distances (e.g., "object is 45cm from robot")
- Accuracy degrades on thin objects (fingers, cables), reflective surfaces, and near occlusion boundaries
- Single frame - no temporal consistency; noisy between frames
- No surface normal output directly

### 3DGS Depth Advantages

3DGS renders **metric depth** maps (when initialized from calibrated multi-view or stereo input):

- Absolute scale: distances in real-world units
- Temporal consistency: Gaussians persist across frames; depth is stable
- Surface normals: directly computed from Gaussian covariance orientation
- Occlusion handling: explicit 3D structure handles occlusion correctly
- Uncertainty quantification: GaussianLSS (CVPR 2025) provides uncertainty-aware depth via soft depth distributions

### Quantitative Comparison (from literature)

**Mode-GS (arXiv 2410.04646)** - Monocular Depth Guided 3DGS:
- Uses MiDaS-derived depth for initialization, then 3DGS refines
- Result: 3DGS rendering significantly outperforms raw MiDaS depth
- Key finding: monocular depth prior + 3DGS optimization beats either alone

**DepthSplat (CVPR 2025)**:
- Bidirectional synergy: 3DGS improves depth; depth priors improve 3DGS
- Depth Anything V2 features outperform MiDaS features as priors
- Achieves state-of-the-art in both depth estimation and novel view synthesis

**SLAM comparison (from 3DGS in Robotics Survey)**:
- RGB-D 3DGS SLAM (with depth): ATE RMSE 0.16-0.25 cm
- RGB-only 3DGS SLAM (no depth): ATE RMSE 0.51-8.51 cm (highly variable)
- **Conclusion**: 3DGS needs depth initialization for reliable geometry; pure monocular is unreliable without strong priors

### Practical Recommendation for ABEE

The ideal configuration is a hybrid:

```
RGB frame -> Depth Anything V2 (metric depth prior) -> 3DGS initialization
                    -> 3DGS optimization (3-10s per scene) -> Metric depth render
                    -> Surface normals -> Object pose
```

This uses MiDaS-class monocular depth as a bootstrapping signal, then improves it with 3DGS consistency. The output is superior to raw monocular depth in both accuracy and temporal stability.

---

## Feasibility Analysis for ABEE

### What ABEE Needs from an Oracle

The current Physics Oracle (SAM2 + MiDaS) provides:
- Object/hand segmentation masks (SAM2)
- Relative depth maps (MiDaS)
- Hard-veto capability (force all agents to THINK)

A 3DGS-enhanced oracle could additionally provide:
- Metric 3D positions (hand, object, robot end-effector)
- Object pose (6DoF rotation + translation)
- Surface normals (grip quality indicator)
- Contact point geometry
- Physical property estimates (mass, friction via PUGS)
- Temporal consistency across frames (Gaussian state persists)

### Scenario Analysis: Handoff Scene

The ABEE handoff scenario has specific properties:
- **Duration**: Brief (2-10 seconds)
- **Camera**: Single (likely monocular) or stereo pair
- **Subjects**: Human hand, object, robot gripper
- **Dynamic**: Hand + object move; background relatively static
- **Scale**: Tabletop / arm-length distances (0.3-1.5m)

**3DGS mode most suited: Online incremental + object-centric tracking**

The recommended architecture for ABEE would be:
1. Background scene reconstructed offline (static Gaussians, trained once)
2. Dynamic foreground (hand, object) tracked per-frame using POGS-style feature+depth loss minimization
3. Object pose extracted from Gaussian centroid/orientation
4. Depth map rendered from known camera pose
5. Stability metric derived from contact point geometry and object pose stability

### Latency Budget Analysis

ABEE runs at 30 FPS (33ms per frame). Current oracle components:
- SAM2 segmentation: ~15-30ms per frame (on RTX 4060 Ti)
- MiDaS depth: ~10-20ms per frame

3DGS oracle estimates (RTX 4060 Ti, 16GB VRAM):
- **Scene rendering (static Gaussians)**: ~5-10ms per frame at 720p
- **POGS-style pose tracking**: ~50-150ms per update (runs at 10-20Hz, interpolated)
- **Dynamic splat update (foreground only)**: ~20-50ms per frame
- **Total 3DGS overhead**: 30-60ms for rendering + tracking

**Assessment**: At the target 30 FPS, a full 3DGS oracle running alongside the current SAM2 pipeline would likely exceed the frame budget on an RTX 4060 Ti. A practical solution: run 3DGS at 10-15 FPS and use interpolation for intermediate frames. The hard-veto oracle function can operate at lower refresh rates since it prevents ACT (not at THINK timing).

### VRAM Budget (RTX 4060 Ti 16GB)

Current ABEE VRAM usage estimates:
- Cosmos-Reason2-8B (4-bit quantized): ~6-8GB
- SAM2 + MiDaS: ~2-3GB
- Overhead (activations, KV cache): ~2GB

Remaining headroom: **~3-6GB**

3DGS VRAM requirements:
- Static scene (small room, ~500K Gaussians): ~1-2GB
- Dynamic foreground objects (hand + object): ~500MB-1GB
- Gaussian rendering compute: ~500MB
- **Total 3DGS addition**: ~2-4GB

**Assessment**: Tight but feasible with careful memory management. 3DGS must share the GPU with the VLMs, which means batching their memory usage carefully. The 16GB VRAM of the RTX 4060 Ti is genuinely important here - the 8GB variant would not be workable.

### Training Time Considerations

For the ABEE deployment scenario:
- **Background scene reconstruction**: One-time, done offline (~1-5 minutes with fast 3DGS variants)
- **Object-specific Gaussian templates**: Pre-built for known object categories
- **Per-session adaptation**: 30-60 seconds if scene changes significantly
- **Online tracking**: No training required - inference only (POGS paradigm)

This is operationally manageable if the workspace (table, robot arm) is known in advance.

---

## RTX 4060 Ti 16GB Compute Assessment

### GPU Architecture Characteristics

The RTX 4060 Ti (AD106 die) key specs for 3DGS workloads:
- CUDA cores: 4,352
- VRAM: 16GB GDDR6 (128-bit bus - the primary bottleneck)
- Memory bandwidth: 288 GB/s (significantly lower than RTX 4090's 1,008 GB/s)
- Tensor cores: 4th gen (good for quantized inference)
- RT cores: 3rd gen (relevant for 3DGUT ray tracing)

**Key limitation**: The 128-bit memory bus is the primary bottleneck for 3DGS workloads, which are highly memory-bandwidth-bound. Rendering speed will be approximately 2-3x slower than RTX 4090 benchmarks published in papers.

### Realistic Performance Estimates for ABEE

Adjusting published benchmarks for RTX 4060 Ti:

| Task | Published (RTX 3090/4090) | Estimated RTX 4060 Ti |
|---|---|---|
| Static scene rendering 720p | 120-160 FPS | 40-70 FPS |
| Dynamic Gaussian rendering | 60-80 FPS | 20-35 FPS |
| Fast 3DGS training (1M Gaussians) | 60s | 180-240s |
| POGS pose tracking update | 50-100ms | 150-300ms |
| Online SLAM mapping | 5-10 FPS | 3-5 FPS |

**Rendering performance is adequate** for ABEE oracle purposes (40-70 FPS > 30 FPS target).
**Pose tracking and SLAM** may be too slow for fully real-time operation at 30 FPS.

### Recommended Configuration for ABEE RTX 4060 Ti

1. Use **gsplat** (the CUDA-optimized Python library, used in Physically Embodied Gaussians) rather than the original graphdeco-inria CUDA rasterizer - typically 1.5-2x faster
2. Limit Gaussian count: foreground objects in handoff scenes need at most 50K-200K Gaussians
3. Render at 480p or 540p for oracle purposes (upscale with bilinear if needed)
4. Run 3DGS oracle at 10-15 Hz; VLMs receive rendered features asynchronously
5. Precompute background Gaussian scene; only update foreground Gaussians

---

## Integration Recommendations

### Recommendation 1: Near-Term (Low Risk, High Value)
**Replace MiDaS with DepthSplat-style Hybrid Depth**

- Keep SAM2 for segmentation
- Replace MiDaS with Depth Anything V2 (stronger monocular depth, same latency)
- Add lightweight 3DGS depth refinement: initialize from Depth Anything V2, refine online for temporal consistency
- Cost: ~15-25ms additional latency; ~1GB additional VRAM
- Value: Metric-scale depth, stable depth across frames, surface normals for free

**Implementation path:**
```python
# In abee_pkg/oracle.py
depth_prior = DepthAnythingV2(encoder='vitl').predict(frame)  # replace MiDaS
gaussians = OnlineGaussianTracker.update(frame, depth_prior, camera_pose)
metric_depth = gaussians.render_depth(camera)
surface_normals = gaussians.render_normals(camera)
```

### Recommendation 2: Medium-Term (Moderate Engineering, Very High Value)
**Add POGS-Style Object Pose Tracking to Oracle**

- Pre-build Gaussian templates for common handoff objects offline
- During handoff: POGS-style DINO feature + depth loss minimization to track object 6DoF pose
- Feed object pose stability (rate of change of pose) as a hard-veto signal
- If object pose is rapidly changing or uncertain: veto to THINK
- Cost: ~2GB VRAM, 150-300ms update latency (runs at 5-10 Hz)
- Value: Direct measurement of "is the object stable in the transfer?" which is exactly what ABEE reasons about

**Oracle hard-veto extension:**
```python
# High pose uncertainty or rapid motion -> force all agents to THINK
if object_pose_velocity > HANDOFF_STABILITY_THRESHOLD:
    return OracleDecision.FORCE_THINK, {"reason": "object_unstable_3dgs"}
```

### Recommendation 3: Long-Term (High Engineering, Transformative Value)
**3DGS-Augmented VLM Input Channels**

- Use SplatTalk / GaussianVLM token approach to feed 3D scene tokens to Cosmos-Reason2-8B
- Option A (no arch change): Render 3DGS depth + normal maps as additional image channels
- Option B (architecture change): Prepend 132 Gaussian scene tokens to VLM context
- This would enable agents to reason about 3D geometry directly ("the object is tilted 15 degrees, grip is suboptimal")
- Cost: Significant engineering; may require fine-tuning Cosmos-Reason2-8B
- Value: 3D spatial reasoning that is currently impossible from flat RGB frames alone

### Recommendation 4: SFT Dataset Enhancement
**Use RoboSplat-Style Data Augmentation**

- For the curated SFT dataset, use 3DGS to generate novel viewpoints, lighting conditions, and object variations from existing successful handoff demonstrations
- RoboSplat (RSS 2025) shows 87.8% policy success from one-shot demonstrations with this approach
- Cost: Moderate (offline pipeline); VRAM not a constraint for offline processing
- Value: Dramatically increases effective size and diversity of the SFT training set

### Decision Matrix

| Integration | Latency Added | VRAM Added | Engineering Effort | Value to ABEE | Recommendation |
|---|---|---|---|---|---|
| Depth Anything V2 (drop-in) | ~5ms | ~500MB | Low | Medium | Do now |
| Online 3DGS depth refinement | ~15ms | ~1GB | Medium | High | Phase 1 |
| POGS object pose tracking | ~100ms @ 10Hz | ~2GB | Medium-High | Very High | Phase 2 |
| 3DGS-augmented VLM tokens | ~20ms | ~1GB | High | Transformative | Phase 3 |
| SFT data augmentation (offline) | N/A | N/A | Low-Medium | High | Parallel track |

---

## Research Gaps and Open Questions

Several areas have insufficient research for confident recommendations:

1. **Monocular 4D Gaussian tracking latency at 30 FPS**: Most published results are on RTX 3090/4090. The 33ms budget on RTX 4060 Ti for online tracking is borderline. Empirical benchmarking on the target hardware is required before committing to this architecture.

2. **3DGS robustness to occlusion during handoff**: The critical moment of handoff (when robot gripper and human hand both hold the object) creates mutual occlusion that 3DGS handles poorly without specialized treatment (Interaction-Aware 4DGS addresses this but requires 80 min training offline).

3. **POGS tracking frequency vs. accuracy trade-off**: The published 5-10 Hz POGS update rate may introduce latency artifacts relative to the 30 FPS VLM input stream. Interpolation between updates is unstudied in high-dynamics scenarios.

4. **Compatibility with Cosmos-Reason2-8B**: No published work directly integrates GaussianVLM tokens with Cosmos-class models. The 132-token budget may need adjustment for Cosmos-Reason2's specific attention architecture.

5. **Memory bandwidth bottleneck quantification**: The RTX 4060 Ti's 128-bit bus vs RTX 4090's 384-bit bus creates a 3x theoretical bandwidth deficit. The practical impact on 3DGS rendering framerate for ABEE-scale scenes (small tabletop, 50K-200K Gaussians) has not been benchmarked specifically.

6. **Differentiable depth for life-points feedback**: Whether 3DGS depth uncertainty signals could be used as an additional signal for the ABEE Life-Points system (high depth uncertainty -> higher cost for ACT) is unexplored but theoretically motivated.

---

## References and Sources

### Foundational Papers

- Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *SIGGRAPH 2023*. arXiv:2308.04079. [Link](https://arxiv.org/abs/2308.04079)

### Robotic Manipulation (Peer-Reviewed)

- Zheng, Y., et al. (2024). GaussianGrasper: 3D Language Gaussian Splatting for Open-Vocabulary Robotic Grasping. *IEEE Robotics and Automation Letters*, 9, 7827-7834. 82 citations. [Consensus](https://consensus.app/papers/gaussiangrasper-3d-language-gaussian-splatting-for-zheng-chen/8dec9c03c7c952a9acb6484021b99a79/)

- Yu, J., Hari, K., et al. (2025). Persistent Object Gaussian Splat (POGS) for Tracking Human and Robot Manipulation of Irregularly Shaped Objects. *ICRA 2025*, pp. 3211-3218. 10 citations. [arXiv](https://arxiv.org/abs/2503.05189)

- Shorinwa, O., Tucker, J., et al. (2024). Splat-MOVER: Multi-Stage, Open-Vocabulary Robotic Manipulation via Editable Gaussian Splatting. *CoRL 2024*. 44 citations. [arXiv](https://arxiv.org/abs/2405.04378)

- PUGS: Zero-shot Physical Understanding with Gaussian Splatting. *ICRA 2025*. [arXiv](https://arxiv.org/abs/2502.12231)

- Lu, G., et al. (2025). GWM: Towards Scalable Gaussian World Models for Robotic Manipulation. *ICCV 2025*. [CVF](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_GWM_Towards_Scalable_Gaussian_World_Models_for_Robotic_Manipulation_ICCV_2025_paper.pdf)

- InternRobotics. (2025). Novel Demonstration Generation with Gaussian Splatting Enables Robust One-Shot Manipulation (RoboSplat). *RSS 2025*. [arXiv](https://arxiv.org/abs/2504.13175)

### Hand-Object Interaction

- Interaction-Aware 4D Gaussian Splatting for Dynamic Hand-Object Interaction Reconstruction. (2025). arXiv:2511.14540. [Link](https://arxiv.org/abs/2511.14540)

- Object and Contact Point Tracking in Demonstrations Using 3D Gaussian Splatting. (2024). arXiv:2411.03555.

- 6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting. *ICCV 2025*. [CVF](https://openaccess.thecvf.com/content/ICCV2025/papers/Jin_6DOPE-GS_Online_6D_Object_Pose_Estimation_using_Gaussian_Splatting_ICCV_2025_paper.pdf)

### VLM Integration

- GaussianVLM: Scene-centric 3D Vision-Language Models using Language-aligned Gaussian Splats for Embodied Reasoning and Beyond. (2025). arXiv:2507.00886. [Link](https://arxiv.org/abs/2507.00886)

- Thai, A., & Peng, S. (2025). SplatTalk: 3D VQA with Gaussian Splatting. *ICCV 2025*. [CVF](https://openaccess.thecvf.com/content/ICCV2025/papers/Thai_SplatTalk_3D_VQA_with_Gaussian_Splatting_ICCV_2025_paper.pdf)

- Li, et al. (2025). SceneSplat: Gaussian Splatting-based Scene Understanding with Vision-Language Pretraining. *ICCV 2025*. [CVF](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_SceneSplat_Gaussian_Splatting-based_Scene_Understanding_with_Vision-Language_Pretraining_ICCV_2025_paper.pdf)

- 3D Vision-Language Gaussian Splatting. *ICLR 2025*. [OpenReview](https://openreview.net/forum?id=SSE9myD9SG)

### Depth Estimation

- Lu, S.-W., Tsai, Y.-H., & Chen, Y.-T. (2025). Toward Real-world BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting. *CVPR 2025*, pp. 17124-17133. 8 citations. [Consensus](https://consensus.app/papers/toward-realworld-bev-perception-depth-uncertainty-lu-tsai/)

- Park, J., et al. (2025). Stereo-GS: Online 3D Gaussian Splatting Mapping Using Stereo Depth Estimation. *Electronics 2025*. [Consensus](https://consensus.app/papers/stereogs-online-3d-gaussian-splatting-mapping-using-park-lee/)

- Xu, H., & Peng, S. (2025). DepthSplat: Connecting Gaussian Splatting and Depth. *CVPR 2025*. [PDF](https://www.cvlibs.net/publications/Xu2025CVPR.pdf)

### NVIDIA Ecosystem

- NVIDIA NuRec Documentation. (2025). [NVIDIA Docs](https://docs.nvidia.com/nurec/index.html)

- NVIDIA. (2025, August 11). NVIDIA Opens Portals to World of Robotics With New Omniverse Libraries, Cosmos Physical AI Models. [NVIDIA Newsroom](https://nvidianews.nvidia.com/news/nvidia-opens-portals-to-world-of-robotics-with-new-omniverse-libraries-cosmos-physical-ai-models-and-ai-computing-infrastructure)

- NVIDIA Technical Blog. (2025). Building Robotic Mental Models with NVIDIA Warp and Gaussian Splatting. [NVIDIA Developer](https://developer.nvidia.com/blog/building-robotic-mental-models-with-nvidia-warp-and-gaussian-splatting/)

- NVIDIA Technical Blog. (2025). How to Instantly Render Real-World Scenes in Interactive Simulation. [NVIDIA Developer](https://developer.nvidia.com/blog/how-to-instantly-render-real-world-scenes-in-interactive-simulation/)

- NVIDIA. (2025). Revolutionizing Neural Reconstruction and Rendering in gsplat with 3DGUT. [NVIDIA Developer](https://developer.nvidia.com/blog/revolutionizing-neural-reconstruction-and-rendering-in-gsplat-with-3dgut/)

- nv-tlabs. (2025). 3DGRUT: Ray tracing and hybrid rasterization of Gaussian particles. [GitHub](https://github.com/nv-tlabs/3dgrut)

### Surveys

- Zhu, S., Wang, G., et al. (2024). 3D Gaussian Splatting in Robotics: A Survey. arXiv:2410.12262. [arXiv](https://arxiv.org/abs/2410.12262)

- Sandula, A.K., et al. (2025). Real-Time 3D Reconstruction via Camera-Lidar (2D) Fusion for Mobile Robots: A Gaussian Splatting Approach. *ICRA 2025*, pp. 14557-14563.

- Gao, K., et al. (2026). NeRF: Neural Radiance Field in 3D Vision: A Comprehensive Review (Updated Post-Gaussian Splatting). arXiv:2210.00379v8 (updated Feb 2026).

---

## Methodology

**Research conducted:** 2026-03-05

**Tools used:**
- WebSearch (12 targeted queries across 5 research angles)
- Consensus Academic Search API (2 structured queries with year filters)
- Paper Search MCP - arXiv and Semantic Scholar (2 queries)
- WebFetch (5 direct paper/documentation fetches for technical detail extraction)

**Search strategy:**
- Phase 1: Broad landscape searches for 3DGS robotics (2025), handoff scenarios, NVIDIA ecosystem
- Phase 2: Targeted searches for specific papers (POGS, GaussianVLM, SplatTalk, PUGS, Interaction-Aware 4DGS)
- Phase 3: Technical detail extraction via direct page fetches for compute benchmarks and pipeline specifications
- Phase 4: Depth comparison literature for MiDaS vs 3DGS analysis

**Source quality assessment:**
- High confidence: IEEE RA-L, CVPR, ICCV, ICRA, RSS, CoRL publications (peer-reviewed)
- Medium confidence: arXiv preprints with >10 citations or NVIDIA official documentation
- Lower confidence: Single-source technical blog posts (used for implementation details only)

**Limitations:**
- RTX 4060 Ti benchmarks are extrapolated from RTX 3090/4090 published numbers; empirical validation on target hardware is recommended before major architectural decisions
- Papers from ICCV 2025 and RSS 2025 may have preprint versions with minor differences from final publication
- The 3DGS landscape is evolving extremely rapidly (daily arXiv submissions); this document reflects the state as of early March 2026
