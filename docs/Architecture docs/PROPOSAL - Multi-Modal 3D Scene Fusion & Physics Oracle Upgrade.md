# PROPOSAL — Multi-Modal 3D Scene Fusion Layer & Physics Oracle Upgrade

**Status:** DRAFT — Open for Discussion  
**Authors:** AJ  
**Date:** 5 March 2026  
**Classification:** Architecture Proposal (not yet merged into master system doc)  
**Supersedes:** Partial sections of `System Architecture - CLASP.md` §Physics Oracle and §Sensor Modalities (if accepted)

---

## 0. Purpose and Scope

This document proposes an **upgrade to the CLASP Physics Oracle layer** and the introduction of a **persistent 3D scene representation** as the primary geometric substrate for all oracle-derived signals.

The current oracle (`clasp_pkg/oracle.py`) has three documented structural weaknesses (identified in the codebase audit, March 2026):

1. **SAM2 auto-segmentation assigns semantic labels by stability score** — "gripper", "object", "hand" label assignment is arbitrary and ungrounded.
2. **MiDaS depth is monocular and relative** — metric clearance distances cannot be computed; depth is not integrated into the hard-veto logic.
3. **Per-frame 2D analysis has no temporal scene model** — each frame is evaluated in isolation; the "normal" scene state is not represented, so only frame-local anomalies can be detected.

The proposal addresses all three issues by introducing a **layered spatial intelligence stack** built on top of Apple SHARP, with optional additive sensor modalities (IR, WiFi CSI, ultrasonic). Every component has a defined **graceful degradation path** — the system must remain functional when only an RGB camera is available.

This document is written as a basis for discussion. Nothing here is a committed implementation decision.

---

## 1. Overview — The Proposed Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                CLASP Perception Pipeline (Proposed)             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4 — VLM Agents (unchanged)                              │
│    Cosmos-Reason2-8B × N, consuming oracle text block          │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3 — Oracle Signal Aggregation (upgraded)                │
│    ConstraintReport: metric units, grounded labels             │
│    Sources: 3DGS delta, SAM2 (prompted), depth, proximity      │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2 — 3D Scene Intelligence                               │
│    SHARP: metric 3DGS, single-image, <1s, >100fps render       │
│    Glinty NDF shader: correct metallic/glint appearance        │
│    RTX Neural Materials: glass / transparent appearance        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1 — Sensor Inputs (additive, all optional except RGB)   │
│    RGB camera (required baseline)                              │
│    Near-IR depth (RealSense D435/D455) — metric contact area   │
│    Thermal IR (FLIR Lepton 3.5) — heat signatures, presence    │
│    Ultrasonic (HC-SR04) — gripper-proximity trigger             │
│    WiFi CSI (ESP32) — macro-level occupancy anomaly            │
└─────────────────────────────────────────────────────────────────┘
```

**Key design principle:** Each layer consumes what is available from the layer below and provides the richest signal it can. Missing sensor data is not a failure — it is a graceful fallback. The VLM agents at the top always receive a well-formed `ConstraintReport`; what changes is the source quality of the numbers inside it.

---

## 2. Component Deep-Dives

### 2.1 Apple SHARP — Metric 3D Gaussian Splatting from Single Images

**Paper:** *Sharp Monocular View Synthesis in Less Than a Second* (Mescheder et al., Apple, arXiv:2512.10685, 2025)  
**Code:** https://github.com/apple/ml-sharp  
**Licence:** Apple ML Research open-source (non-commercial research)

#### What it does

SHARP takes a **single RGB photograph** and regresses the full parameters of a 3D Gaussian Splatting scene representation via one feedforward pass through a neural network. Key properties:

| Property | Value |
|---|---|
| Inference time | < 1 second on a standard GPU |
| Render speed | > 100 fps at high resolution |
| Depth representation | **Metric with absolute scale** — real-world metres |
| Generalisation | Zero-shot across ETH3D, ScanNet++, TanksAndTemples, WildRGBD, Middlebury |
| LPIPS improvement | 25–34% reduction vs best prior model (Flash3D, LVSM, TMPI) |
| DISTS improvement | 21–43% reduction vs best prior model |

The **metric absolute scale** is the property that matters most for us. Current MiDaS depth is relative (unitless; you cannot convert pixel disparity to centimetres without an explicit calibration baseline). SHARP produces Gaussians at real-world scale, enabling oracle signals like `contact_area` and `centroid_velocity_div` to be expressed in cm² and cm/s respectively — directly improving the physical interpretability of the `ConstraintReport`.

#### How it solves the "re-splatting drift" problem

The concern with periodic re-splatting for a live digital twin is that alignment between successive splats requires ICP or feature matching, which introduces drift and computational overhead. SHARP eliminates this because:

- Any new frame can produce a **fresh independent metric splat** without needing the previous one
- The metric scale is absolute, so two SHARP splats from different times are already in the same coordinate system if the camera calibration is fixed
- Delta detection becomes: `Gaussians(t) vs Gaussians(t-N)` — change is directly the geometric difference between two metric point clouds

Recommended operational mode for live tracking:

```
T=0 (scene setup):    Run SHARP → reference splat S_ref (background model)
T=live (per-frame):   Run SHARP → current splat S_t
                      Compute delta: D = hausdorff(S_t, S_ref) per spatial region
                      Regions with D > θ → "changed" → pass to SAM2 for labelling
```

This isolates novel object interactions (robot arm, human hand entering frame) from the stable background without per-frame SAM2 on the full image.

#### Current limitations to be aware of

- **No `.ply`/`.splat` export yet** — GitHub issue #8, community PRs in progress. Rendering is currently within the Python API only. For Blender/UE5 integration this matters; for our Python pipeline it does not.
- **No inter-frame Gaussian identity tracking** — SHARP splats across frames do not maintain consistent Gaussian indices. For the delta-detection use case above this is fine (we diff point clouds, not indices). For animated object tracking it would require a separate tracker (e.g. frame-to-model ICP on the extracted point cloud).
- **Transparent surfaces**: SHARP uses standard SH appearance — glass and chrome are handled at the rendering layer (see §2.3 and §2.4), not at the reconstruction layer.

#### Questions for team discussion

- Do we run SHARP per-frame (maximum freshness, ~1s per frame) or on a schedule (every N frames, then interpolate)?
- Reference splat policy: re-generate S_ref at trajectory start? Or keep a long-term factory floor reference updated daily?
- License: Apple ML Research is non-commercial. If this work moves toward commercial deployment, we need a replacement (nerfstudio splatfacto, or train our own feed-forward 3DGS regressor).

---

### 2.2 Additional Sensor Modalities

These are all **additive layers** — none is a required dependency. Each is assessed for value, cost, and integration complexity.

#### 2.2.1 Near-Infrared Depth — Intel RealSense D435/D455

| Attribute | Value |
|---|---|
| Depth resolution | ~1–3cm accuracy at 0.3–3m range |
| Output | Dense metric depth map, aligned RGB |
| Cost | ~£150–250 (D435), ~£230–350 (D455 — wider FOV, better for workspace scenes) |
| Integration | USB3, OpenCV/pyrealsense2, drop-in for MiDaS replacement |
| Limitation | Structured-light IR fails in direct sunlight; outdoor/skylight factory bays need ToF version (D457) |

**Impact on oracle:** Replaces `DepthOracle` (MiDaS monocular) with metric ground truth. `min_clearance` becomes real centimetres. This is the **single highest-value hardware addition** for the handoff oracle specifically.

**Current gap it closes:** The audit noted that `depth_stats` from `DepthOracle` is injected into the prompt text but never influences `ConstraintReport.should_veto`. With a RealSense, we can add a hard veto threshold like `if min_clearance_cm < 3.0: veto()` with physical meaning.

#### 2.2.2 Thermal Infrared — FLIR Lepton 3.5

| Attribute | Value |
|---|---|
| Resolution | 160×120 px LWIR (8–14μm) |
| Output | Surface temperature per pixel |
| Cost | ~£60–90 on PureThermal breakout board |
| Integration | USB, [pylepton](https://github.com/groupgets/pylepton) library |
| Limitation | Cannot penetrate glass or most plastics; direct sunlight saturation |

**Primary use case for us:** Not handoff detection directly — thermal cannot tell you whether a grip is stable. More useful as a **factory-wide monitoring layer**: motor heat, human presence detection before a robot enters a workspace, heat stress on components. Could feed SAM2 as a secondary modality for human hand detection (hands are ~33°C in a cold factory environment — trivially separable).

**Secondary use case:** Verify human-hand contact in the handoff (`is human hand actually in contact with object?`) using the thermal signature of skin contact.

#### 2.2.3 Ultrasonic Proximity — HC-SR04 / JSN-SR04T

| Attribute | Value |
|---|---|
| Range / accuracy | 2cm–4m, ~2mm precision |
| Latency | < 5ms round-trip |
| Cost | ~£2–4 (HC-SR04), ~£6 (JSN-SR04T waterproof) |
| Integration | GPIO (Raspberry Pi / Arduino), or USB bridge |
| Limitation | Poor angular resolution (~30° cone); soft materials absorb signal |

**Use case:** Mount **one sensor on (or adjacent to) the gripper** as a direct proximity trigger. It answers the binary question "is an object within 8cm?" with near-zero latency — faster than any camera pipeline. This does not add to scene reconstruction; it adds a **hard real-time safety gate** at the execution layer.

Integration into oracle:

```python
# ConstraintReport extension
proximity_cm: float = float("inf")   # distance to nearest object from gripper mount

@property
def should_veto(self) -> bool:
    return (
        self.has_grip_break
        or self.has_velocity_spike
        or self.physics_score < PHYSICS_SCORE_MIN
        or self.vision_reliability < VISION_RELIABILITY_MIN
        or self.proximity_cm > PROXIMITY_RELEASE_MAX_CM   # new: object too far for safe release
    )
```

#### 2.2.4 WiFi CSI — ESP32 with CSI Firmware

| Attribute | Value |
|---|---|
| Wavelength | 2.4GHz (~12cm), 5GHz (~6cm) |
| Spatial resolution | Dm-scale bulk structure; human body localisation, not object detail |
| Cost | ~£8–15 per ESP32 node |
| Integration | ESP32-CSI-Tool firmware, Python receiver |
| Limitation | Metallic environments cause severe multipath — factories are among the hardest environments for WiFi CSI |

**Honest assessment:** This is the most exciting conceptually and the most difficult practically. A metallic factory floor is specifically the environment that degrades WiFi CSI reconstruction quality most severely due to reflections from machinery and rack surfaces. The primary viable use case is **zone-level occupancy** ("a human is present in sector 3") rather than object-level detail. This should be treated as a Phase 2 or research addition, not a core oracle dependency.

**Reference:** RF-NeRF (Zhao et al., arXiv:2309.08592, 2023) demonstrates WiFi-based NeRF reconstruction; however their environments are office/home, not factory floor. No published results for metallic industrial environments exist in the literature as of this writing.

---

### 2.3 Glinty Normal Distribution Functions — Constant-Time Shader

**Paper:** *Evaluating and Sampling Glinty NDFs in Constant Time* (Kemppinen, Paulin, Thonat, Thiery, Lehtinen, Boubekeur — Adobe Research / Aalto University, ACM ToG / SIGGRAPH Asia 2025)  
**Implementation:** https://www.shadertoy.com/view/tcdGDl (standalone GLSL fragment shader)  
**DOI:** https://dl.acm.org/doi/10.1145/3763282

#### Why this is relevant

Standard 3DGS uses spherical harmonics per Gaussian to encode view-dependent colour. SH are smooth basis functions — they can represent broad specular lobes but **cannot represent the high-frequency, stochastic glint pattern** of machined metal, brushed aluminium, chrome, or metallic powder coatings. The factory environment is dominated by exactly these surfaces.

The consequence for perception: when the VLM sees a hand approaching a metallic gripper, the appearance of the gripper changes dramatically between frames as specular highlights move with viewpoint. If our 3DGS scene model averages these into a smooth SH lobe, the rendered reference scene **does not match what the camera sees**, degrading the delta signal.

#### What the paper solves

The method represents micro-facet geometry between the micro and macro scales as a **4D point process on an implicit multiscale grid**. Finding all highlight-causing facets under a pixel is O(1) regardless of surface area — no precomputation, no memory overhead, zero storage cost beyond the standard BRDF parameters. Key properties:

- Converges to GGX/Beckmann NDF as surface area increases — compatible with existing material pipelines
- Supports importance sampling — can be used in path-traced (RTX Path Tracing) pipelines
- Supports per-facet colour — metallic flake effects (gold/silver particles in paint)
- Anisotropic support built in
- Provided as a **standalone GLSL fragment shader** — drop into any renderer

#### Integration path

For us, this is a **rendering layer addition** to the 3DGS digital twin. It affects how the reference scene *looks* when rendered, not how it is reconstructed. Concretely:

1. Tag Gaussians with material type during initial SHARP splat (or via a small material classifier on the RGB)
2. For Gaussians tagged `metallic`: replace SH evaluation with Glinty NDF shader evaluation
3. The digital twin now renders metallic surfaces with correct view-dependent glint behaviour
4. Delta detection against this reference is more accurate because the reference appearance model is physically correct

This is most valuable when the **rendered twin is shown to the VLM** (future work — current pipeline feeds camera frames, not rendered twin frames). Short-term, it primarily improves the visual quality of any human-facing dashboard or annotation tool.

---

### 2.4 NVIDIA RTX Neural Materials / Neural Shaders

**SDK:** RTX Neural Shaders — https://github.com/NVIDIA-RTX/Rtxns  
**RTX Neural Materials:** Not yet released (notify-me at developer.nvidia.com/rtx-kit)  
**Platform requirement:** DirectX Shader Model 6.9 / Cooperative Vectors; Tensor Core acceleration

#### What it is

RTX Neural Shaders is an SDK that allows trained neural networks to be **embedded inline inside HLSL/GLSL shaders**, running on Tensor Cores via the new DirectX Cooperative Vectors feature (Shader Model 6.9, Agility SDK 1.717 preview). This enables neural material representations — BSDFs encoded as small networks — to run at real-time performance.

RTX Neural Materials (forthcoming) specifically targets: glass with coatings, layered thin films, anisotropic metals, and complex multi-material blends. These are the surfaces that 3DGS and SH appearance models handle worst.

#### Relevance to 3DGS transparent surface problem

Standard 3DGS has no mechanism for refractive/transmissive appearance — it can approximate a transparent surface as a collection of translucent Gaussians but cannot faithfully represent IOR-based refraction, caustics, or wavelength-dependent transmission. This is the "transparent surfaces cause artefacts" limitation noted in the audit.

RTX Neural Materials addresses this at the **rendering layer** (how the model looks when rendered to screen) via a neural BSDF that can encode:
- Index of refraction
- Coating layer stacks (e.g. coated glass, anodised aluminium)
- Wavelength-dependent transmission spectra
- Roughness + scattering in a single unified representation

**Scope clarification:** This improves the **visual fidelity of the digital twin display** and the accuracy of any rendered reference frames fed to the VLM. It does not help a camera *see through* glass in a physical scene — that requires optics (polarisation filters, NIR to avoid reflections) not rendering.

#### Current availability on RTX 4060 Ti

RTX Neural Shaders SDK is available now (GitHub). It requires Shader Model 6.9 support — the 4060 Ti (Ada Lovelace) supports this via the DirectX 12 Agility SDK preview. RTX Neural Materials itself is not yet released. This component is **future-roadmap** rather than immediately actionable.

---

## 3. Unified Integration Architecture

### 3.1 Revised Oracle Layer — `PhysicsOracle` (proposed)

The proposed `PhysicsOracle` replaces the current two-class structure (SAM2Oracle + DepthOracle) with a four-layer fusion:

```
┌──────────────────────────────────────────────────────────────┐
│                   PhysicsOracle (proposed)                   │
├────────────────┬─────────────────────────────────────────────┤
│  Input         │  RGB frame (always) + optional sensors      │
├────────────────┼─────────────────────────────────────────────┤
│  Layer A       │  SHARP → current metric splat S_t           │
│                │  Delta vs S_ref → change regions            │
│                │  Fallback: S_ref = None → S_t only          │
├────────────────┼─────────────────────────────────────────────┤
│  Layer B       │  SAM2 (prompted on change regions only)     │
│                │  Point prompts derived from SHARP Gaussians │
│                │  → grounded mask labels in metric space     │
│                │  Fallback: auto-segment full frame          │
├────────────────┼─────────────────────────────────────────────┤
│  Layer C       │  Metric depth (RealSense) → clearances      │
│                │  Thermal (FLIR) → skin contact flag         │
│                │  Ultrasonic → proximity_cm                  │
│                │  Fallback: MiDaS monocular (relative)       │
├────────────────┼─────────────────────────────────────────────┤
│  Layer D       │  WiFi CSI → zone occupancy anomaly          │
│                │  Fallback: absent (no-op)                   │
├────────────────┼─────────────────────────────────────────────┤
│  Output        │  ConstraintReport (metric units, grounded)  │
│                │  oracle_text_block → VLM agent prompts      │
└────────────────┴─────────────────────────────────────────────┘
```

### 3.2 Revised `ConstraintReport` (proposed extension)

```python
@dataclass
class ConstraintReport:
    # Current fields (carry over unchanged)
    contact_area: float = 0.0           # gripper-object overlap IoU
    contact_delta: float = 0.0          # dIoU/dt
    centroid_velocity_div: float = 0.0  # hand vs gripper centroid divergence
    aspect_deformation: float = 0.0
    occlusion_ratio: float = 0.0
    vision_reliability: float = 1.0
    has_grip_break: bool = False
    has_velocity_spike: bool = False
    physics_score: float = 1.0

    # NEW: Metric fields (from RealSense or SHARP)
    contact_area_cm2: float = -1.0      # -1.0 = not available (metric unknown)
    min_clearance_cm: float = -1.0      # -1.0 = not available
    gripper_velocity_cms: float = -1.0  # -1.0 = not available

    # NEW: Supplementary sensor flags
    proximity_cm: float = float("inf")  # ultrasonic gripper mount
    thermal_hand_contact: bool | None = None  # None = sensor absent
    zone_occupancy_anomaly: bool = False     # WiFi CSI zone flag

    # NEW: Scene delta from SHARP
    scene_delta_magnitude: float = 0.0  # Hausdorff distance S_t vs S_ref
    scene_delta_available: bool = False # False if SHARP not running
```

### 3.3 Hardware Build Order (Suggested Prioritisation)

| Phase | Hardware | Cost | Oracle Impact |
|---|---|---|---|
| **Phase 1** (now) | Software only: SHARP + prompted SAM2 | £0 | Fixes label assignment; adds metric depth |
| **Phase 2** | Intel RealSense D455 | ~£300 | Metric clearance; replaces MiDaS; feeds SHARP alignment |
| **Phase 3** | HC-SR04 ultrasonic on gripper mount | ~£4 | Real-time proximity gate (<5ms) |
| **Phase 4** | FLIR Lepton 3.5 + PureThermal board | ~£90 | Human presence, thermal contact flag |
| **Phase 5** | ESP32 CSI nodes (2–3 units) | ~£40 | Zone occupancy; research-grade, may need custom AP firmware |

Phase 1 is **entirely within the existing software environment** — no new hardware required. SHARP runs on the RTX 4060 Ti. Prompted SAM2 (using SHARP Gaussian centroids as point prompts) uses the existing SAM2 checkpoint.

---

## 4. Open Questions for Team Discussion

These are the items that need a decision before any implementation begins:

### 4.1 SHARP licensing

Apple ML Research licence is non-commercial research use. If CLASP is ever a component of a commercial product (physical robot deployment, licensing to a factory operator), SHARP cannot be the backbone. Do we care about this now, or is this a post-research concern?

**Options:**
- A) Use SHARP for research/competition, replace with nerfstudio splatfacto (Apache 2.0) if commercialisation is planned
- B) Train our own feed-forward splat regressor on our own handoff data — highest quality, most effort, fully owned
- C) Use COLMAP + 3DGS for offline reference splat only; SHARP for live delta only

### 4.2 SHARP per-frame vs scheduled re-splat

SHARP takes <1s per frame. Our trajectories have ~25–60 frames. Running SHARP per-frame costs ~25–60s per trajectory in wall time — acceptable if batched, but it doubles the pipeline latency if sequential.

**Options:**
- A) Run SHARP on every frame (maximum freshness, highest compute)
- B) Run SHARP every N frames (e.g. N=5), interpolate delta signal between splats
- C) Run SHARP only on the reference scene; use traditional optical flow for live delta

### 4.3 Ultrasonic mounting

The HC-SR04 needs physical attachment to the robot gripper or a fixed mount near the handoff zone. This requires either cooperation with the robot hardware team or modification of the test rig. Is this in scope for current experiments?

### 4.4 Factory digital twin scope

The WiFi CSI / multi-room factory twin concept is architecturally distinct from the handoff oracle — it is a **separate system** that the oracle could eventually consume from. Should this be a separate project with its own document, or is it in scope for CLASP Phase 2+?

### 4.5 Neural Materials on 4060 Ti

RTX Neural Materials requires Shader Model 6.9 (DirectX Agility SDK preview). This is theoretically available on Ada Lovelace, but preview SDKs on Linux have historically had rough driver support. Has this been tested on our rig? If it causes driver instability, it is not worth pursuing until stable release.

---

## 5. What This Does Not Change

To be explicit — the following CLASP components are **not affected** by this proposal:

- Agent architecture (blind ensemble, P×T×M asymmetry matrix)
- Hyper-GRPO bandit policy
- LiveKV (Redis) + ArchiveKV (FAISS) dual-cache memory system
- SFT dataset serialisation pipeline
- NIM API / local inference toggle
- Consensus scoring and Life-Points mechanics
- Dashboard telemetry

This proposal only touches `clasp_pkg/oracle.py` and the `ConstraintReport` data structure. The oracle output (an `oracle_text_block` string injected into agent prompts) remains structurally identical — it is just sourced from better data.

---

## 6. References

| Item | Source |
|---|---|
| SHARP paper | Mescheder et al., arXiv:2512.10685 (Apple, 2025) |
| SHARP code | https://github.com/apple/ml-sharp |
| Glinty NDFs paper | Kemppinen et al., ACM ToG / SIGGRAPH Asia 2025 — https://dl.acm.org/doi/10.1145/3763282 |
| Glinty shader | https://www.shadertoy.com/view/tcdGDl |
| RTX Neural Shaders SDK | https://github.com/NVIDIA-RTX/Rtxns |
| RTX Kit overview | https://developer.nvidia.com/rtx-kit |
| RF-NeRF (WiFi CSI) | Zhao et al., arXiv:2309.08592 (2023) |
| Dynamic 3DGS | Wu et al., arXiv:2308.04079 (2024) |
| Intel RealSense D455 | https://www.intelrealsense.com/depth-camera-d455/ |
| FLIR Lepton 3.5 | https://www.flir.co.uk/products/lepton/ |
| ESP32-CSI-Tool | https://github.com/StevenMHernandez/ESP32-CSI-Tool |

---

*This document is a proposal. No code changes are implied until design decisions in §4 are resolved.*
