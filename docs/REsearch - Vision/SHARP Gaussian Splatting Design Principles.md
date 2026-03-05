

## Design principles

You can treat SHARP’s Gaussians as a small, fixed set of learned “physical tokens” and inject them as a single compressed state vector, not as raw coordinates, so the VLM sees something analogous to visual prompt tokens rather than thousands of text numbers. [arxiv](https://arxiv.org/html/2403.13438v4)

- Use a fixed-size **state vector** per scene: pool your 3DGS into K–64 bins or object slots, then summarize each as a few scalars (centroid, extent, density, occupancy, dynamics) instead of per-point tokens. [ijcai](https://www.ijcai.org/proceedings/2025/1200.pdf)
- Align geometry to the camera: express centroids and gradients in a canonical, camera-centric frame to reduce positional variance, similar to SpatialPIN’s perspective canonicalization and 3D-aware prompting. [arxiv](https://arxiv.org/html/2509.13317v1)
- Map to a low‑entropy textual grammar: use a compact, templated “physical language” with fixed field names and short numeric strings so the LM learns consistent attention patterns, akin to region tokens and visual prompts. [github](https://github.com/VITA-Group/VLM-3R)
- Keep token budget small and stable: never exceed a fixed S tokens for the state; if you have more Gaussians, aggregate them into a slot (e.g., by semantic class or spatial voxel) and pool features. [arxiv](https://arxiv.org/html/2601.13132v1)

Concrete representation: you want something like “PHYS{…}” containing at most tens of scalars, not hundreds of lines of coordinates.

## From 3DGS to Physical State Vector

Assume SHARP gives you Gaussians \(G_i = (\mu_i \in \mathbb{R}^3, \Sigma_i, c_i, \alpha_i)\) for \(i=1..N\), plus possibly learned semantic labels or CLIP features. [arxiv](https://arxiv.org/abs/2411.18667)

1. Normalize and reparameterize  
   - Transform centroids into camera space: \(\hat{\mu}_i = T_{\text{world}\to\text{cam}} \mu_i\). [arxiv](https://arxiv.org/html/2403.13438v4)
   - Extract scalar shape features from \(\Sigma_i\): principal radii, anisotropy ratio, or log-volume.  
   - Use density/opacity statistics: raw \(\alpha_i\), local gradient norms \(\|\nabla \alpha\|\), and occupancy across space. [arxiv](https://arxiv.org/abs/2411.18667)  

2. Slotting / pooling (avoid token inflation)  
   - Partition space into K slots, e.g. by:
     - Semantic cluster (per object instance/label, if you have segmentation). [arxiv](https://arxiv.org/html/2601.13132v1)
     - Fixed 3D grid in camera frustum (e.g., 3×3×3 = 27 voxels).  
   - For each slot \(s\), aggregate Gaussians \(G_i\) in that region and compute:
     - Slot centroid: mean of \(\hat{\mu}_i\).  
     - Extents: max − min per axis, or pooled radii.  
     - Density features: mean \(\alpha\), max \(\alpha\), mean \(\|\nabla \alpha\|\).  
     - Mass: number of Gaussians, cumulative opacity.  

3. Global invariants  
   - Add a few scene-level scalars: average depth, free‑space ratio, number of occupied slots, gravity-aligned up axis, etc., mirroring how 3D reasoning frameworks summarize geometry before prompting VLMs. [ijcai](https://www.ijcai.org/proceedings/2025/1200.pdf)

Result: a state \(S \in \mathbb{R}^{K \times D} \to \mathbb{R}^{D'}\) after a small MLP or PCA that gives you, say, a 64–128‑dimensional compact **Physical State Vector**.

## Textual encoding strategy

You have two basic options to avoid attention drift:

- **Hard visual-like prompting**: map \(S\) through a learned linear layer into pseudo-visual tokens and splice them directly into Cosmos-Reason2’s visual stream (no extra text tokens; training-time change). [research.aalto](https://research.aalto.fi/fi/publications/nvsmask3d-hard-visual-prompting-withcamera-pose-interpolation-for/)
- **Geometric textual prompting** (what you asked for): serialize \(S\) into a single short, regular string; then prompt the VLM with this string as a special “PHYS” block, similar to SpatialPIN’s paragraphs of 3D descriptors but more compressed. [arxiv](https://arxiv.org/html/2509.13317v1)

Given you want a mathematically clean text-based interface, I’d do:

- One header line with global stats.  
- Then K fixed slots, each on a single line, with a rigid field order and minimal punctuation.  
- No natural-language fluff; keep field names fixed (e.g., “cx”, “cy”, “cz”, “dx”, “dy”, “dz”, “dens”, “grad”).  

Example conceptual format:

```text
[PHYS_STATE] ver=1 slots=8 cx_mean=0.02 cy_mean=0.15 cz_mean=3.10 depth_min=0.40 depth_max=7.80 free_ratio=0.35 mass=1240
S0 cx=0.10 cy=0.05 cz=1.80 dx=0.50 dy=0.40 dz=0.60 dens=0.82 grad=0.20 occ=0.12 sem=table
S1 cx=-0.40 cy=0.00 cz=2.50 dx=0.30 dy=0.35 dz=0.30 dens=0.60 grad=0.35 occ=0.08 sem=chair
...
[/PHYS_STATE]
```

This yields a small, bounded number of tokens but encodes centroids, extents, and density gradients in a form the LM can reuse across scenes. [arxiv](https://arxiv.org/html/2403.13438v4)

## Python schema for serialization

Below is a Python “schema” and serializer that takes SHARP-style Gaussians and returns such a string. You can plug in your actual 3DGS and Cosmos-Reason2 interfaces.

```python
from dataclasses import dataclass, asdict
from typing import List, Optional, Literal, Dict, Any
import numpy as np

CoordFrame = Literal["world", "camera"]

@dataclass
class GaussianSplat:
    # SHARP / 3DGS basic parameters
    mean: np.ndarray              # shape (3,), world coordinates
    cov: np.ndarray               # shape (3, 3)
    opacity: float                # alpha or density proxy
    color: Optional[np.ndarray] = None  # (3,) RGB, optional
    semantic_label: Optional[str] = None
    density_grad: Optional[np.ndarray] = None  # (3,) gradient of density

@dataclass
class SlotFeatures:
    cx: float
    cy: float
    cz: float
    dx: float
    dy: float
    dz: float
    dens: float
    grad: float
    occ: float
    sem: str

@dataclass
class PhysicalState:
    version: int
    coord_frame: CoordFrame
    slots: List[SlotFeatures]
    cx_mean: float
    cy_mean: float
    cz_mean: float
    depth_min: float
    depth_max: float
    free_ratio: float
    mass: float

def _transform_to_camera(means: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    # means: (N, 3), T_world_cam: (4, 4)
    N = means.shape[0]
    homog = np.concatenate([means, np.ones((N, 1))], axis=1)
    cam = (T_world_cam @ homog.T).T[:, :3]
    return cam

def _compute_slots(
    means_cam: np.ndarray,
    opacities: np.ndarray,
    density_grads: Optional[np.ndarray],
    semantics: List[Optional[str]],
    num_slots: int = 8
) -> List[SlotFeatures]:
    """Simple k-means-style partitioning into slots for compactness."""
    N = means_cam.shape[0]
    if N == 0:
        return []

    # k-means++ init (very lightweight)
    centers = [means_cam[np.random.randint(N)]]
    for _ in range(1, num_slots):
        d2 = np.min(np.linalg.norm(means_cam[:, None, :] - np.array(centers)[None, :, :], axis=-1)**2, axis=1)
        probs = d2 / (d2.sum() + 1e-8)
        idx = np.random.choice(N, p=probs)
        centers.append(means_cam[idx])
    centers = np.stack(centers, axis=0)  # (K, 3)

    # assign points
    dists = np.linalg.norm(means_cam[:, None, :] - centers[None, :, :], axis=-1)  # (N, K)
    assign = np.argmin(dists, axis=1)  # (N,)

    slots: List[SlotFeatures] = []
    for k in range(num_slots):
        mask = (assign == k)
        if not np.any(mask):
            # empty slot: encode as zero-volume, zero-density
            slots.append(SlotFeatures(
                cx=0.0, cy=0.0, cz=0.0,
                dx=0.0, dy=0.0, dz=0.0,
                dens=0.0, grad=0.0, occ=0.0,
                sem="none"
            ))
            continue

        pts = means_cam[mask]          # (M, 3)
        alphas = opacities[mask]       # (M,)
        if density_grads is not None:
            grads = density_grads[mask]    # (M, 3)
            grad_mag = float(np.mean(np.linalg.norm(grads, axis=-1)))
        else:
            grad_mag = 0.0

        cx, cy, cz = pts.mean(axis=0).tolist()
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        dx, dy, dz = (maxs - mins).tolist()

        dens = float(alphas.mean())
        # approximate occupancy as normalized total opacity
        occ = float(np.clip(alphas.sum() / (alphas.size + 1e-8), 0.0, 1.0))

        # majority semantic label if available
        labels = [sem for sem, m in zip(semantics, mask) if m and sem is not None]
        if labels:
            sem = max(set(labels), key=labels.count)
        else:
            sem = "unknown"

        slots.append(SlotFeatures(
            cx=float(cx), cy=float(cy), cz=float(cz),
            dx=float(dx), dy=float(dy), dz=float(dz),
            dens=dens, grad=grad_mag, occ=occ, sem=sem
        ))
    return slots

def build_physical_state(
    gaussians: List[GaussianSplat],
    T_world_cam: np.ndarray,
    num_slots: int = 8,
    free_space_estimate: float = 0.0
) -> PhysicalState:
    if len(gaussians) == 0:
        return PhysicalState(
            version=1,
            coord_frame="camera",
            slots=[],
            cx_mean=0.0, cy_mean=0.0, cz_mean=0.0,
            depth_min=0.0, depth_max=0.0,
            free_ratio=1.0,
            mass=0.0
        )

    means_world = np.stack([g.mean for g in gaussians], axis=0)        # (N, 3)
    means_cam = _transform_to_camera(means_world, T_world_cam)         # (N, 3)
    opacities = np.array([g.opacity for g in gaussians], dtype=float)  # (N,)
    semantics = [g.semantic_label for g in gaussians]

    if any(g.density_grad is not None for g in gaussians):
        grads = np.stack([
            g.density_grad if g.density_grad is not None else np.zeros(3)
            for g in gaussians
        ], axis=0)
    else:
        grads = None

    slots = _compute_slots(means_cam, opacities, grads, semantics, num_slots=num_slots)

    depths = means_cam[:, 2]
    cx_mean, cy_mean, cz_mean = means_cam.mean(axis=0).tolist()
    depth_min = float(depths.min())
    depth_max = float(depths.max())
    mass = float(opacities.sum())

    free_ratio = float(np.clip(free_space_estimate, 0.0, 1.0))

    return PhysicalState(
        version=1,
        coord_frame="camera",
        slots=slots,
        cx_mean=float(cx_mean),
        cy_mean=float(cy_mean),
        cz_mean=float(cz_mean),
        depth_min=depth_min,
        depth_max=depth_max,
        free_ratio=free_ratio,
        mass=mass
    )

def serialize_physical_state_to_string(state: PhysicalState) -> str:
    """Compact textual prompt block for injection into a VLM."""
    header = (
        f"[PHYS_STATE] ver={state.version} frame={state.coord_frame} "
        f"slots={len(state.slots)} "
        f"cx_mean={state.cx_mean:.3f} cy_mean={state.cy_mean:.3f} cz_mean={state.cz_mean:.3f} "
        f"depth_min={state.depth_min:.3f} depth_max={state.depth_max:.3f} "
        f"free_ratio={state.free_ratio:.3f} mass={state.mass:.3f}"
    )

    lines = [header]
    for idx, slot in enumerate(state.slots):
        lines.append(
            f"S{idx} "
            f"cx={slot.cx:.3f} cy={slot.cy:.3f} cz={slot.cz:.3f} "
            f"dx={slot.dx:.3f} dy={slot.dy:.3f} dz={slot.dz:.3f} "
            f"dens={slot.dens:.3f} grad={slot.grad:.3f} occ={slot.occ:.3f} "
            f"sem={slot.sem}"
        )

    lines.append("[/PHYS_STATE]")
    return "\n".join(lines)

# Example usage (pseudo-code, not executable as-is):
#
# sharps_gaussians = [...]  # List[GaussianSplat] from SHARP
# T_world_cam = ...         # 4x4 camera pose matrix
# phys_state = build_physical_state(sharps_gaussians, T_world_cam, num_slots=8)
# phys_string = serialize_physical_state_to_string(phys_state)
# 
# final_prompt = (
#   f"{phys_string}\n"
#   f"IMAGE: <cosmos-image-token>\n"
#   f"QUESTION: {user_question}\n"
# )
```

This schema:

- Encodes centroids and splat-density gradients via pooled slots so Cosmos-Reason2 sees a compact, consistent “physical header” for every scene.  
- Keeps token count bounded and low while maintaining geometric structure (positions, extents, densities, occupancy).  
- Resembles a geometric textual inversion / visual prompt tuning layer but remains entirely in the text channel, giving you a mathematically clean bridge between SHARP’s 3DGS and real-time VLM reasoning. [arxiv](https://arxiv.org/abs/2411.18667)
