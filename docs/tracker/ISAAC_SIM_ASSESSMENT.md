# Isaac Sim Assessment for CLASP

## What Isaac Sim Offers

Isaac Sim is NVIDIA's robotics simulation platform built on Omniverse. For CLASP, the relevant capabilities are:

### 1. Synthetic Data Generation (CosmosWriter)
- Generates synchronized RGB + Depth + Segmentation + Edges from simulated scenes
- Output feeds directly into Cosmos Transfer for photorealistic augmentation
- Could generate thousands of synthetic handoff trajectories for training data

### 2. Physics Simulation
- PhysX-based rigid/soft body physics
- Realistic hand-object contact and grip simulation
- Ground-truth force, velocity, and contact data

### 3. Sensor Simulation
- Camera, LiDAR, IMU, contact sensors
- Can simulate thermal cameras (with custom extensions)
- Allows testing multi-modal sensing pipelines in simulation

### 4. Isaac Lab (RL Training)
- Pre-built manipulation environments
- RL policy training with parallel environments
- Sim-to-real transfer workflows

## Hardware Feasibility

**Minimum GPU:** RTX 3070 (8GB VRAM)
**Recommended:** RTX 4080 (16GB VRAM)
**Ideal:** RTX Ada 6000 (48GB VRAM)

**Our hardware:** RTX 4060 Ti 16GB
- MEETS recommended spec for basic scenes
- CANNOT run Isaac Sim + Cosmos-Reason2 inference simultaneously (both need VRAM)
- Would need to use Isaac Sim for data generation ONLY (offline), then switch to CLASP for inference

## Value Proposition for CLASP

### HIGH VALUE: Synthetic Training Data
- **Problem**: We need 500-1000+ handoff trajectories with ground-truth labels for SFT
- **Isaac Sim solves this**: Simulate diverse handoff scenarios programmatically
- Generate hundreds of trajectories with precise ground-truth:
  - Exact release frame (from physics simulation)
  - Contact forces, velocities, accelerations
  - Object pose, hand pose, grip state
  - Safe window defined by physics (not human labeling)
- Then augment with Cosmos Transfer for photorealism

### MEDIUM VALUE: Multi-Sensor Testing
- Test thermal + depth + RGB fusion in simulation before buying hardware
- Validate Physics Oracle against simulated ground-truth
- Prototype new modalities without physical sensors

### LOW VALUE: Direct Policy Training
- CLASP doesn't train a policy — it's a VLM-based POMDP evaluator
- Isaac Lab's RL training is for motor control policies, not VLM reasoning
- However, could generate VLM training data from RL rollouts

## Integration Architecture

```
Isaac Sim (offline, data generation)
  │
  ├── CosmosWriter → RGB + Depth + Seg clips
  │     └── Cosmos Transfer → Photorealistic augmentation
  │
  ├── Physics Ground Truth → Safe window labels
  │     ├── Contact forces → grip_stable threshold
  │     ├── Object velocity → release_safe threshold
  │     └── Hand pose → approach/transfer/release phases
  │
  └── Export → CLASP SFT Dataset (LLaVA format)
        ├── Video clips (RGB, 4 FPS)
        ├── Ground-truth labels (THINK/ACT per frame)
        └── Think traces (from Claude distillation on synthetic data)
```

## Practical Considerations

### Pros
- Solves the data scarcity problem (biggest bottleneck for SFT)
- Perfect ground-truth labels (physics simulation, not human annotation)
- Cosmos ecosystem integration (CosmosWriter → Transfer → Reason2 training)
- Impressive for Cookoff judges (full NVIDIA stack: Isaac Sim → Cosmos → NIM)

### Cons
- **Time investment**: Setting up a handoff scene in Isaac Sim takes days
- **Sim-to-real gap**: Simulated handoffs may not transfer perfectly to real video
- **VRAM conflict**: Can't run Isaac Sim and CLASP simultaneously on RTX 4060 Ti
- **Scope creep risk**: Could consume all remaining time if not scoped tightly
- **Not required for Cookoff**: Submission is already strong without it

## Recommendation

**FOR COOKOFF SUBMISSION (TODAY): NO**
- No time to set up Isaac Sim properly
- Current CLASP system is submission-ready without it

**FOR POST-COOKOFF TRAINING PHASE: YES, HIGH PRIORITY**
- Isaac Sim is the best path to generating the 500+ labeled trajectories needed for SFT
- Use CosmosWriter → Cosmos Transfer pipeline for photorealistic augmentation
- Generate ground-truth from physics simulation (eliminates manual labeling)
- Demonstrates full NVIDIA ecosystem integration (bonus for any future evaluation)

**Suggested timeline:**
- Week 1-2: Set up basic handoff scene in Isaac Sim (robot arm + object + human hand proxy)
- Week 2-3: Generate 500 synthetic trajectories with CosmosWriter
- Week 3: Augment with Cosmos Transfer for visual diversity
- Week 3-4: Convert to SFT dataset, run cloud training

## Resources
- [Isaac Sim CosmosWriter Tutorial](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_cosmos.html)
- [Isaac Sim Requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- [Isaac Lab Manipulation](https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/)
- [Cosmos Transfer Integration](https://nvidia-cosmos.github.io/cosmos-cookbook/)
