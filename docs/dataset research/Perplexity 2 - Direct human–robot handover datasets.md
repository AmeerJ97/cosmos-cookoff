<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Im looking for datasets that are human robot object transfer datasets as im registered in nvidia cookoff competition. human-robot handoff dataset OR human-robot object transfer dataset OR collaborative handover robotics real-world OR physical interaction human-robot passing object video OR sensor data handoff human robot real examples -synthetic -simulation -generated

Here are real-world (non-sim/synthetic) datasets and data sources that match your human–robot/object handoff query and are likely acceptable for the NVIDIA Cookoff.

## Direct human–robot handover datasets

- **FACT HRC / Collaborative Object Handover in a Robot Crafting Assistant**
    - Type: human–robot collaboration with many **handover episodes**.
    - Modality: RGB‑D (OAK‑D) upper‑body skeleton (25 keypoints), facial emotion estimates, full robot kinematics (base + arm joints, end‑effector goals).
    - Scale: 20 sessions, 565 handover episodes (human→robot, robot→human, bidirectional).
    - Use: learning policies for *when* to initiate handover and *where* to place the object (left/mid/right).
    - How to find: search for “Functional And Creative Tasks Human-Robot Collaboration FACT HRC dataset” or the paper title “Collaborative Object Handover in a Robot Crafting Assistant”.[^1]
- **Human-to-Robot Handover: RL-Based Control (2024)**
    - Type: human→robot handover experiments with a manipulator and anthropomorphic gripper.
    - Modality: real‑world robot state, control, and human interaction data (used for RL-based control).
    - Use: good if you want trajectories and reward structure for RL handover.
    - How to find: paper “Human-to-Robot Handover: Reinforcement Learning-Based Control” on PMC; check supplemental/links for data or contact authors.[^2]
- **H2RH-SGS (Learning human-to-robot handovers through 3D scene GS)**
    - Type: demonstrations of human→robot handovers, with both reconstructed scenes and **real‑robot experiments**.
    - Modality: RGB images, hand/object masks, target 6‑DoF gripper pose, pre‑grasp labels.
    - Caveat: demo data is partly generated from Gaussian Splatting reconstructions, but the *source* is real capture, and they validate on a physical robot.[^3]
    - How to find: search for “Learning human-to-robot handovers through 3D scene Gaussian Splatting H2RH-SGS dataset”; follow project page or supplementary materials.


## Human–human handover datasets useful for pretraining

- **MH2HO – Multi-sensor Dataset of Multiple Sequential Human‑to‑Human Object Handovers**
    - Type: human–human shelving/un‑shelving with many sequential handovers.
    - Modality:
        - 13 upper‑body bones for giver \& receiver,
        - 27 motion‑capture markers per person at 120 Hz,
        - dual RGB‑D streams at 30 Hz,
        - anthropometrics (height, arm span, etc.).
    - Scale: 12 pairs of participants, 30 handovers per trial, 1 440 handovers total.[^4]
    - Use: learning approach trajectories, timing, grasp/ungasp patterns, then transfer to robot.
    - How to find: Zenodo record “Multi-sensor Dataset of Multiple Sequential Human-to-Human Object Handovers in Shelving and Un-shelving Tasks (MH2HO)”.[^4]
- **3HANDS – Human–human object handover dataset**
    - Type: detailed human–human handover motions targeted at training **handover control** for supernumerary robotic limbs.
    - Modality: high‑quality human pose trajectories, timing and location of handover events.
    - Use: pretraining generative trajectory models, predicting handover location and initiation.[^5]
    - How to find: search “3HANDS dataset learning from humans for generating naturalistic handovers”.


## General human–object interaction datasets with handoff‑like motions

These are not dedicated handoff datasets but can help with perception (pose, object tracking, contact):

- **HUMOTO: A 4D Dataset of Mocap Human Object Interactions**
    - Type: high‑fidelity mocap + object motion for everyday activities.
    - Modality: full‑body motion capture, precisely modeled 3D objects and articulated parts.
    - Scale: 735+ sequences, 7 875 seconds at 30 fps, 63 objects, 72 articulated parts.[^6][^7]
    - Use: training perception or motion priors for reaching, grasping, and passing‑like interactions.
    - How to find: project page linked from the arXiv “HUMOTO: A 4D Dataset of Mocap Human Object Interactions”.[^7]
- **BEHAVE**
    - Type: full‑body human–object interaction dataset with accurate 3D reconstructions and RGB videos.
    - Modality: RGB, depth, 3D human–object pose and contact over time.
    - Use: perception models (detecting object pose in hand, estimating grasp state/contact) that you then apply in a handoff controller.[^8]


## Quick comparison for picking a dataset

| Dataset / source | Human–robot vs human–human | Sensors / data | Approx. scale | Real‑world vs sim |
| :-- | :-- | :-- | :-- | :-- |
| FACT HRC (crafting assistant) [^1] | Human–robot | RGB‑D skeleton, facial, robot kinematics | 20 sessions, 565 handovers | Real‑world |
| RL Human‑to‑Robot Handover [^2] | Human–robot | Robot control/state, interaction logs | Dozens of trials (see paper) | Real‑world |
| H2RH‑SGS demos [^3] | Human–robot | RGB, masks, 6‑DoF gripper target, labels | 16 objects, multiple demos | Real + reconstructed scenes |
| MH2HO [^4] | Human–human | Mocap (bones + markers), RGB‑D | 1 440 handovers | Real‑world |
| 3HANDS [^5] | Human–human | Detailed human pose trajectories | Many handover sequences | Real‑world |
| HUMOTO [^7][^6] | Human–object | Mocap + 3D objects | 735 sequences | Real‑world mocap |
| BEHAVE [^8] | Human–object | RGB‑D, 3D reconstructions | Hundreds of interactions | Real‑world |

If you tell me which sensing setup your Cookoff track expects (e.g., RGB‑D, only RGB, or also robot joint states), I can narrow this down to 1–2 best-fit datasets and suggest how to frame them to stay within the “no synthetic/simulated/generated” rules.
<span style="display:none">[^10][^9]</span>

<div align="center">⁂</div>

[^1]: https://www.themoonlight.io/en/review/collaborative-object-handover-in-a-robot-crafting-assistant

[^2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11479096/

[^3]: https://arxiv.org/html/2507.08726

[^4]: https://zenodo.org/records/7895500

[^5]: https://arxiv.org/html/2503.04635v1

[^6]: https://arxiv.org/html/2504.10414v1

[^7]: https://arxiv.org/abs/2504.10414

[^8]: https://www.youtube.com/watch?v=8SbNo4ePMGc

[^9]: https://personalrobotics.cs.washington.edu/publications/micelli2011handoff.pdf

[^10]: https://arxiv.org/html/2508.09855v1

