<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Im looking for datasets that are human robot object transfer datasets as im registered in nvidia cookoff competition. human-robot handoff dataset OR human-robot object transfer dataset OR collaborative handover robotics real-world OR physical interaction human-robot passing object video OR sensor data handoff human robot real examples -synthetic -simulation -generated

Here are real-world (non-sim/synthetic) datasets and data resources that are closest to what you need for human–robot object handoffs for the NVIDIA Cookoff.

## Direct human–robot handover datasets

- **Human-to-Robot Handover (RL control paper, 2024)** – Multi-sensor data collected on a manipulator with an anthropomorphic gripper for human‑to‑robot handovers; includes state/action trajectories suitable for learning policies or imitation.[^1]
    - Data: robot joint states, gripper state, force/torque and vision streams (see “Dataset and experimental setup” section).[^1]
    - Use: train or benchmark handover controllers, predict handover timing and approach poses.[^1]
- **RH20T – Contact-rich Robotic Manipulation Dataset** – Large-scale real robot teleoperation dataset with force‑torque, RGB‑D, and in‑hand cameras across many tasks; includes object passing and contact‑rich manipulation episodes.[^2]
    - Hardware: 6‑DoF arms with F/T sensors, gripper, 8–10 RGB‑D cameras, microphones.[^2]
    - Use: learn grasping, approach, contact strategies for handovers from real trajectories.[^2]
- **Robot-to-Human Construction Tool Handover Dataset (IAARC)** – Dataset of robot‑to‑human construction tool handovers with multi‑angle RGB‑D and annotated 6‑DoF grasp poses for tools with irregular shapes.[^3]
    - Data: multi‑view RGB‑D images, detailed grasp pose annotations, scene layouts for handover.[^3]
    - Use: grasp prediction, pose selection, safe approach for handing tools to humans.[^3]


## Human–human handover datasets (very useful for pretraining)

- **Bimanual Human-to-Human Object Handovers Dataset (2023)** – 240 recordings of bimanual handovers with motion capture and dual RGB‑D, designed explicitly to support human–robot and robot–human handover learning.[^4]
    - Data:
        - 13 upper‑body bones per person, 27 marker trajectories, object pose, all at 120 Hz.[^4]
        - Two RGB‑D streams at 30 Hz, phase labels (reach/transfer/retreat) for each participant.[^4]
    - Use: model human reach trajectories, predict handover points, pretrain policy or trajectory generators.[^4]
- **Multi-sensor Dataset of Human–Human Handovers (2018)** – >1000 recordings over 76 configurations with different roles/positions, built to inspire human‑robot handover methods.[^5]
    - Data: synchronized multi‑sensor streams (motion tracking, possibly force sensors, RGB‑D).[^5]
    - Use: recognize that a handover is occurring, estimate timing and spatial configuration.[^5]
- **3HANDS Dataset – Human–Human Object Handover (2022)** – Pose‑rich dataset used to train generative models for naturalistic handover motions and timing.[^6]
    - Data: detailed human pose trajectories for giver and receiver during handover.[^6]
    - Use: generate robot handover trajectories, predict handover location and when to initiate transfer.[^6]


### Small comparison table

| Dataset | Human vs robot | Modalities | Annotations / Focus |
| :-- | :-- | :-- | :-- |
| Human-to-Robot RL (2024) | Human–robot | Robot states, gripper, sensors | RL control for handover success/safety [^1] |
| RH20T | Human–robot | F/T, in-hand cams, 8–10 RGB‑D | Contact-rich manipulation, teleop demos [^2] |
| Construction Tool Handover | Robot–human | Multi‑view RGB‑D, grasp poses | 6‑DoF tool grasp and handover scenes [^3] |
| Bimanual H–H Dataset | Human–human | Mocap, 2×RGB‑D | Phases, object pose, bimanual dynamics [^4] |
| Multi-sensor H–H (2018) | Human–human | Multi‑sensor (mocap, vision, etc.) | Handover detection, configuration diversity [^5] |
| 3HANDS | Human–human | Pose trajectories | Naturalistic handover motion models [^6] |

## Datasets via “learning from human videos”

These are not packaged as classic handover datasets but give you pipelines and data assumptions if you want to repurpose human interaction videos:

- **HRT1 – One-Shot Human-to-Robot Trajectory Transfer** – AR‑captured RGB‑D human demos from a “robot’s eye” view, used to extract 3D hand/object trajectories and map to robot gripper paths.[^7]
    - Data: RGB‑D videos, camera poses, hand/object tracks from HoloLens‑like setup.[^7]
- **Object-centric 3D Motion Field from Human Videos** – Framework that uses general human–object interaction video dataset $\mathcal{D}_{human}$ plus SAM2 segmentation to build 3D motion fields for robot control.[^8]
    - Insight: you can take generic real videos of handovers and process them into 3D motion datasets.[^8]
- **H2RH-SGS – Learning Human-to-Robot Handovers through 3D Scene Gaussian Splatting** – Builds a simulated handover dataset from reconstructed real scenes, with each tuple containing an RGB image, masks, target 6‑D camera transform, and a binary pre‑grasp label.[^9]
    - Useful if you want to augment limited real handover data with photorealistic reconstructions.[^9]


## How to pick for the NVIDIA Cookoff

- If you need **direct robot trajectories and sensor data** for a physical robot: start with the Human‑to‑Robot RL handover data and RH20T, then add the construction tool handover dataset for diverse object geometry.[^3][^2][^1]
- If you want to **learn timing, intent, and handover pose prediction**: pretrain on the bimanual human‑human handover dataset, multi‑sensor human‑human handover dataset, and 3HANDS, then fine‑tune on the smaller human‑robot sets.[^6][^5][^4]
- If you plan a **video‑based controller**: follow the HRT1 and object‑centric 3D motion field setups to convert human videos into robot‑usable trajectories, optionally combining with RH20T or your own robot logs.[^7][^8][^2]

If you tell me your exact robot platform, sensing (just RGB‑D vs mocap/FT), and whether NVIDIA requires specific formats, I can filter this list down and suggest 1–2 “primary” datasets plus concrete preprocessing steps.
<span style="display:none">[^10]</span>

<div align="center">⁂</div>

[^1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11479096/

[^2]: https://rh20t.github.io

[^3]: https://www.iaarc.org/publications/fulltext/036_Robot-to-Human_Construction_Tool_Handover_Grasp_Prediction_for_6-DOF_Robotic_Arm_with_Parallel_Gripper.pdf

[^4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10294053/

[^5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6297854/

[^6]: https://arxiv.org/html/2503.04635v1

[^7]: https://arxiv.org/html/2510.21026v1

[^8]: https://arxiv.org/html/2506.04227v1

[^9]: https://arxiv.org/html/2507.08726

[^10]: https://scazlab.yale.edu/sites/default/files/files/ROMAN 2022 handovers.pdf

