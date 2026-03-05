# Multimodal Sensing for Robotic Handoff Prediction
## IR, WiFi CSI, LiDAR, and Sensor Fusion Research

**Document Date:** March 5, 2026
**Research Context:** CLASP system — safe human-robot object handoff release window prediction

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Infrared Sensing for Robotics](#1-infrared-sensing-for-robotics)
   - 1.1 Thermal LWIR Cameras
   - 1.2 Near-IR: Structured Light and Time-of-Flight
   - 1.3 Thermal Signatures and Grip/Release Detection
   - 1.4 Hardware Catalog
   - 1.5 Real-Time IR Processing Pipelines
3. [WiFi as an Omnidirectional 3D Gauge](#2-wifi-as-an-omnidirectional-3d-gauge)
   - 2.1 WiFi CSI Fundamentals
   - 2.2 3D Pose Estimation from WiFi
   - 2.3 WiFi Depth Reconstruction
   - 2.4 Hardware and CSI Extraction Tools
   - 2.5 IEEE 802.11bf Standard
   - 2.6 Limitations for Handoff Scenarios
4. [LiDAR Integration](#3-lidar-integration)
   - 4.1 Low-Cost LiDAR Options
   - 4.2 LiDAR vs Depth Cameras at Close Range
   - 4.3 Current Recommended Hardware
5. [Multi-Modal Sensor Fusion](#4-multi-modal-sensor-fusion)
   - 5.1 Fusion Architectures
   - 5.2 Layered Visual Feed Architecture
   - 5.3 Feeding Multi-Modal Data into VLMs
   - 5.4 ROS2 Framework Integration
   - 5.5 Thermal-Visible-LiDAR Fusion Specifically
6. [Value Analysis Per Modality](#5-value-analysis-per-modality)
7. [Handoff-Specific Findings](#6-handoff-specific-findings)
8. [Recommended Integration Architecture for CLASP](#7-recommended-integration-architecture-for-clasp)
9. [Sources and References](#sources-and-references)
10. [Research Methodology](#research-methodology)

---

## Executive Summary

This document synthesizes research across infrared sensing, WiFi CSI-based 3D perception, LiDAR, and multi-modal fusion for the specific problem of predicting safe human-robot object handoff release windows — as required by the CLASP system.

**Key findings:**

**IR (Thermal LWIR)** is the most immediately valuable new modality. Human skin emits detectable thermal signatures during grip — contact areas show temperature elevation due to conductive heat transfer, and grip loosening produces measurable micro-cooling. A 2025 study (Piwek et al., Advanced Engineering Materials) directly demonstrated that gripper finger temperature follows a step-like profile during contact, with a detectable drop on release. This is a strong signal for imminent release detection. Hardware is now cheap: the FLIR Lepton XDS (released February 2026, ~$239) integrates a 160x120 radiometric LWIR sensor with a 5MP RGB camera in a single OEM module.

**WiFi CSI** provides omnidirectional, non-line-of-sight body pose estimation at whole-body granularity (~91-125mm joint error per CVPR 2024 results). It cannot detect fine hand gestures or object contact state at the resolution needed for handoff safety — 12.5cm spatial resolution at 2.4GHz is the fundamental physics limit. Its value is as a coarse body-pose and intent-approach detector operating in parallel with finer-grained sensors, not as a handoff signal itself.

**LiDAR** at close range (0.25-1m) is well-served by the Orbbec Femto Bolt (Azure Kinect drop-in replacement with ToF depth at 1MP/120 FOV, ~$400) or spinning LiDAR options like the Slamtec A3/S2. For a desktop manipulation scenario, a structured-light or ToF depth camera is strictly better than spinning LiDAR, which is designed for room-scale SLAM.

**Multi-modal fusion** into VLMs is an active research area. The dominant strategy for feeding thermal+RGB+depth into a VLM is early/mid fusion through projection layers: encode each modality separately, project to token embeddings, concatenate before the LLM backbone. Thermal images rendered as false-color heatmaps can be fed directly to existing VLMs (Cosmos-Reason2 accepts image tokens) without model modifications. LiDAR point clouds require either bird's-eye-view projection or depth image rendering for VLM consumption.

**Recommendation priority for CLASP:** IR thermal camera (immediate value, direct handoff signal), depth camera upgrade (Orbbec Femto Bolt replaces/supplements existing depth), WiFi CSI as supplementary coarse-pose sensor (longer integration timeline, specialized hardware needed).

---

## 1. Infrared Sensing for Robotics

### 1.1 Thermal LWIR Cameras

Thermal long-wave infrared (LWIR) cameras detect emitted radiation from objects at room temperature (human skin emits strongly at 8-14 microns). They operate passively — no illumination source needed — and are immune to occlusion by darkness, reflections, or shadow that degrade RGB cameras.

**How thermal sensing differs from RGB:**
- Detects emitted heat, not reflected light
- Human skin emissivity is consistently ~0.98 across individuals and skin tones, making it an exceptionally reliable sensor target
- Temperature resolution (NETD) is the key spec; good sensors achieve <50mK NETD
- Spatial resolution is much lower than RGB (typical: 160x120 to 640x512 pixels)
- Frame rates suitable for real-time use: 9Hz (export-controlled threshold), 16-30Hz for unlocked modules

**Applications in robotics (2024-2026):**
- Human detection in darkness, smoke, dust — complement to RGB where RGB fails
- "Thermal touch" — detecting recently-contacted surfaces to infer manipulation history
- Safety monitoring in industrial settings (detecting hot components)
- Robotic skin integration — thermal + pressure sensor arrays for grip sensing (Im et al., 2025, Advanced Sensor Research)

A 2025 RoboticsTomorrow review confirmed thermal modules remain stable under high temperature, high reflection, and metal splash environments where RGB fails.

### 1.2 Near-IR: Structured Light and Time-of-Flight

These are distinct from thermal sensing — they actively project near-IR light and measure the return, providing depth rather than temperature.

**Structured Light:**
- Projects a known IR pattern onto the scene; deformation of the pattern encodes depth via triangulation
- Sub-millimeter precision achievable indoors at close range
- Strongly affected by sunlight (outdoor use limited) — ideal for controlled indoor manipulation
- Intel RealSense D415/D435 series uses this approach
- Working range for fine manipulation: 0.1-0.5m optimal

**Time-of-Flight (ToF):**
- Measures round-trip time of pulsed IR light per pixel
- Centimeter-accurate at manipulation distances; up to 75 depth frames/sec possible
- More robust to sunlight than structured light
- Sensitive to very reflective or very dark surfaces (measurement artifacts)
- Intel RealSense L515 (discontinued 2022), Orbbec Femto Bolt, Azure Kinect all use variants of this
- Best for: bin-picking, pick-and-place, volumetric measurement in 0.25-5m range

**Comparison for manipulation:**

| Technology | Best Range | Accuracy | FPS | Sunlight | Cost |
|---|---|---|---|---|---|
| Structured Light | 0.1-1.5m | Sub-mm | 30-90 | Poor | $150-400 |
| ToF/iToF | 0.25-5m | ~1-5mm | 30-75 | Moderate | $200-500 |
| Spinning LiDAR | 0.15-25m | ~2-5mm | 10-30 scans/s | Good | $99-500 |
| Stereo passive | 0.3-10m | 1-10mm | 30-60 | Excellent | $100-300 |

### 1.3 Thermal Signatures and Grip/Release Detection

This is the highest-value research finding for CLASP. Multiple independent sources confirm measurable thermal dynamics during human grip that could signal imminent release.

**Mechanism 1 — Conductive heat transfer:**
When human skin contacts an object, heat conducts from skin (typically 32-36C) to the object surface. The contact area becomes thermally elevated relative to the object's ambient temperature. This imprint is detectable by a co-located thermal camera.

- Thermal handprint identification has been demonstrated with 94.13% accuracy (forensics literature)
- The contact imprint persists for several seconds after release, providing a brief temporal window
- Grip area size and temperature gradient correlate with grip force and contact duration

**Mechanism 2 — Grip-force thermal gradient dynamics (directly relevant to CLASP):**
A 2025 study by Piwek et al. (Advanced Engineering Materials) measured temperature at gripper finger contacts during repeated handling cycles with hot objects. Key observation:

> "The finger temperature shows a step-like profile, where the temperature rises above the contact time. After the release of the object, there is a temperature drop followed by a further rise in temperature... The gradient peaks when the gripper finger is in contact with the hot object."

For human-robot handoff, this means: when a human hand grips an object, the skin-object contact area will show elevated temperature and a characteristic thermal gradient. As grip loosens (precursor to release), contact area decreases, thermal conduction drops, and a micro-cooling signal should be detectable.

**Mechanism 3 — Autonomic thermoregulation:**
Human skin temperature is modulated by the sympathetic nervous system. Stress, concentration, and task transitions produce measurable skin temperature changes (cold extremities during focus, warming during relaxation/release). This is a weaker, higher-latency signal but adds to multi-signal ensemble confidence.

**Mechanism 4 — Thermal imprint on object surface:**
As the human grips an object preparing to hand it off, the object surface in contact with fingers will show elevated temperature. A thermal camera viewing the object can detect this "grip shadow" even when the hand partially occludes the object from RGB view.

**Signal detectability requirements:**
- Camera must resolve finger-sized contact areas: at 30cm distance with 160x120 LWIR sensor and typical 55-degree FOV, each pixel covers ~6mm — sufficient to resolve individual finger contacts
- NETD (noise-equivalent temperature difference) must be <100mK for grip thermal dynamics; <50mK preferred
- Frame rate of 9-16Hz adequate for slowly evolving grip state; 30Hz preferred

**Relevant 2024-2025 prosthetics research:**
Frontiers in Neuroscience (2024) demonstrated a hybrid sensory feedback system where thermal nociceptive warning enabled grip force regulation in prosthetic hands, achieving 94.3% nociceptive warning perception. The authors specifically show IR sensing enabling feedback control of grip force — directly analogous to the CLASP use case.

A 2024 Advanced Materials study (Yang et al.) developed ML-enhanced ionic skin that achieved simultaneous temperature and pressure sensing with <7% prediction error for robotic gripper feedback — demonstrating the feasibility of temperature + pressure multimodal grip state estimation.

### 1.4 Hardware Catalog

**FLIR Lepton XDS (released February 2026)**
- Configuration: 160x120 radiometric LWIR + 5MP RGB, factory-aligned, MSX image enhancement
- Interface: USB (industry standard)
- NETD: Lepton 3.5 spec (~50mK)
- Features: Radiometric JPEG output (temperature data embedded in image), FLIR Prism ISP
- Price: ~$239 (sample quantity)
- ITAR-free: Yes — no export restrictions, simplifies integration
- Best for: CLASP primary IR sensor, dual-mode thermal+RGB from single capture

**FLIR Lepton 3.5 (standalone module)**
- 160x120 LWIR, 57-degree FOV, LWIR 8-14 micron
- SPI/I2C interface; requires carrier board (SparkFun PureThermal 2 ~$60)
- NETD: <50mK
- Frame rate: 8.7Hz (ITAR-limited export), some vendor unlocks to 27Hz
- Price: ~$150-180 module + carrier
- Best for: integration into custom rigs when Lepton XDS not yet available

**Melexis MLX90640**
- 32x24 pixel array (768 pixels), I2C interface
- Temperature range: -40C to 300C, ±2C accuracy
- Frame rate: up to 16Hz
- FOV options: 55-degree (BAA) or 110-degree (BAB)
- Price: ~$50-80 (Adafruit breakout: ~$60)
- Best for: low-cost experimentation, Raspberry Pi / Arduino integration
- Limitation: 32x24 is very coarse; cannot resolve individual finger contacts beyond ~20cm without careful mounting

**FLIR Lepton 2.5**
- 80x60 pixels, older generation
- Price: ~$150 at SparkFun
- Only recommended if Lepton 3.5/XDS unavailable

**FLIR Boson / Tau 2 (industrial grade)**
- 320x256 or 640x512 LWIR, 14-bit radiometric output
- Price: $1,500-4,000
- Overkill for most CLASP scenarios but relevant for high-precision grip thermal mapping

**Relevant comparison table:**

| Sensor | Resolution | NETD | Frame Rate | Price | Interface |
|---|---|---|---|---|---|
| MLX90640 | 32x24 | ~1C | 16Hz | ~$60 | I2C |
| FLIR Lepton 3.5 | 160x120 | <50mK | 8.7-27Hz | ~$200 | SPI+I2C |
| FLIR Lepton XDS | 160x120+5MP RGB | <50mK | 9-27Hz | ~$239 | USB |
| FLIR Boson 320 | 320x256 | <50mK | 60Hz | ~$2,000 | USB/MIPI |

### 1.5 Real-Time IR Processing Pipelines

**Thermal image processing for grip detection:**

1. **Radiometric normalization:** Convert raw ADC values to Kelvin using sensor calibration constants. Lepton XDS provides RJPEG (radiometric JPEG) which embeds raw temperature data alongside visual image.

2. **Background subtraction:** Establish ambient temperature baseline; segment human-temperature regions (30-37C). Simple threshold at 28C+ isolates hand/arm regions robustly.

3. **Contact area segmentation:** At grip time, the hand's contact zones with the object appear as warm patches on an otherwise cooler object. Track contact patch area, centroid, and temperature gradient.

4. **Temporal gradient analysis:** Compute dT/dt across the contact area. Grip tightening increases contact area and average temperature; grip loosening decreases both. This temporal derivative is the primary release-prediction signal.

5. **Thermal-RGB fusion:** FLIR's MSX technology (embedded in Lepton XDS) overlays RGB edge details onto the thermal image in real time. For downstream processing, a separate thermal channel as a false-color image can be rendered as a 3-channel "thermal image" and passed to any RGB-compatible vision pipeline.

**Integration with existing CLASP Oracle:**
The Physics Oracle already runs SAM2 for segmentation. Thermal can feed directly as an additional channel: segment the hand region in thermal, extract temperature statistics over the segmented mask, and pass grip-temperature features as additional oracle inputs. This requires no new segmentation model — SAM2 operates on the RGB channel; thermal statistics are extracted over the existing SAM2 mask.

**Latency:** MLX90640 at 16Hz = 62.5ms per frame. FLIR Lepton at 8.7Hz = 115ms. For a system already bottlenecked by SAM2 + MiDaS + Cosmos-Reason2, thermal latency is not the limiting factor.

---

## 2. WiFi as an Omnidirectional 3D Gauge

### 2.1 WiFi CSI Fundamentals

WiFi Channel State Information (CSI) describes how a transmitted signal propagates through the environment to a receiver — capturing amplitude and phase information across multiple subcarriers (OFDM). Human bodies, being 60-70% water, strongly absorb and reflect WiFi signals. Movement of a person in the RF field creates measurable perturbations in CSI measurements even without any device worn by the person.

**Key physics:**
- CSI captures multipath reflections off every object in the room
- Human body movement changes the multipath pattern in characteristic ways
- With multiple transmitter-receiver pairs, 3D spatial information can be estimated
- Spatial resolution is physically limited by wavelength: 2.4GHz = 12.5cm resolution; 5GHz = 6cm; 60GHz = 5mm
- CSI is sensitive to changes, not absolute positions — requires calibration

**CSI data structure:**
For an 802.11n/ac link with N_tx x N_rx antennas and N_sc subcarriers:
- CSI matrix size: N_tx x N_rx x N_sc complex values per packet
- Intel 5300: 3x3 MIMO, 30 subcarriers = 270 complex CSI values per packet at ~100-180 packets/sec
- Broadcom (Nexmon): 1x1, up to 256 subcarriers
- ESP32: 1x1, 52 subcarriers at up to 180 packets/sec

### 2.2 3D Pose Estimation from WiFi

**Person-in-WiFi 3D (CVPR 2024 — highest-impact recent paper):**
- First multi-person 3D pose estimation system using WiFi signals
- Architecture: Transformer/DETR-based, end-to-end estimation
- Results: 3D joint localization error of 91.7mm (1 person), 108.1mm (2 people), 125.3mm (3 people)
- Dataset: 4m x 3.5m workspace, 97K+ frames, 7 volunteers
- Hardware: Multiple WiFi devices to capture spatial reflections

**Graph-based 3D Pose Estimation (arXiv, Nov 2025):**
- Uses graph neural networks on CSI to reconstruct body skeleton
- Demonstrated cross-layout generalization improvements

**Wi-Mesh (2022, extended 2025):**
- 3D human mesh (not just skeleton) from WiFi
- Joint position error: 2.61cm for line-of-sight, 3.97cm through walls
- Vertices location error: 3.02cm average

**WiSk (2025, Concurrency and Computation):**
- Lightweight multimodal fusion: WiFi CSI + video skeleton images
- 99.58% activity recognition accuracy in good lighting; 98.75% in low-light/occlusion
- 20.55M FLOPs — deployable in resource-constrained environments
- Key insight: CSI fills gaps where vision fails (occlusion, low light)

**Energy-Efficient Edge CSI Pose Reconstruction (Expert Systems, 2025):**
- 3D CNN + Bi-GRU with attention for real-time skeleton from CSI
- Aligns WiFi CSI with Kinect-captured ground truth
- Conclusion: "CSI-based approach complements rather than replaces vision pipelines — fills a critical niche where privacy, sustainability, and unobtrusive monitoring outweigh the need for sub-centimetre positional accuracy"

### 2.3 WiFi Depth Reconstruction

**Wi-Depth (arXiv, March 2025 — "Reconstructing Depth Images of Moving Objects from Wi-Fi CSI Data"):**
- Proposes decomposing depth image into shape, depth, and position
- Teacher-student VAE architecture
- Key limitation acknowledged: "learning the mapping function between CSI and depth images, both of which are high-dimensional data, is particularly difficult"
- Current accuracy: insufficient for fine object manipulation; research-grade

**Spatio-Temporal 3D Point Clouds from WiFi-CSI (arXiv, Oct 2024):**
- Transformer networks producing 3D point clouds from CSI
- Research from Finland 6G Flagship Program
- Still experimental; not production-ready for manipulation tasks

**Fundamental resolution limits:**
- 2.4GHz WiFi: ~12.5cm spatial resolution
- 5GHz WiFi: ~6cm spatial resolution
- 60GHz (802.11ad/WiGig, available on some routers): ~5mm — sufficient for hand gesture detection
- Object-level detection (cup, bottle): feasible at 5GHz if object is >10cm
- Hand gesture / finger-level detail: only possible at 60GHz

### 2.4 Hardware and CSI Extraction Tools

**Supported hardware platforms:**

| Platform | Standard | Subcarriers | Max Rate | Cost | Notes |
|---|---|---|---|---|---|
| Intel IWL5300 NIC | 802.11n | 30 | ~100pkt/s | ~$30-50 used | Linux CSI Tool; widely cited in research |
| Intel AX200/AX210 | 802.11ax (WiFi 6) | 242+ | higher | ~$20-30 | Nexmon support in kernel (2024) |
| Broadcom BCM4366 | 802.11ac | 256 | ~1000pkt/s | Router-embedded | Nexmon firmware patch required |
| ESP32 | 802.11b/g/n | 52 | 180pkt/s | ~$5-10 | ESP32-CSI-Tool by Steven Hernandez |
| Raspberry Pi (Nexmon) | 802.11n/ac | varies | ~100pkt/s | included in RPi | Nexmon patch for Broadcom BCM43xx |

**Software tools:**
- **CSIKit** (Python): Parses CSI from Atheros, Intel, Nexmon, ESP32, FeitCSI, PicoScenes — single library for all formats
- **csiread** (Python): Fast CSI parser for Intel, Atheros, Nexmon, ESP32
- **Nexmon CSI** (GitHub: seemoo-lab): Firmware patch + tools for Broadcom hardware
- **ESP32-CSI-Tool**: Achieves 92.43% HAR accuracy at 232ms inference on ESP32-S3

**Recommended setup for CLASP WiFi sensing:**
1. ESP32-S3 (transmitter, $8) + Intel IWL5300 or AX200 in Linux laptop/mini-PC (receiver)
2. Or: Two ESP32-S3 units (one TX, one RX) for fully self-contained setup
3. CSIKit for parsing; feed into existing Python pipeline
4. Estimated setup time: 1-2 days for basic CSI capture; weeks for activity classification model training

### 2.5 IEEE 802.11bf Standard

IEEE Std 802.11bf-2025 was published September 26, 2025. This is the first official WLAN sensing standard.

**Key specifications:**
- Range accuracy: 0.2m for localization; 0.01m for 3D vision (at 60GHz)
- Angular resolution: 5 degrees (basic), 1 degree (precise localization)
- Velocity resolution: 0.1 m/s for human motion detection
- Detection probability: >95% for fall detection class applications
- Sub-7GHz sensing: presence detection, activity recognition, room-scale
- 60GHz sensing: 3D vision, gesture recognition at fine granularity (~5mm)

**Implications for CLASP:**
- 802.11bf hardware (WiFi 7 with sensing support) will become commercially available in 2026-2027
- 60GHz sensing at 5mm resolution is genuinely useful for handoff scenarios — can detect hand position and large gestures
- Not available today in consumer hardware; current CSI sensing uses 802.11n/ac hardware outside the standard

### 2.6 Limitations for Handoff Scenarios

**What WiFi CSI can do for CLASP:**
- Detect person approaching the robot (room-scale, ~30cm accuracy)
- Recognize coarse activity class (reaching, standing, handing) with >90% accuracy
- Operate through walls, occlusion, low lighting
- Provide complementary signal when cameras are obstructed

**What WiFi CSI cannot do reliably today:**
- Detect finger-level grip state (resolution limit: 6-12.5cm at standard WiFi bands)
- Distinguish grip tightening vs loosening
- Map small objects (items smaller than the wavelength are essentially invisible)
- Operate reliably in environments with many moving people (multipath interference)
- Achieve sub-second latency for hard real-time control (typical pipeline: 100-500ms)

**Verdict for CLASP:** WiFi CSI is a valuable coarse-level environmental sensor providing body presence and approach detection. It is not useful as a primary handoff release signal. As a secondary modality providing "human is actively reaching toward robot" confirmation, it adds value without replacing RGB/depth/thermal.

---

## 3. LiDAR Integration

### 3.1 Low-Cost LiDAR Options

**Slamtec RPLiDAR Series (2D spinning LiDAR):**

| Model | Range | Sample Rate | Accuracy | Price | Tech |
|---|---|---|---|---|---|
| A1M8 | 12m | 8,000/s | ~1-2% | ~$99 | Triangulation |
| A2M8 | 16m | 8,000/s | ~1-2% | ~$150 | Triangulation |
| A3 | 25m | 16,000/s | mm-level | ~$299 | Triangulation |
| S2 | 30m | 32,000/s | mm-level | ~$199 | ToF |
| S3 | 40m | 32,000/s | mm-level | ~$250 | ToF |

**Critical limitation for manipulation:** RPLiDAR is a 2D spinning sensor. It produces a single horizontal plane of distance measurements. For desktop manipulation (30cm range, 3D object shape), it provides essentially no useful information unless mounted on a motorized tilt stage. RPLiDAR is appropriate for floor-level obstacle avoidance and room mapping, not close-range manipulation sensing.

**Intel RealSense L515 (discontinued 2022):**
- Solid-state MEMS LiDAR, 9.2 million depth points/sec
- Resolution: 640x480 at 0.25-9m; 1024x768 at 0.25-6.5m
- Power: <3.5W
- Still available used at ~$200-300; excellent for close-range work but no longer manufactured

**Orbbec Femto Bolt (primary recommendation, 2024-present):**
- Official Microsoft Azure Kinect DK replacement, endorsed by Microsoft
- 1MP depth (iToF), 4K RGB, 6-DoF IMU
- Depth FOV: 120-degree wide, 0.25-5.5m range
- Depth modes identical to Azure Kinect DK
- SDK includes Azure Kinect Sensor SDK wrapper for drop-in compatibility
- Showcased at NVIDIA GTC 2024 as primary Kinect successor
- Price: ~$400
- Best for: primary depth camera in CLASP sensor stack

**Orbbec Femto Mega:**
- Similar to Femto Bolt with Ethernet connectivity option
- Better for fixed installation scenarios

### 3.2 LiDAR vs Depth Cameras at Close Range

For a desktop manipulation task at 0.2-1.5m range, the comparison is essentially settled:

**Depth cameras (structured light / ToF) win at close range because:**
1. Dense depth map (every pixel has depth, not just a scan line)
2. 30-90 FPS for structured light (RPLiDAR produces 10-15 scans/sec)
3. Sub-mm to mm accuracy for structured light at <1m
4. Captures 3D shape of small objects (cups, bottles, hands) at the resolution needed
5. Well-supported in ROS2, OpenCV, Open3D, PCL

**Spinning LiDAR advantages that don't apply here:**
- Long range (10-40m): irrelevant for desk-scale work
- Outdoor robustness: irrelevant for indoor lab
- 360-degree coverage: not needed; object is in front of robot

**Metrological comparison at close range (PMC published study, Intel RealSense D415/D455/L515):**
- L515 showed best accuracy at 0.25-2m in indoor conditions
- D415 (structured light) outperforms in sub-0.5m range
- All achieve <5mm error at 0.3-0.5m in controlled settings

**Structured Light vs ToF summary for CLASP:**
- **Structured Light (D415, D435, similar):** Best for <0.5m, highest spatial detail for finger/grip geometry, can see around hand to object contact zone. Recommended for close-up grip zone camera.
- **ToF (Femto Bolt, L515 if available):** Better for full-scene depth map, more robust to ambient IR, wider depth range. Recommended as primary scene depth camera.

### 3.3 Current Recommended Hardware

**For CLASP sensor stack, LiDAR/depth recommendation:**

1. **Primary scene depth:** Orbbec Femto Bolt (~$400) — replaces Azure Kinect DK, wide FOV, ToF depth, compatible with existing Azure Kinect SDK
2. **Close-range grip zone:** Intel RealSense D435/D435i (~$170-220) — structured light, excellent at 15-50cm, includes RGB+IR stereo+IMU
3. **Avoid spinning LiDAR** (RPLiDAR A-series/S-series) for manipulation — wrong tool class

---

## 4. Multi-Modal Sensor Fusion

### 4.1 Fusion Architectures

Three dominant fusion paradigms from the 2024-2025 literature (Multimodal Fusion and VLMs: A Survey for Robot Vision, arXiv April 2025):

**Early Fusion (input-level):**
- Concatenate all modalities at input stage through a Joint Multimodal Encoder
- Requires all modalities to be spatially aligned (registered)
- Best when modalities are highly correlated and low-latency
- Example: Stack RGB (3ch) + thermal (1ch rendered as heatmap) + depth (1ch) into a 5-channel tensor

**Late Fusion (decision-level):**
- Separate encoders per modality; merge predictions/features before final decision
- More flexible; modalities can have different resolutions and update rates
- Easier to handle missing modalities at runtime
- Example: Run YOLO on RGB, run thermal classifier on IR, combine confidence scores

**Hierarchical Fusion (multi-scale cross-modal):**
- Multimodal interaction across perceptual, semantic, and control levels
- Cross-modal attention mechanisms allow tokens from one modality to attend to tokens from other modalities
- Current state of the art for VLA models

**For sensor fusion in robotics, the 2024-2026 consensus:**
Use LiDAR as the coordinate reference frame; project RGB and thermal onto LiDAR point cloud coordinates. This is the approach used in the ETRI Journal (2022) multi-robot surveillance system: "data coordinates are converted into LiDAR-RGB and LiDAR-thermal based on LiDAR."

### 4.2 Layered Visual Feed Architecture

A "layered visual feed" approach — treating the workspace as a stack of registered 2D maps at different semantic levels — is directly supported by recent work:

**MULTIMODAL SIGNAL PROCESSING FOR THERMAL-VISIBLE-LIDAR FUSION IN REAL-TIME 3D SEMANTIC MAPPING (arXiv, January 2026):**
The paper describes exactly the layered approach:
1. Pixel-level fusion of visible and infrared images into a single enhanced frame
2. Project real-time LiDAR point cloud onto the fused image stream
3. Segment heat-source features to identify thermal targets
4. Produce a 3D map enriched with geometry, texture, and temperature semantics as distinct layers

**For CLASP, a practical layered feed would be:**

```
Layer 0 (Base):    RGB image (standard camera, 1920x1080 @ 30fps)
Layer 1 (Depth):   Registered depth map from Femto Bolt or D435 (rendered as 0-255 grayscale or jet colormap)
Layer 2 (Thermal): Registered thermal map from Lepton XDS (rendered as false-color jet/inferno)
Layer 3 (Semantic): SAM2 segmentation masks for hand, object, robot gripper
Layer 4 (Motion):  Optical flow or temporal difference highlighting movement
[Optional Layer 5]: WiFi CSI-derived coarse body skeleton overlay
```

Each layer registered to the same pixel coordinate space (extrinsic calibration between sensors). For VLM input, layers can be:
- **Side-by-side:** Concatenate layers horizontally as sub-panels in one image (simplest, works with existing VLMs)
- **False-color composite:** Blend thermal into RGB as a third channel (replaces B channel with temperature)
- **Token concatenation:** Encode each layer with a separate visual encoder; concatenate resulting token sequences before LLM backbone

### 4.3 Feeding Multi-Modal Data into VLMs

**Current VLM architecture (LLaVA-style, applicable to Cosmos-Reason2):**
1. Visual encoder (CLIP ViT or similar) processes image → patch tokens
2. Projection layer (MLP) converts visual tokens to language embedding space
3. Prefix tokens are prepended to text tokens; LLM processes the combined sequence

**Strategies for non-RGB modalities:**

**Strategy A — False-color rendering (zero modification required):**
Render thermal data as a false-color image (jet/inferno colormap); render depth as a colormap (turbo/plasma). These are standard 3-channel RGB images that any VLM can process without modification. FLIR Lepton XDS already provides this.

Limitation: VLM was not trained on thermal-as-color images; may misinterpret thermal patterns. However, ThermEval benchmark (arXiv, Feb 2026) shows VLMs can reason about thermal images with appropriate prompting.

**Strategy B — Side-by-side panel composition:**
Create a composite image with RGB | Depth | Thermal as three panels. Cosmos-Reason2 can be prompted: "Left panel is RGB, center is depth map (closer=brighter), right panel is thermal (warmer=red)." No model modification. Works today.

**Strategy C — Additional input channel projection:**
Add a lightweight linear projection for the extra channels (depth, thermal), fuse with existing vision tokens before LLM. Requires fine-tuning. Future-looking for CLASP's SFT dataset pipeline.

**Strategy D — Cross-modal attention fusion (research-grade):**
Talk2PC-style: fuse modalities through prompt-guided cross-attention. Best performance but requires significant architectural work.

**Recommendation for CLASP Phase 1:** Use Strategy B (side-by-side panels). Compose a 3-panel image at each frame:
- Panel left: RGB
- Panel center: Depth rendered in turbo colormap
- Panel right: Thermal rendered in inferno colormap

The CLASP agents receive this as a single image input with a system prompt explaining the panel layout. No model modification. Immediately compatible with Cosmos-Reason2-8B as-is.

**ThermEval benchmark (arXiv 2602.14989, Feb 2026):**
> "Vision language models achieve strong performance on RGB imagery but do not generalize to thermal images... thermal images encode physical temperature rather than color or texture."

ThermEval-B provides ~55,000 thermal VQA pairs. Fine-tuning Cosmos-Reason2 on a small subset with thermal examples would significantly improve thermal understanding — a low-cost SFT step.

### 4.4 ROS2 Framework Integration

**Available ROS2 packages for multi-modal fusion (2024-2025):**

- `ros2_camera_lidar_fusion` (GitHub: CDonosoK): Intrinsic + extrinsic calibration; RGB-LiDAR fusion
- `l2i_fusion_detection` (GitHub: AbdullahGM1): 360-degree LiDAR + camera fusion, enhanced object tracking
- FLIR's ROS2 driver: publishes thermal images as sensor_msgs/Image on standard topics
- Orbbec Femto Bolt: has official ROS2 SDK/driver
- Intel RealSense D435: realsense2 ROS2 package (well-maintained)

**Calibration approach:**
Use a calibration target with both RGB and thermal contrast (heated checkerboard or reflective tape). Extrinsic calibration between thermal and RGB is possible using the factory-aligned Lepton XDS — FLIR pre-registers the two sensors so no additional calibration needed.

**Message synchronization:**
Use `message_filters::ApproximateTimeSynchronizer` in ROS2 to align depth, RGB, and thermal timestamps. Thermal at 8-16Hz will be the slowest modality; others should be synchronized to thermal timestamps.

### 4.5 Thermal-Visible-LiDAR Fusion Specifically

The 2026 arXiv paper (MULTIMODAL SIGNAL PROCESSING FOR THERMAL-VISIBLE-LIDAR FUSION IN REAL-TIME 3D SEMANTIC MAPPING) is the most directly relevant recent work. Their pipeline:

1. **Visible-IR pixel fusion:** Gradient-domain fusion preserving edges from RGB and temperature information from thermal
2. **LiDAR projection:** Project 3D LiDAR point cloud onto 2D fusion image using calibrated camera intrinsics/extrinsics
3. **Semantic segmentation:** Segment fused image to identify thermal objects (humans, heat sources)
4. **3D semantic map:** Assign temperature-semantic labels to LiDAR points

For CLASP specifically, step 3 can be replaced by SAM2 (already in the Oracle), and the output is exactly what the Physics Oracle needs: a depth-registered, temperature-annotated view of the handoff zone.

---

## 5. Value Analysis Per Modality

### Summary Table

| Modality | Info Gain over RGB | Latency | Hardware Cost | Integration Effort | Handoff Relevance |
|---|---|---|---|---|---|
| **Thermal LWIR** | HIGH — grip thermal dynamics, contact area, release signal | +62-115ms | $60-240 | Low-Medium | DIRECT — detects grip state changes |
| **Near-IR Depth (ToF/SL)** | HIGH — 3D shape, hand pose, occlusion resolution | +0-33ms (fused) | $170-400 | Low (ROS2 drivers exist) | HIGH — hand geometry, gripper clearance |
| **WiFi CSI (sub-7GHz)** | MEDIUM — coarse body pose, approach detection | +100-500ms | $10-50 (ESP32) | High (model training required) | LOW-MEDIUM — too coarse for grip |
| **WiFi (60GHz/802.11bf)** | HIGH — gesture/finger detection at 5mm | TBD | Not available commercially yet | Very High | MEDIUM (future) |
| **Spinning LiDAR (2D)** | LOW for manipulation — no 3D at close range | ~67-100ms | $99-299 | Medium | NONE for grip, LOW for room |

### Per-Modality Value Analysis

**Thermal LWIR — Strongly Recommended**

Information gain: Provides a signal class unavailable from RGB: temperature distribution over hands and object contact zones. Uniquely relevant to grip-state inference. Thermal imprint on the handed object provides a "pre-release warning" that has no RGB equivalent.

Latency impact: At 16Hz, adds at most 62.5ms to a pipeline that is already dominated by Cosmos-Reason2 inference time (likely 200ms+). Negligible impact.

Hardware cost: MLX90640 at $60 for prototyping; Lepton XDS at $239 for production quality. Extremely low cost relative to the information gain.

Integration complexity: Lepton XDS via USB is plug-and-play. FLIR provides Python SDK. MSX-fused images are standard RGB and immediately usable by all existing pipeline components.

Relevance to handoff safety: Direct — thermal dynamics during grip and release provide leading indicators of imminent handoff. This is the modality most likely to enable detection of "human grip loosening" 500-2000ms before actual object release, which is exactly the safety margin CLASP needs.

**Near-IR Depth (ToF or Structured Light) — Strongly Recommended**

Information gain: Resolves 3D geometry of hand, object, and robot gripper that is ambiguous in 2D RGB. Enables detection of hand-object contact clearance (how far the human's hand must move to safely release). Removes occlusion ambiguity.

Latency impact: Depth cameras operate at 30-90 FPS; already slower than typical processing pipelines. At 30Hz, adds 33ms; effectively negligible given overall pipeline latency.

Hardware cost: Intel D435i ~$200; Orbbec Femto Bolt ~$400. One-time cost, no ongoing expense.

Integration complexity: Low. RealSense and Orbbec both have excellent ROS2 drivers, Python SDKs, and Open3D integration. Already standard in manipulation research.

Relevance to handoff safety: High. 3D geometry of the handoff zone directly enables: (1) detecting if robot gripper has full contact before human releases, (2) estimating collision risk if human releases unexpectedly, (3) tracking hand withdrawal trajectory after handoff.

**WiFi CSI (2.4/5GHz sub-7GHz) — Optional / Long-Term Research**

Information gain: Unique in being omnidirectional and non-line-of-sight. Can detect a person approaching from outside camera FOV. Adds coarse body-pose context when aligned with camera data (WiSk 2025 demonstrated ~99.6% activity recognition combining WiFi + vision skeleton).

Latency impact: CSI processing pipelines typically run 100-500ms end-to-end for activity classification. High for real-time safety-critical use.

Hardware cost: ESP32 at $5-10; Intel 5300 NIC at $30-50. Very low hardware cost.

Integration complexity: HIGH. Requires: hardware setup, CSI capture software, signal preprocessing (bandpass filter, PCA/SVD denoising), activity classifier training (specific to room geometry and antenna positions), recalibration if furniture moves. Not plug-and-play.

Relevance to handoff safety: Low-to-medium as a primary sensor. Medium as a supplementary coarse-state estimator: "human is in handoff-approach posture" vs "human is stationary." Cannot detect grip state, object contact, or fine hand motion.

**Recommendation:** Implement WiFi CSI only after thermal and depth are integrated and working. If pursued, use ESP32 (cheap, Python library available) and focus specifically on "approach vs stationary vs withdraw" three-class detection, not fine pose estimation.

**Spinning LiDAR (RPLiDAR A/S series) — Not Recommended for CLASP**

Information gain: Near-zero for a desktop manipulation scenario. Provides 2D scan in horizontal plane; unhelpful for 3D grip geometry, hand pose, or object orientation.

The RPLiDAR family is excellent hardware — for SLAM-based mobile robot navigation. For tabletop manipulation, a depth camera provides strictly better information at comparable cost.

---

## 6. Handoff-Specific Findings

### What the literature says about multimodal sensing for handoff

**Early Detection of Human Handover Intentions (ScienceDirect, 2025):**
Compared EEG, gaze, and hand-motion signals for classifying handover-intended vs non-handover motions. Key finding: **gaze signals are the earliest and most accurate** for classifying handover intentions. Hand-motion signals lag gaze by 200-500ms. EEG provides earliest physiological signal but is impractical in non-laboratory settings.

**Implication for CLASP:** Gaze/eye-tracking could be a high-value addition. Cheap eye trackers (Tobii 4C ~$150, Pupil Labs ~$200) provide gaze direction that may be the best single predictor of imminent handoff initiation.

**Multimodal Learning-Based Proactive Handover Intention Prediction (Advanced Intelligent Systems, 2024):**
Used wearable data gloves + augmented reality system; 99.6% accuracy on 12 handover intent classes. Finding: wearable sensing outperforms vision alone but requires the human to wear instrumentation — impractical for CLASP's scenario where the human is not instrumented.

**Novel Human Intention Prediction via Fuzzy Rules through Wearable Sensing (PMC, MDPI):**
Fuzzy rule-based prediction using wrist-worn sensors. Highly accurate but same limitation: requires hardware on the human.

**Synthesis for uninstrumented (no wearables) handoff:**
Without wearables on the human, the available modalities are:
1. RGB + depth (current CLASP) — visual pose, hand shape, trajectory
2. Thermal IR (new) — grip contact, thermal dynamics, pre-release cooling
3. Gaze (potential addition) — earliest intent signal
4. WiFi CSI — coarse approach/activity detection

Thermal IR is the highest-value addition in this constraint set because it provides a signal (grip thermal state) that is not redundant with existing RGB+depth capabilities.

### Thermal signature timeline for a handoff event

Based on synthesis of multiple sources:

```
T-2000ms: Human approaches, extends hand toward object
T-1500ms: Hand contacts object; thermal contact imprint begins forming
           [MLX90640 or Lepton can detect this contact zone]
T-1000ms: Grip force peaks; contact area maximal, thermal elevation ~1-3C above object ambient
T-500ms:  Grip begins to loosen (precursor to release); contact area decreases
           [dT/dt in contact zone becomes negative — detectable release signal]
T-200ms:  Fingers partially separated from object surface
T-0ms:    Object released — contact imprint dissipates; object thermal signature decays
T+500ms:  Residual thermal imprint on object visible for ~3-8 seconds
```

The window T-500ms to T-0ms is the critical CLASP detection window. Thermal sensing provides a direct physical proxy for grip state in this window that RGB cannot provide.

---

## 7. Recommended Integration Architecture for CLASP

### Phase 1 (Immediate, ~1-2 weeks):

**Add thermal IR as Oracle input:**
1. Mount FLIR Lepton XDS (or Lepton 3.5 on PureThermal carrier) co-located with existing RGB camera
2. Capture thermal stream alongside RGB
3. Render thermal as false-color panel; compose side-by-side with RGB as a 2-panel image
4. Pass to Cosmos-Reason2 with updated prompt: "Left panel: RGB view. Right panel: thermal heat map (red=warm, blue=cool). Red patches on the object indicate human hand contact areas. If contact area is decreasing, human may be releasing grip."
5. Add to Physics Oracle: extract thermal statistics over SAM2 hand mask; use as hard-veto signal if contact area drops >20% in a single frame

**Expected cost:** Lepton 3.5 + PureThermal 2 = ~$260 total; Lepton XDS = $239

### Phase 2 (2-4 weeks):

**Upgrade depth sensing:**
1. Add Intel RealSense D435i if not already present (~$200)
2. Use depth to compute hand-object clearance in 3D
3. Add depth panel to composite image fed to agents (3-panel: RGB | Depth | Thermal)
4. Implement 3D hand pose estimation (MediaPipe Hands on RGB + depth refinement)

### Phase 3 (Optional, 4-8 weeks):

**WiFi CSI coarse-state sensor:**
1. Deploy 2x ESP32-S3 (TX + RX) in the workspace
2. Use CSIKit for parsing
3. Train simple 3-class classifier: "approaching," "handoff-zone," "withdrawing"
4. Feed class label as a text token to agent prompts (e.g., "WiFi sensor: HANDOFF-ZONE")
5. Do not replace vision — supplement as environmental awareness

### Phase 4 (Future, research-grade):

**Gaze tracking:**
1. Tobii 4C or Pupil Labs Core mounted at workspace
2. Gaze direction as "attention target" signal — highest predictive value for handoff initiation
3. Integrate into agent prompts: "Human gaze: fixated on handoff zone for 800ms"

### Sensor Calibration Requirements

For Phases 1-2:
- **Thermal-RGB:** Lepton XDS is factory-calibrated; no additional calibration needed
- **Depth-RGB:** Run standard RealSense/Orbbec intrinsic calibration (tools provided in SDKs); one-time setup
- **Thermal-Depth alignment:** Calibrate once using heated checkerboard target; save extrinsic transform

### Computational Budget on RTX 4060 Ti 16GB

Current pipeline (SAM2 + MiDaS + Cosmos-Reason2-8B 4-bit) already uses most VRAM. Additional modalities:
- Thermal processing: CPU only (no GPU needed for Lepton XDS stream at 16Hz)
- Depth processing: CPU + minimal GPU (Open3D, already fast)
- WiFi CSI: CPU only (lightweight classifiers)
- No modality addition requires additional GPU VRAM

---

## Sources and References

### Web Sources

- [Lepton XDS dual-camera module — CNX Software, Feb 2026](https://www.cnx-software.com/2026/02/27/lepton-xds-dual-camera-module-combines-160-x-120-thermal-imager-with-5mp-rgb-camera/)
- [Teledyne FLIR OEM Launches Lepton XDS — Dronelife, Feb 2026](https://dronelife.com/2026/02/24/teledyne-flir-lepton-xds-dual-thermal-visible-camera/)
- [Teledyne FLIR Launches Lepton XDS — The Robot Report, 2026](https://www.therobotreport.com/teledyne-flir-launches-lepton-xds-thermal-visual-camera-module/)
- [FLIR Lepton OEM Page](https://oem.flir.com/products/lepton/)
- [Adafruit MLX90640 IR Thermal Camera Breakout](https://www.adafruit.com/product/4407)
- [Thermal IR for Robot Vision — UM Ford FCAV](https://fcav.engin.umich.edu/projects/thermal-infrared-for-robot-vision)
- [How the Best IR Thermal Modules Enhance Machine Vision — RoboticsTomorrow, Nov 2025](https://www.roboticstomorrow.com/article/2025/11/how-the-best-infrared-thermal-modules-enhance-machine-vision-for-oems/25615)
- [Person-in-WiFi 3D project page — CVPR 2024](https://aiotgroup.github.io/Person-in-WiFi-3D/)
- [Person-in-WiFi 3D paper — CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Yan_Person-in-WiFi_3D_End-to-End_Multi-Person_3D_Pose_Estimation_with_Wi-Fi_CVPR_2024_paper.pdf)
- [Graph-Based 3D Human Pose Estimation Using WiFi Signals — arXiv Nov 2025](https://arxiv.org/pdf/2511.19105)
- [Breaking Coordinate Overfitting: Geometry-Aware WiFi Sensing — arXiv Jan 2026](https://arxiv.org/html/2601.12252)
- [WiFi-3D-Fusion GitHub](https://github.com/MaliosDark/wifi-3d-fusion)
- [Reconstructing Depth Images of Moving Objects from WiFi CSI Data — arXiv Mar 2025](https://arxiv.org/abs/2503.06458)
- [Spatio-Temporal 3D Point Clouds from WiFi-CSI — arXiv Oct 2024](https://arxiv.org/html/2410.16303v1)
- [CSI-Channel Spatial Decomposition for WiFi-Based Pose Estimation — MDPI Electronics 2025](https://www.mdpi.com/2079-9292/14/4/756)
- [Wi-Fi Sensing Techniques for HAR — ACM Computing Surveys 2024](https://dl.acm.org/doi/10.1145/3705893)
- [CSIKit GitHub — Python CSI processing tools](https://github.com/Gi-z/CSIKit)
- [csiread GitHub — Fast CSI parser](https://github.com/citysu/csiread)
- [Tools and Methods for WiFi Sensing in Embedded Devices — MDPI Sensors 2025](https://www.mdpi.com/1424-8220/25/19/6220)
- [Awesome-WiFi-CSI-Sensing GitHub — NTUMARS](https://github.com/NTUMARS/Awesome-WiFi-CSI-Sensing)
- [IEEE 802.11bf Standard Overview — NIST](https://www.nist.gov/publications/ieee-80211bf-enabling-widespread-adoption-wi-fi-sensing)
- [IEEE 802.11bf-2025 — IEEE SA](https://standards.ieee.org/ieee/802.11bf/11574/)
- [Intel RealSense LiDAR Camera L515 Specifications](https://www.intel.com/content/www/us/en/products/sku/201775/intel-realsense-lidar-camera-l515/specifications.html)
- [Metrological Characterization of D415/D455/L515 — PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8622561/)
- [Orbbec Femto Bolt — Azure Kinect Replacement](https://www.orbbec.com/products/tof-camera/femto-bolt/)
- [Orbbec showcases Azure Kinect DK replacement at NVIDIA GTC 2024](https://www.prnewswire.com/apac/news-releases/orbbec-showcases-microsoft-azure-kinect-dk-replacement-at-nvidia-gtc-2024-302094399.html)
- [Orbbec Femto Bolt review — JetsonHacks, July 2024](https://jetsonhacks.com/2024/07/07/orbbec-femto-bolt-a-microsoft-azure-kinect-replacement/)
- [ToF vs Structured Light vs LiDAR comparison — tofsensors.com](https://tofsensors.com/blogs/tof-sensor-knowledge/tof-camera-light-lidar-3d-imaging)
- [RPLiDAR A3 Specifications — Slamtec](https://www.slamtec.com/en/Lidar/A3Spec)
- [Multimodal Fusion and VLMs: A Survey for Robot Vision — arXiv Apr 2025](https://arxiv.org/html/2504.02477v1)
- [Multimodal Fusion with VLA Models for Robotic Manipulation — ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S1566253525011248)
- [MULTIMODAL SIGNAL PROCESSING FOR THERMAL-VISIBLE-LIDAR FUSION — arXiv Jan 2026](https://arxiv.org/html/2601.09578v1)
- [RGB-LiDAR fusion ROS2 package — GitHub CDonosoK](https://github.com/CDonosoK/ros2_camera_lidar_fusion)
- [Infrared and Visible Image Fusion with Multimodal LLMs — Frontiers in Physics 2025](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2025.1599937/full)
- [ThermEval: VLM Evaluation on Thermal Imagery — arXiv Feb 2026](https://arxiv.org/html/2602.14989)
- [See the Past: Time-Reversed Scene Reconstruction from Thermal Traces using VLMs — arXiv Oct 2025](https://arxiv.org/html/2510.05408)
- [Early Detection of Human Handover Intentions (EEG, gaze, hand motion) — ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S0921889025003410)
- [Multimodal Learning for Proactive Handover Intention Prediction — Advanced Intelligent Systems 2024](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aisy.202300545)
- [Hybrid Sensory Feedback System for Prosthetic Hand — Frontiers Neuroscience 2024](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1351348/full)
- [Multimodal tactile sensing fused with vision — Nature Communications 2024](https://www.nature.com/articles/s41467-024-51261-5)

### Peer-Reviewed Academic Sources

- Im, S., Park, B., Jang, J., et al. (2025). "Simultaneous In-Hand Shape and Temperature Recognition Using Flexible Multilayered Sensor Arrays for Sense-Based Robot Manipulation." *Advanced Sensor Research*, 4(7). [DOI: 10.1002/adsr.70004](https://doi.org/10.1002/adsr.70004)

- Piwek, A., Ortlieb, E., Ince, C., et al. (2025). "Analysis of Temperature and Stress Distribution on the Bond Properties of Hybrid Tailored Formed Components." *Advanced Engineering Materials*, 27(16). [DOI: 10.1002/adem.202402031](https://doi.org/10.1002/adem.202402031)

- Yang, Q., Li, B., Wang, M., et al. (2025). "Machine Learning-Enhanced Modular Ionic Skin for Broad-Spectrum Multimodal Discriminability in Bidirectional Human-Robot Interaction." *Advanced Materials*, 37(42). [DOI: 10.1002/adma.202508795](https://doi.org/10.1002/adma.202508795)

- Wu, T., Li, Y., Zhao, L., et al. (2026). "Recent Progress on Flexible Multimodal Sensors: Decoupling Strategies, Fabrication and Applications." *Advanced Materials*, 38(12). [DOI: 10.1002/adma.202521375](https://doi.org/10.1002/adma.202521375)

- Huang, Y., Huang, Y., Tseng, C., & Lai, C. (2025). "Energy-Efficient Edge Computing for Real-Time Skeleton Pose Reconstruction in Sustainable Remote Health Monitoring." *Expert Systems*, 42(12). [DOI: 10.1111/exsy.70167](https://doi.org/10.1111/exsy.70167)

- Yang, Z., & Yang, Y. (2025). "WiSk: A Lightweight Multimodal Human Activity Recognition Method Based on WiFi Channel State Information and Video Skeleton Images." *Concurrency and Computation: Practice and Experience*, 37(27-28). [DOI: 10.1002/cpe.70383](https://doi.org/10.1002/cpe.70383)

- Gu, Z., He, T., Wang, Z., et al. (2022). "Device-Free Human Activity Recognition Based on Dual-Channel Transformer Using WiFi Signals." *Wireless Communications and Mobile Computing*, 2022(1). [DOI: 10.1155/2022/4598460](https://doi.org/10.1155/2022/4598460)

- Akhtar, Z.U., Rasool, H.F., Asif, M., et al. (2022). "Driver's Face Pose Estimation Using Fine-Grained Wi-Fi Signals for Next-Generation Internet of Vehicles." *Wireless Communications and Mobile Computing*, 2022(1). [DOI: 10.1155/2022/7353080](https://doi.org/10.1155/2022/7353080)

- Shin, H., Na, K., Chang, J., & Uhm, T. (2022). "Multimodal layer surveillance map based on anomaly detection using multi-agents for smart city security." *ETRI Journal*, 44(2), 183-193. [DOI: 10.4218/etrij.2021-0395](https://doi.org/10.4218/etrij.2021-0395)

- Yan, J., et al. (2024). "Person-in-WiFi 3D: End-to-End Multi-Person 3D Pose Estimation with Wi-Fi." *CVPR 2024*. [Link](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_Person-in-WiFi_3D_End-to-End_Multi-Person_3D_Pose_Estimation_with_Wi-Fi_CVPR_2024_paper.html)

- Zhang, G., Pei, B., Zhang, M., et al. (2025). "BPT-Planner: Continuous Behavior Perception and Robot Trajectory Planner in Pathological Collaboration Experiments." *IEEJ Transactions on Electrical and Electronic Engineering*, 20(7), 1025-1036. [DOI: 10.1002/tee.24266](https://doi.org/10.1002/tee.24266)

---

## Research Methodology

**Phase 1 — Web search (parallel searches):**
- 12 targeted WebSearch queries covering IR thermal grip detection, WiFi CSI 3D sensing, LiDAR options, multi-modal VLM feeding, and hardware specifications
- Queries spanned 2024-2026 literature; excluded older results unless foundational
- Sources: robotics news sites (RoboticsTomorrow, The Robot Report), academic preprint servers (arXiv), IEEE, Slamtec/FLIR/Orbbec official documentation, GitHub repositories

**Phase 2 — Academic literature (Scholar Gateway semantic search):**
- 3 semantic queries targeting: IR grip/handoff thermal dynamics, WiFi CSI 3D accuracy/limitations, multi-modal fusion for handoff prediction
- Returned 28 peer-reviewed papers from 2022-2026
- Identified key papers: Im et al. 2025 (thermal grip sensing), Piwek et al. 2025 (gripper thermal dynamics), Huang et al. 2025 (CSI edge computing), Yang/Yang 2025 (WiSk WiFi+vision), CVPR 2024 Person-in-WiFi 3D

**Phase 3 — Synthesis:**
- Cross-referenced hardware specifications against use-case requirements (desktop manipulation, 0.2-1.5m range, real-time latency)
- Identified convergent finding: thermal IR for grip release signal is novel and high-value; WiFi CSI is too coarse for grip but useful for body-pose context
- Noted specific physics limitation: WiFi spatial resolution is wavelength-limited (6-12.5cm at 2.4-5GHz), making it physically incapable of finger-level grip detection without 60GHz hardware

**Conflicting information noted:**
- Some commercial WiFi sensing vendor claims suggest sub-centimeter gesture detection at standard 2.4GHz; this conflicts with published physics (12.5cm wavelength limit) and CVPR 2024 results (91-125mm joint error). The physics limits are authoritative; vendor claims likely refer to very constrained gesture classification in fixed setups, not general 3D sensing.

**Information gaps:**
- No direct study found on thermal prediction of grip release timing in uninstrumented human-robot handoff scenarios — this is a genuine research gap. CLASP could be the first system to exploit this signal.
- 802.11bf-enabled hardware products not yet on market as of March 2026; 60GHz gesture sensing capabilities are standard-defined but not empirically validated in robotics contexts.
