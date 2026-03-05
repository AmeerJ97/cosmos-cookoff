# **Advanced Semantic-Geometric Feature Fusion: Grounding NVIDIA Cosmos-Reason2 via Apple SHARP 3D Gaussian Splatting**

## **The Paradigm Shift in Embodied Artificial Intelligence**

The transition from two-dimensional visual perception to three-dimensional physical reasoning represents one of the most significant paradigm shifts in the evolution of embodied artificial intelligence. Historically, Vision-Language Models (VLMs) have relied entirely on dense two-dimensional pixel arrays or highly abstracted textual descriptions to infer spatial relationships. While these methods have achieved remarkable success in static image captioning and basic visual question answering, navigating and interacting with the complex, dynamic physical world requires a profoundly different class of spatial comprehension. Embodied agents, autonomous vehicles, and robotic manipulators require an intrinsic understanding of metric geometry, precise object localization, physical affordances, and complex spatial constraints.

The recent introduction of Apple’s SHARP (Sharp Monocular View Synthesis) architecture and NVIDIA’s Cosmos-Reason2 model creates a unique, highly potent intersection between state-of-the-art three-dimensional reconstruction and embodied physical reasoning.2 Apple’s SHARP algorithm possesses the unprecedented capability to rapidly generate a metric, absolute-scale 3D Gaussian Splatting (3DGS) representation from a single monocular image in less than one second, utilizing a single feedforward neural network pass.2 This eliminates the traditional computational bottlenecks associated with multi-view stereopsis or prolonged Neural Radiance Field (NeRF) optimization.6 Concurrently, NVIDIA’s Cosmos-Reason2—a sophisticated model explicitly engineered for physical AI and built upon the highly capable Qwen3-VL-8B-Instruct backbone—offers advanced physical AI reasoning, precise spatio-temporal understanding, and robust object detection through 2D and 3D point localization.3

However, the core architectural engineering challenge lies in bridging these two distinct mathematical spaces. Directly feeding millions of continuous 3D Gaussian primitives into a discrete, text-based Vision-Language Model presents severe architectural bottlenecks. These bottlenecks manifest primarily in the form of "Token Inflation"—where the sheer volume of numerical data exhausts the sequence length of the model—and "Attention Drift"—where the model's self-attention mechanisms become overwhelmingly biased toward dense geometric noise at the expense of semantic logical reasoning.8 A standard 3DGS scene consists of approximately 1.2 million individual Gaussian primitives, each parameterized by position, scale, orientation, color, and opacity.11 Converting these raw continuous parameters into explicit text tokens immediately overwhelms the VLM context window, rendering real-time physical reasoning computationally impossible.

To resolve this critical integration barrier, this report delineates a mathematically rigorous pipeline for "Semantic-Geometric Feature Fusion." By systematically employing advanced representation techniques analogous to Visual Prompt Tuning (VPT) and Geometric Textual Inversion, the multi-dimensional 3D Gaussian Splatting point cloud is algorithmically condensed into an optimized "Physical State Vector." This vector encapsulates the most critical physical priors—specifically 3D centroid coordinates and splat-density gradients—in a highly compressed format that is seamlessly ingestible by the Cosmos-Reason2 prompting interface.13 Specifically, this architecture leverages the underlying Qwen3-VL model's native coordinate handling formatting to ensure maximum spatial fidelity.15 The resulting framework enables real-time, physics-aware reasoning without incurring catastrophic token overhead or sacrificing the underlying mathematical integrity of the three-dimensional representation.

## **Mathematical Deconstruction of Apple SHARP's 3D Gaussian Splatting**

To effectively compress a 3DGS scene into a Vision-Language Model-compatible prompt, the fundamental parameterization and mathematical structure of Apple's SHARP output must first be thoroughly deconstructed. The SHARP model employs a specialized feedforward neural pipeline that maps a single high-resolution RGB input (typically 1536 by 1536 pixels) directly to a dense cloud of 3D Gaussians.12 The resulting scene output strictly adheres to standard computer vision coordinate conventions—specifically the OpenCV format where the X-axis points right, the Y-axis points down, and the Z-axis points forward into the depth of the scene.17 The geometric center of the synthesized 3DGS scene is roughly situated at the origin coordinate on the X and Y axes, extending positively along the Z-axis.17

Unlike traditional mesh-based or voxel-based representations, 3D Gaussian Splatting models the environment as an unstructured cloud of semi-transparent, anisotropic volumetric primitives. This explicit representation allows for highly efficient differentiable rasterization, providing a continuous, fully differentiable landscape that is ideal for gradient-based extraction techniques.6

Table 1 provides a comprehensive breakdown of the core mathematical parameters that define each individual 3D Gaussian primitive within the Apple SHARP output structure.

| Parameter Designation | Mathematical Symbol | Dimensionality | Physical Interpretation and Representation |
| :---- | :---- | :---- | :---- |
| **Position / Centroid** | **![][image1]** | 3D Vector ![][image2] | Defines the geometric mean of the Gaussian distribution in the absolute metric coordinate space. Represents the spatial anchor of the primitive. |
| **Covariance / Scale** | **![][image3]** | 3D Vector ![][image4] | Represents the scaling factors along the three principal axes. Utilizes an exponential activation function to ensure strictly positive values, defining the volumetric stretch of the splat.11 |
| **Covariance / Rotation** | **![][image5]** | 4D Quaternion ![][image6] | A unit quaternion defining the 3D orientation of the Gaussian primitive. Combined with the scale matrix ![][image3], it constructs the positive semi-definite covariance matrix ![][image7].11 |
| **Opacity / Density** | **![][image8]** | 1D Scalar | Regulates the transparency of the primitive. A value approaching 1.0 indicates a fully opaque solid, while values near 0.0 represent volumetric fog or empty space.11 |
| **Radiance / Color** | **![][image9]** | **![][image10]**\-Dimensional Vector | Encoded via Spherical Harmonics (SH) coefficients (e.g., ![][image11]) to capture complex, view-dependent color variations and specularity rather than static RGB values.11 |

The covariance matrix ![][image12] is of particular importance for spatial reasoning. Because a standard covariance matrix must be physically valid (i.e., positive semi-definite) to represent a real volume, the direct optimization of its raw elements is highly unstable. Therefore, the architecture independently predicts the scale vector ![][image3] and the rotation quaternion ![][image5], mathematically guaranteeing a valid volumetric shape for every primitive.19 For the purposes of providing physical common sense to a VLM, the interaction between the scale, rotation, and opacity parameters dictates the perceived "solidity" and boundary sharpness of physical objects in the environment.

Furthermore, the scale of the reconstruction generated by SHARP is metric and absolute. This is a critical departure from many monocular depth estimation models that produce relative, scale-ambiguous outputs. The absolute metric scale ensures that the extracted coordinates correspond directly to real-world physics, allowing downstream models like Cosmos-Reason2 to accurately gauge distances, calculate necessary torque for robotic manipulation, and predict collision trajectories with profound accuracy.2

## **Architectural Analysis of NVIDIA Cosmos-Reason2**

The downstream recipient of the extracted geometric data is Cosmos-Reason2, a sophisticated multi-modal large language model explicitly trained to serve as the cognitive engine for physical AI and embodied robotic agents.3 To interface with this model effectively, one must understand the idiosyncratic mechanisms of its underlying architecture. Cosmos-Reason2 is post-trained upon the Qwen3-VL-8B-Instruct foundation, a state-of-the-art vision-language architecture characterized by substantial advancements in long-context processing, spatial perception, and complex mathematical reasoning.3

A paramount feature of the Qwen3-VL backbone is the Interleaved-MRoPE (Multimodal Rotary Positional Embedding) mechanism. Traditional language models utilize one-dimensional positional embeddings designed strictly for sequential text. Early vision-language models adapted this by unrolling 2D images into flat sequences. The Interleaved-MRoPE system, however, allocates full-frequency robust positional embeddings over temporal, width, and height dimensions simultaneously.21 This tri-dimensional frequency allocation natively supports complex spatial layouts, making the model highly receptive to geometrically structured prompt data, provided the data is formatted correctly.23

Furthermore, the architecture integrates a mechanism known as DeepStack, which fuses multi-level Vision Transformer (ViT) features to capture fine-grained visual details and aggressively sharpen the alignment between image features and textual representations.21 This tight visual-textual coupling allows the model to inherently understand object positions, complex viewpoints, and multi-object occlusions, establishing a powerful baseline for both 2D and 3D spatial grounding.21

To handle the immense data requirements of video and spatial environments, Cosmos-Reason2 supports a native 256,000-token context window, dynamically expandable up to one million tokens.3 While this massive context capacity is highly beneficial for long-horizon temporal reasoning—such as tracking objects across a prolonged robotic assembly task—it does not negate the necessity for data compression. As will be discussed in the subsequent sections, filling the context window with raw numerical coordinate noise fundamentally degrades the model's ability to reason, regardless of its theoretical maximum capacity.

Crucially, Qwen3-VL (and by extension Cosmos-Reason2) has evolved its coordinate handling system. Previous iterations relied on absolute pixel coordinates mapped to the specific resolution of the input image. The current generation has transitioned to a normalized, relative coordinate system ranging precisely from 0 to 1000 across all spatial axes.15 When the model generates bounding boxes or point localizations, it outputs these coordinates bounded by specific formatting tags, typically \<|point\_3d|\> or similar variants, representing the precise spatial anchor of the object relative to the total scene volume.15 Leveraging this exact syntactic structure is the key to seamlessly injecting three-dimensional centroids from Apple SHARP without triggering architectural rejection or hallucination.

In addition to its base architecture, Cosmos-Reason2 undergoes rigorous post-training with physical common sense and embodied reasoning datasets using both supervised fine-tuning and reinforcement learning.7 This specialized training imbues the model with an inherent understanding of world dynamics, physics constraints, and chain-of-thought planning without requiring human-annotated intermediate steps.7 When evaluated on complex physical physics-based benchmarks such as PhysReason, models of this caliber are tasked with predicting object trajectories, evaluating mass distributions, and solving spatial logic puzzles.24 Therefore, the data provided to the model must explicitly highlight the physical properties of the environment—such as object solidity, structural boundaries, and volumetric extent—to activate these deep-seated physical reasoning capabilities fully.

## **The Pathologies of Integration: Token Inflation and Attention Drift**

The desire to provide Vision-Language Models with exhaustive three-dimensional knowledge often leads engineers to design naive integration pipelines that dump raw geometric data into the prompt space. These approaches invariably fail due to two severe, interconnected computational pathologies: Token Inflation and Attention Drift. Understanding the mechanistic origins of these failures is essential for designing a mathematically "clean" Semantic-Geometric Feature Fusion protocol.

### **The Mechanics of Token Inflation**

"Token Inflation" refers to the exponential and unsustainable growth in the prompt sequence length required to describe multi-dimensional visual or physical data using discrete text tokens.8 In the context of 3D Gaussian Splatting, a single object within a scene might be represented by tens of thousands of individual Gaussian primitives. If an integration system attempts to serialize even a fraction of these primitives into textual tuples—for example, rendering a single splat as the string \[0.452, 0.221, 0.887, 0.950, 0.110,...\]—the resulting text generates dozens of tokens per primitive. For a complete scene containing Apple SHARP's standard 1.2 million Gaussians 12, the prompt length would rapidly scale into tens of millions of tokens.

While Cosmos-Reason2 boasts a 256K token context window 7, exceeding this limit results in catastrophic out-of-memory errors and immediate model failure. However, even if the point cloud is aggressively downsampled to fit within the 256K boundary, the sheer length of the prompt destroys real-time inference viability. The computational complexity of the standard Transformer self-attention mechanism scales quadratically with sequence length. Processing 200,000 tokens of raw numerical data requires massive GPU compute cycles and induces latency measured in tens of seconds or minutes.

Research into autonomous Vision-Language Navigation (VLN) models, such as the FantasyVLN framework, perfectly illustrates this phenomenon.8 The researchers demonstrated that relying on explicit, multimodal Chain-of-Thought (CoT) reasoning that generates imagined visual observations directly in the token space incurs severe token inflation.8 This inflation makes real-time navigation entirely impractical, as the robotic agent spends the vast majority of its processing time reading and writing geometric coordinates rather than executing physical actions.27 Token inflation fundamentally limits the responsiveness of embodied agents, forcing them into a state of computational paralysis where the cost of perceiving the environment vastly outweighs the capacity to act within it.

### **The Mechanics of Attention Drift**

The second, and arguably more insidious, pathology is "Attention Drift" (also referred to in the literature as perceptual bias or reasoning degradation). This phenomenon occurs when a dense concentration of geometric or visual features overwhelms the VLM's cross-attention mechanisms, causing the model to lose focus on its primary semantic instruction.9

In a standard Vision-Language-Action (VLA) architecture, the model is tasked with balancing highly abstract logical instructions (e.g., "carefully extract the fragile component from the dense clutter") against the raw perceptual input of the environment. When the prompt is flooded with thousands of explicit spatial coordinates, the self-attention matrices allocate a disproportionate amount of probability mass to analyzing the minute relationships between these numbers.10 Consequently, the attention heads drift away from the semantic goal.

This drift manifests physically in embodied agents as catastrophic planning failures, distractor sensitivity, and over-grasping.30 For instance, a robot might successfully calculate the exact millimeter coordinates of every object in a bin but completely forget which specific object it was instructed to pick up. A study on multi-station manufacturing operations highlights this exact vulnerability: when a VLM-based planner is required to both reason semantically and maintain long-horizon numerical world states within a transient context window, the coupling is inherently fragile.31 Cross-station consistency degrades over time, leading to "world-state drift" and unrecoverable execution failures.31

To counteract Attention Drift, the feature fusion strategy cannot rely on providing the VLM with more raw data. Instead, it must comprehensively abstract the raw point cloud into a highly compressed, semantically rich representation. The VLM should be provided with the exact mathematical conclusions of the geometry (e.g., "this is a dense, sharp boundary at coordinate X"), rather than being forced to compute those conclusions itself from a sea of raw positional tokens. This necessitates the use of advanced abstraction techniques, namely Visual Prompt Tuning and Geometric Textual Inversion.

## **Semantic-Geometric Feature Fusion: Abstraction Mechanisms**

To successfully inject the rich, continuous data of the Apple SHARP 3DGS output into the Cosmos-Reason2 context space without triggering Token Inflation or Attention Drift, the continuous variables must be distilled into a finite, highly potent set of prompt-compatible tokens. This requires projecting complex spatial priors into a compressed latent space.

### **Geometric Textual Inversion and Visual Prompt Tuning (VPT)**

Visual Prompt Tuning (VPT) represents a highly efficient paradigm for adapting massive, pre-trained Vision-Language Models to specific downstream visual tasks. Unlike traditional fine-tuning, which requires updating billions of weights across the entire Transformer backbone, VPT introduces a minimal number of learnable parameters—continuous prompting vectors—strictly into the input space.14 In a standard Vision Transformer (ViT) architecture consisting of ![][image13] layers, VPT injects these learnable, continuous visual prompts directly into the token sequence alongside the standard image patches and text embeddings.32

By extending the core philosophy of VPT into the three-dimensional domain, researchers can achieve what is effectively "Geometric Textual Inversion." In this process, the complex geometry of a 3D Gaussian Splatting scene is algorithmically inverted into a series of continuous latent vectors that function exactly like text tokens.13 Rather than feeding raw ![][image2] coordinates to the language model, the 3D scene is first segmented into discrete, functional clusters (e.g., "manipulable object surface," "rigid environmental boundary," "occlusion edge"). A pre-trained visual encoder or a lightweight 3D projection module compresses the complex Gaussian parameters of these clusters into a compact set of high-dimensional embedding vectors.

The efficacy of this approach is powerfully demonstrated by the *SplatTalk* architecture, a recently proposed framework for 3D Visual Question Answering.35 *SplatTalk* introduces a self-supervised method that natively integrates language features directly into the 3D Gaussian Splatting field. The system extracts visual tokens from a 2D VLM and projects them into visual-language feature maps. These maps are subsequently learned within the 3DGS feed-forward network, producing a dual-purpose field that encodes both spatial geometry and semantic language information simultaneously.37 During the inference phase, the Large Language Model does not parse raw coordinates; instead, it directly queries these highly compressed, trained language features extracted from the 3D Gaussians.35

To further optimize this process and explicitly prevent token inflation, "entropy-adaptive token sampling" is employed.35 This mathematical gating mechanism evaluates the Shannon entropy of the local Gaussian distributions across the scene. Regions of the physical environment characterized by low structural entropy—such as a large, flat, featureless wall or an empty expanse of floor—are aggressively pooled and represented by a single semantic latent token. Conversely, regions exhibiting high structural entropy—such as the intricate handle of a tool, a densely cluttered bin, or a sharp occlusion boundary—retain higher token fidelity, allocating more latent vectors to represent the complex geometry. This adaptive sampling ensures that the VLM's context window is strictly reserved for the most physically relevant features of the environment.

### **Mitigating Attention Drift via Structured Attention Modules**

Once the raw 3D geometry has been compressed into latent visual tokens, the final serialization format must be carefully structured to guide the VLM's attention mechanism and prevent reasoning degradation. Recent advancements in VLM optimization have introduced several techniques to enforce this structural discipline.

One approach involves utilizing a "dual sparsifier" module, which efficiently utilizes dense language representations while preserving strict semantic fidelity.40 This dual nature consists of a task-guided pathway, which selects scene tokens based solely on their relevance to the overarching global instruction, and a location-guided pathway, which retrieves fine-grained geometric features conditioned on specific spatial cues.40 By separating the "what" from the "where," the model is less likely to confuse geometric parameters with logical goals.

Furthermore, implementing an "Instance-driven VLM Attention" (InstVLM) framework allows for the direct injection of high-level semantic and geographic priors directly into the spatial coordinates.41 By formatting the text prompt to explicitly demarcate geometric constraints from semantic goals—using strict JSON or XML-like schemas—the VLM's self-attention layers are guided to treat the continuous coordinate tokens as modifying attributes attached to specific semantic entities, rather than as primary sequence drivers that dictate the logic of the sentence.

Table 2 highlights the comparative advantages of the Semantic-Geometric Feature Fusion approach against legacy methodologies for transferring 3D spatial data into language models.

| Methodology for 3D Data Injection | Structural Format | Context Window Token Cost | Risk of Attention Drift | Physical Reasoning Efficacy |
| :---- | :---- | :---- | :---- | :---- |
| **Raw Point Cloud Serialization** | Flattened textual lists of floating-point coordinates ![][image2] | **Extreme** (Frequently exceeds 1M+ tokens) | **Severe** (Dense mathematical noise completely overwhelms semantic reasoning logic) | Poor (The VLM struggles significantly to infer volume or object solidity from isolated textual points) |
| **Multi-View 2D Renderings** | Grids of 2D images fed to standard Vision Encoders | High (Depends heavily on image resolution and patch size) | Moderate (Inherent 2D visual bias often overrides true 3D spatial logic) | Moderate (Relies entirely on 2D occlusion cues; fundamentally lacks absolute metric scale) |
| **Purely Textual Chain-of-Thought** | Abstract, generative natural language descriptions of the scene | Low | Minimal | Weak (Critically lacks precise spatial grounding; prone to hallucinating object locations) 8 |
| **Semantic-Geometric Feature Fusion (Proposed)** | Condensed JSON schema with \<|point\_3d|\>, Density Gradients, and VPT Latents | **Optimal** (Typically \< 500 tokens per scene) | **Minimal** (Strict structural schema explicitly anchors attention to relevant entities) | **High** (Explicitly provides the VLM with pre-computed volume, normalized centroids, and surface rigidity) |

## **Mathematical Extraction of Splat-Density Gradients**

While centroid coordinates and bounding volumes provide the foundation for spatial localization, they do not inherently convey the physical complexity, surface rigidity, or boundary sharpness of an object. In embodied AI, knowing *where* an object is located is only half the problem; understanding the structural integrity and physical uncertainty of the object is critical for tasks like grasping, collision avoidance, and fine manipulation. To impart this "Physical Common Sense" to Cosmos-Reason2, the architecture must extract a metric that acts as a proxy for structural volatility. In the realm of 3D Gaussian Splatting, this metric is derived from the "splat-density gradient."

In standard 3DGS optimization, areas of high structural complexity, sharp occlusion boundaries, intricate textures, or physical uncertainty naturally induce significantly higher gradients in the spatial coordinates of the Gaussians during the backpropagation training phase.18 The neural network must constantly split, clone, and reposition primitives in these areas to accurately represent the high-frequency details.

To mathematically extract this volatility as a usable, static feature for a VLM prompt, the analysis leverages the pixel-aware density control paradigm formulated in Pixel-GS.43 Traditional 3DGS algorithms measure the densification gradient merely by taking the absolute magnitude of the positional gradient. However, this naive approach often fails in areas with sparse initializing points, leading to conflicts in gradient direction and resulting in unnatural, blurred, or "needle-like" artifacts.43

Pixel-GS resolves this by introducing a more robust, pixel-area-weighted mathematical formulation. The density gradient is computed not just by the positional shift of the Gaussian, but by weighting that shift by the actual number of pixels the Gaussian influences across all observed viewpoints.

The advanced density gradient metric, denoted as ![][image14] for a given Gaussian primitive ![][image15], is mathematically defined as:

![][image16]  
Where the variables are defined as follows:

* ![][image17] represents the total number of distinct camera viewpoints ![][image18] where the specific Gaussian ![][image15] is visible and participates in the rendering calculation.44  
* ![][image19] is the exact number of pixels that Gaussian ![][image15] covers in viewpoint ![][image18].44  
* ![][image20] represents the gradient of the Normalized Device Coordinates (NDC) for the Gaussian across the ![][image21] and ![][image22] axes.45

For the purpose of Vision-Language Model grounding, this ![][image14] metric serves as an exceptionally powerful proxy for physical state uncertainty or structural density. By mapping and aggregating regions where ![][image14] exceeds an adaptive threshold ![][image23] 44, the feature fusion system can explicitly highlight areas representing sharp boundaries, thin structures, or highly intricate geometries.18

Instead of forcing the VLM to analyze a dense, chaotic point mesh to deduce object shapes, the model is directly provided with a condensed "density volatility" score for each clustered entity. A low aggregated density gradient indicates a smooth, continuous, solid surface (e.g., a wall or a tabletop). A high aggregated density gradient indicates a volatile, complex, or sharply defined structure (e.g., the spokes of a bicycle, a fragile handle, or a dense cluster of cables). This single numerical value mathematically grounds the LLM's understanding of physical common sense, allowing it to modulate its planned robotic actions accordingly—approaching low-gradient objects with standard velocity while employing cautious, high-precision maneuvers for high-gradient objects.

## **Formulating the Physical State Vector Architecture**

The theoretical concepts detailed above—Geometric Textual Inversion, entropy-adaptive token sampling, and pixel-aware density gradients—culminate in the formulation of the "Physical State Vector." This is a highly formatted, serialized text string designed to serve as the optimal, zero-loss interface between the continuous Apple SHARP 3DGS output and the discrete Cosmos-Reason2 textual prompting environment.

The Physical State Vector operates on a rigid, hierarchical schema designed explicitly to mitigate Attention Drift by clearly defining the relationships between semantic concepts and physical realities. The schema comprises five critical components for every detected physical entity:

1. **Semantic Anchor:** The language-based classification or functional description of the object cluster (e.g., "manipulable\_object," "rigid\_barrier").  
2. **Normalized Centroid Localization:** The precise 3D positional coordinates defining the center of mass of the object cluster. Crucially, these must be mathematically mapped into the Qwen3-VL relative coordinate system.  
3. **Spatial Span (Volumetric Extent):** Derived from the covariance eigenvalues of the clustered Gaussians, this value represents the functional physical volume of the object, providing necessary data for collision avoidance.  
4. **Density Gradient Volatility:** The aggregated, pixel-aware gradient metric discussed previously, acting as the primary indicator for surface rigidity and geometric complexity.  
5. **Geometric Latent Token:** A reserved textual placeholder (e.g., \<v\_geo\_obj\>) representing the insertion point for the Visual Prompt Tuning embedding vector.

### **Coordinate Normalization for Qwen3-VL Integration**

A fundamental incompatibility exists between the raw output of Apple's SHARP model and the input expectations of the Qwen3-VL backbone underlying Cosmos-Reason2. SHARP operates in an OpenCV coordinate system utilizing an absolute metric scale, where values represent actual spatial distances (e.g., meters) from the camera origin.2 Conversely, Qwen3-VL has been explicitly trained to process relative spatial coordinates that are strictly normalized to an integer scale ranging from 0 to 1000\.15

Feeding raw metric coordinates into Cosmos-Reason2 will result in immediate spatial hallucination, as the model's Interleaved-MRoPE positional embeddings cannot contextualize values outside their trained integer bounds. Therefore, a deterministic mathematical transformation must be applied prior to serialization.

Given an absolute metric 3D point ![][image24] situated within a defined physical bounding volume where ![][image25] denote the maximum operational boundaries of the specific robotic scene, the normalized coordinates ![][image26] are computed as follows:

![][image27]  
![][image28]  
![][image29]  
Once calculated, these normalized integers are not simply inserted into the prompt as plain text. They must be injected using the model's specialized structural formatting. Qwen3-VL relies on distinct syntactic tags, specifically \<|point\_3d|\> and \</|point\_3d|\>, to explicitly signal to the attention mechanism that the enclosed integers represent a coherent spatial coordinate.15 Utilizing this exact syntax guarantees that the VLM activates its deep spatial reasoning pathways rather than processing the numbers as arbitrary text.

### **The Serialized String Schema**

To ensure a mathematically "clean" injection that maximizes physical common sense retrieval while adhering to stringent token limits, the extracted data is serialized into a highly structured, JSON-like string. This string is embedded directly within the overarching natural language prompt.

JSON

{  
  "Scene\_State": {  
    "Entity\_Clusters":\</|point\_3d|\>",  
        "Volumetric\_Span": "",  
        "Density\_Gradient": 0.825,  
        "Latent\_Geometry": "\<v\_geo\_01\>"  
      },  
      {  
        "ID": "env\_obstacle\_01",  
        "Semantic\_Anchor": "rigid\_barrier",  
        "Centroid\_3D": "\<|point\_3d|\>\</|point\_3d|\>",  
        "Volumetric\_Span": "",  
        "Density\_Gradient": 0.150,  
        "Latent\_Geometry": "\<v\_geo\_02\>"  
      }  
    \],  
    "Global\_Context": "Metric scale absolute. Coordinates normalized 0-1000. Z represents forward depth."  
  }  
}

In this optimized schema, the token count is radically minimized. An entire object, representing potentially tens of thousands of Gaussians, is compressed into fewer than 40 tokens. The Density\_Gradient provides immediate physical context: the high value of 0.825 for the first object instantly signals to the VLM's physical reasoning engines that this item possesses complex, volatile boundaries requiring careful handling. The low value of 0.150 for the barrier indicates a smooth, dense, unyielding obstacle. Finally, the Latent\_Geometry token serves as the critical hook where the external visual encoder's continuous embeddings are concatenated into the language model's input space, effectively executing the Visual Prompt Tuning methodology.14

## **Programmatic Serialization: Python Schema Implementation**

To bridge the gap between Apple's SHARP .ply output files and the Cosmos-Reason2 textual string format, a robust, programmatic data processing pipeline is required. The following Python schema demonstrates the optimal architectural approach to parse the raw 3D Gaussian parameters, apply spatial clustering for entropy reduction, execute the density gradient heuristic, mathematically normalize the coordinates, and generate the final prompt string.

Python

import numpy as np  
import json  
from sklearn.cluster import DBSCAN

class SemanticGeometricFusionNode:  
    """  
    Advanced processing node designed to parse Apple SHARP 3DGS output,   
    extract critical physical features (centroids, volumetric spans, gradients),   
    and serialize them into a mathematically clean string for NVIDIA Cosmos-Reason2.  
    """  
    def \_\_init\_\_(self, scene\_bounds, coord\_scale=1000):  
        \# scene\_bounds: Dict specifying the physical metric limits of the operational volume.  
        \# Required to map absolute metric data to Qwen3-VL relative coordinates.  
        self.bounds \= scene\_bounds   
        self.scale \= coord\_scale  
          
    def \_normalize\_coordinate(self, val, axis\_min, axis\_max):  
        """  
        Executes the linear transformation mapping a metric coordinate   
        to the Qwen3-VL required 0-1000 integer scale.  
        """  
        normalized \= (val \- axis\_min) / (axis\_max \- axis\_min)  
        \# Apply strict clamping to prevent out-of-bounds generation  
        clamped \= max(0.0, min(1.0, normalized))  
        return int(clamped \* self.scale)

    def \_compute\_cluster\_density\_gradient(self, cluster\_gaussians):  
        """  
        Approximates the density gradient volatility based on the Pixel-GS theoretical framework.  
        In a fully integrated CUDA implementation, this integrates multi-view pixel coverage directly from the rasterizer.  
        Here, we utilize a highly correlated statistical heuristic based on the variance of the   
        scaling parameters combined with the mean opacity of the cluster.  
        """  
        \# Extract scale vectors (assuming standard PLY column structure post-position)  
        scales \= cluster\_gaussians\[:, 3:6\]   
        \# Extract scalar opacity  
        opacities \= cluster\_gaussians\[:, 6\]   
          
        \# High variance in scale magnitude combined with high opacity   
        \# strongly correlates with sharp, volatile geometric boundaries.  
        scale\_norms \= np.linalg.norm(scales, axis=1)  
        scale\_variance \= np.var(scale\_norms)  
        mean\_opacity \= np.mean(opacities)  
          
        \# Calculate raw heuristic gradient  
        raw\_gradient \= scale\_variance \* mean\_opacity  
          
        \# Normalize to a standardized 0.0 \- 1.0 range for the VLM prompt  
        normalized\_gradient \= min(1.0, raw\_gradient / (np.max(raw\_gradient) \+ 1e-6))  
        return round(float(normalized\_gradient), 3)

    def extract\_physical\_state\_vector(self, ply\_data, semantic\_labels=None):  
        """  
        Executes the core Geometric Textual Inversion sequence.  
        Condenses raw continuous 3DGS arrays into discrete VLM-compatible tokens.  
          
        ply\_data: numpy array of shape (N, D) containing SHARP Gaussians  
                   
        """  
        positions \= ply\_data\[:, 0:3\]  
          
        \# 1\. Entropy-Adaptive Token Sampling via Density-Based Spatial Clustering (DBSCAN)  
        \# This critical step groups millions of continuous points into discrete, functional physical entities.  
        \# eps and min\_samples dictate the physical granularity of the parsed scene.  
        clustering \= DBSCAN(eps=0.05, min\_samples=50).fit(positions)  
        labels \= clustering.labels\_  
          
        unique\_labels \= set(labels)  
        entity\_clusters \=  
          
        for cluster\_id in unique\_labels:  
            if cluster\_id \== \-1:  
                \# Discard low-density noise primitives to prevent token pollution  
                continue   
                  
            cluster\_mask \= (labels \== cluster\_id)  
            cluster\_pts \= ply\_data\[cluster\_mask\]  
              
            \# Extract and Normalize the Metric Centroid  
            centroid\_metric \= np.mean(cluster\_pts\[:, 0:3\], axis=0)  
            c\_x \= self.\_normalize\_coordinate(centroid\_metric, self.bounds\['x\_min'\], self.bounds\['x\_max'\])  
            c\_y \= self.\_normalize\_coordinate(centroid\_metric, self.bounds\['y\_min'\], self.bounds\['y\_max'\])  
            c\_z \= self.\_normalize\_coordinate(centroid\_metric, self.bounds\['z\_min'\], self.bounds\['z\_max'\])  
              
            \# Extract Volumetric Span (Peak-to-Peak bounds)  
            span\_metric \= np.ptp(cluster\_pts\[:, 0:3\], axis=0)  
            s\_x \= int((span\_metric / (self.bounds\['x\_max'\] \- self.bounds\['x\_min'\])) \* self.scale)  
            s\_y \= int((span\_metric / (self.bounds\['y\_max'\] \- self.bounds\['y\_min'\])) \* self.scale)  
            s\_z \= int((span\_metric / (self.bounds\['z\_max'\] \- self.bounds\['z\_min'\])) \* self.scale)  
              
            \# Compute the Density Gradient Volatility Metric  
            density\_grad \= self.\_compute\_cluster\_density\_gradient(cluster\_pts)  
              
            \# Determine Semantic Anchor (Fallback to generic ID if specific labels are unavailable)  
            anchor \= semantic\_labels.get(cluster\_id, "unknown\_physical\_entity") if semantic\_labels else f"entity\_{cluster\_id}"  
              
            \# Construct the highly structured Cluster Dictionary  
            cluster\_dict \= {  
                "ID": f"obj\_{cluster\_id}",  
                "Semantic\_Anchor": anchor,  
                "Centroid\_3D": f"\<|point\_3d|\>\[{c\_x}, {c\_y}, {c\_z}\]\</|point\_3d|\>",  
                "Volumetric\_Span": f"\[{s\_x}, {s\_y}, {s\_z}\]",  
                "Density\_Gradient": density\_grad,  
                "Latent\_Geometry": f"\<v\_geo\_{cluster\_id}\>"  
            }  
            entity\_clusters.append(cluster\_dict)  
              
        return entity\_clusters

    def generate\_cosmos\_prompt(self, entity\_clusters, system\_instruction):  
        """  
        Serializes the extracted semantic-geometric data into a single, clean prompt string,  
        explicitly designed to guide the VLM's attention mechanism and prevent reasoning drift.  
        """  
        state\_schema \= {  
            "Scene\_State": {  
                "Entity\_Clusters": entity\_clusters,  
                "Global\_Context": "Metric scale absolute. Coordinates normalized 0-1000. Z represents forward depth."  
            }  
        }  
          
        \# Serialize with indentation for clear structural demarcation in the prompt sequence  
        serialized\_state \= json.dumps(state\_schema, indent=2)  
          
        \# Combine the structured physical state with the natural language logic instruction  
        final\_prompt \= (  
            f"System Instruction: {system\_instruction}\\n\\n"  
            f"Physical State Vector:\\n"  
            f"{serialized\_state}\\n\\n"  
            f"Based exclusively on the provided spatial coordinates, volumetric spans, and density gradients, "  
            f"execute the optimal physical reasoning chain and predict the necessary trajectory."  
        )  
        return final\_prompt

### **Analytical Breakdown of the Serialization Pipeline**

The Python implementation explicitly executes the theoretical solutions required to bypass Token Inflation and Attention Drift. By utilizing Density-Based Spatial Clustering of Applications with Noise (DBSCAN), the architecture inherently compresses millions of explicit primitives into a manageable handful of highly relevant functional entities. This programmatic step is the direct implementation of the "entropy-adaptive token sampling" observed in advanced 3D VQA models.35 The VLM is no longer burdened with analyzing empty space or redundant internal volume; it receives only the functional boundaries of physical objects.

Furthermore, instead of passing the complex ![][image30] mathematical covariance matrices (![][image7]) directly into the prompt—a naive approach which would immediately confuse the VLM's attention heads with irrelevant mathematical noise—the script utilizes the numpy.ptp (peak-to-peak) function. This extracts the functional, axis-aligned bounding box size of the object cluster. This operation provides Cosmos-Reason2 with the necessary spatial layout required for complex collision avoidance and precise affordance prediction without the token overhead of raw matrix serialization.48

Finally, the generation of the string \<v\_geo\_{cluster\_id}\> acts as a programmable hook for the underlying neural network. In a fully realized, integrated deployment, this specific text string is intercepted at the tokenizer level and algorithmically replaced with the continuous, high-dimensional embedding vector produced by the VLM's visual projector. This executes the core principle of Visual Prompt Tuning, seamlessly fusing textual logic with pure spatial latent states at the deepest levels of the Transformer architecture.14

## **Application in Long-Horizon Physics-Aware Planning**

When Apple SHARP's rapid, high-fidelity real-time 3DGS reconstructions are analytically processed and fed into NVIDIA Cosmos-Reason2 via the Physical State Vector, the resulting integrated system achieves unprecedented capabilities in complex embodied reasoning tasks.

Cosmos-Reason2’s core architecture is explicitly benchmarked against rigorous, physics-based evaluations such as the PhysReason benchmark and the BALROG framework.24 The PhysReason dataset, for instance, comprises intricate knowledge-based and reasoning-based problems that require an inherent, deep-seated understanding of physical theorems, mass distributions, and mechanical constraints.24 By systematically supplying the VLM with mathematically clean, rigorously normalized 3D centroids, precise volumetric spans, and algorithmically derived splat-density gradients, the LLM is no longer forced to "guess" or hallucinate physical states from flat, 2D pixel observations. It is provided with the exact physical truth of the environment.

Consider a practical application in an autonomous robotic manufacturing facility. A mobile manipulator is tasked with retrieving parts from a dynamically changing multi-station assembly line.31 The environment is highly cluttered, and objects frequently shift positions or become partially occluded. As the robot approaches a workstation, the Apple SHARP module processes a single monocular frame from the robot's primary camera, updating the complete 3D Gaussian Splatting field of the workstation in under one second.4

The SemanticGeometricFusionNode parses these shifting Gaussians instantly. It detects a new object in the workspace. More importantly, it calculates a sudden, massive increase in the density gradient (![][image14]) at a specific set of coordinates, indicating that the new object possesses highly complex, sharp, or fragile boundaries—perhaps a shattered component or an exposed circuit board.

The resulting prompt string update instantly triggers Cosmos-Reason2’s advanced chain-of-thought reasoning pathways.7 Because the coordinate tokens (\<|point\_3d|\>) remain tightly constrained within a rigid JSON schema, the model's multi-head attention does not drift into confusion; it remains firmly anchored to the physical reality of the metric scene.15 The high density gradient value informs the VLM's physical common sense, prompting it to immediately abandon its standard, high-speed grasping trajectory. Instead, Cosmos-Reason2 generates a modified, highly cautious interaction plan, routing the robotic manipulator around the rigid boundaries to execute a precise, low-impact retrieval of the object, completely avoiding the sharp geometric features highlighted by the physical state vector.

This level of responsive, mathematically grounded physical reasoning is unattainable using purely textual descriptions, and it is computationally impossible using raw point cloud serialization. Only through the meticulous application of Semantic-Geometric Feature Fusion can the full potential of real-time 3D reconstruction and Vision-Language reasoning be simultaneously realized.

#### **Works cited**

1. \[2512.10685\] Sharp Monocular View Synthesis in Less Than a Second \- arXiv, accessed March 4, 2026, [https://arxiv.org/abs/2512.10685](https://arxiv.org/abs/2512.10685)  
2. cosmos-reason2-8b Model by NVIDIA, accessed March 4, 2026, [https://build.nvidia.com/nvidia/cosmos-reason2-8b/modelcard](https://build.nvidia.com/nvidia/cosmos-reason2-8b/modelcard)  
3. Sharp Monocular View Synthesis in Less Than a Second \- Apple Machine Learning Research, accessed March 4, 2026, [https://machinelearning.apple.com/research/sharp-monocular-view](https://machinelearning.apple.com/research/sharp-monocular-view)  
4. SHARP \- Apple, accessed March 4, 2026, [https://apple.github.io/ml-sharp/](https://apple.github.io/ml-sharp/)  
5. TUM AI Lecture Series \- The 3D Gaussian Splatting Adventure: Past, Present, Futur (George Drettakis) \- YouTube, accessed March 4, 2026, [https://www.youtube.com/watch?v=DjOqkVIlEGY](https://www.youtube.com/watch?v=DjOqkVIlEGY)  
6. Cosmos-Reason2 — Cosmos, accessed March 4, 2026, [https://docs.nvidia.com/cosmos/latest/reason2/index.html](https://docs.nvidia.com/cosmos/latest/reason2/index.html)  
7. runjtu/vpr-arxiv-daily: Automatically Update Visual Place Recognition Papers Daily using Github Actions (Update Every 12 hours). VPR is difficult and adapt to the times, if u not read related NEW paper as exhausitive as u can, u'll be challenged by reviewers. The repo is now related to spatial intelligence, which is similar to the previous task. Keys can be found in yaml., accessed March 4, 2026, [https://github.com/runjtu/vpr-arxiv-daily](https://github.com/runjtu/vpr-arxiv-daily)  
8. Mitigating Hallucination in Multimodal Reasoning via Functional Attention Control \- arXiv, accessed March 4, 2026, [https://arxiv.org/html/2510.10285v1](https://arxiv.org/html/2510.10285v1)  
9. Mitigating Object Hallucinations in Large Vision-Language Models with Assembly of Global and Local Attention | Request PDF \- ResearchGate, accessed March 4, 2026, [https://www.researchgate.net/publication/394677903\_Mitigating\_Object\_Hallucinations\_in\_Large\_Vision-Language\_Models\_with\_Assembly\_of\_Global\_and\_Local\_Attention](https://www.researchgate.net/publication/394677903_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_with_Assembly_of_Global_and_Local_Attention)  
10. Apple SHARP \- FiftyOne Model Zoo Integration, accessed March 4, 2026, [https://docs.voxel51.com/plugins/plugins\_ecosystem/apple\_sharp.html](https://docs.voxel51.com/plugins/plugins_ecosystem/apple_sharp.html)  
11. Sharp Monocular View Synthesis: Real-Time 3D Rendering \- Emergent Mind, accessed March 4, 2026, [https://www.emergentmind.com/papers/2512.10685](https://www.emergentmind.com/papers/2512.10685)  
12. CVPR Poster Geometrically-driven Aggregation for Zero-shot 3D Point Cloud Understanding, accessed March 4, 2026, [https://cvpr.thecvf.com/virtual/2024/poster/29775](https://cvpr.thecvf.com/virtual/2024/poster/29775)  
13. Hodasia/Awesome-Vision-Language-Finetune: Awesome List of Vision Language Prompt Papers \- GitHub, accessed March 4, 2026, [https://github.com/Hodasia/Awesome-Vision-Language-Finetune](https://github.com/Hodasia/Awesome-Vision-Language-Finetune)  
14. Qwen3-VL is impressive\! : r/LocalLLaMA \- Reddit, accessed March 4, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1olytpd/qwen3vl\_is\_impressive/](https://www.reddit.com/r/LocalLLaMA/comments/1olytpd/qwen3vl_is_impressive/)  
15. AndriiShramko/4DGS-Video-Generator: Professional Desktop Application for Converting Video Frames to 4D Gaussian Splatting Sequences using Apple's SHARP Model \- GitHub, accessed March 4, 2026, [https://github.com/AndriiShramko/4DGS-Video-Generator](https://github.com/AndriiShramko/4DGS-Video-Generator)  
16. Initial commit · apple/Sharp at d026b7c \- Hugging Face, accessed March 4, 2026, [https://huggingface.co/apple/Sharp/commit/d026b7c1cb9bb3c08ca460d6830da74aab5cfd8e](https://huggingface.co/apple/Sharp/commit/d026b7c1cb9bb3c08ca460d6830da74aab5cfd8e)  
17. GaussianUDF: Inferring Unsigned Distance Functions through 3D Gaussian Splatting \- arXiv, accessed March 4, 2026, [https://arxiv.org/html/2503.19458v2](https://arxiv.org/html/2503.19458v2)  
18. Efficient Density Control for 3D Gaussian Splatting \- arXiv, accessed March 4, 2026, [https://arxiv.org/html/2411.10133v4](https://arxiv.org/html/2411.10133v4)  
19. Cosmos-Reason2 models understand the physical common sense and generate appropriate embodied decisions in natural language through long chain-of-thought reasoning processes. \- GitHub, accessed March 4, 2026, [https://github.com/nvidia-cosmos/cosmos-reason2](https://github.com/nvidia-cosmos/cosmos-reason2)  
20. Qwen/Qwen3-VL-8B-Instruct \- Hugging Face, accessed March 4, 2026, [https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)  
21. Qwen3-VL is the multimodal large language model series developed by Qwen team, Alibaba Cloud. \- GitHub, accessed March 4, 2026, [https://github.com/QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)  
22. Qwen3 VL 8B Instruct \- API, Providers, Stats | OpenRouter, accessed March 4, 2026, [https://openrouter.ai/qwen/qwen3-vl-8b-instruct](https://openrouter.ai/qwen/qwen3-vl-8b-instruct)  
23. Daily Papers \- Hugging Face, accessed March 4, 2026, [https://huggingface.co/papers?q=physics-aware%20planning](https://huggingface.co/papers?q=physics-aware+planning)  
24. Discovering High Level Patterns from Simulation Traces \- ResearchGate, accessed March 4, 2026, [https://www.researchgate.net/publication/400661907\_Discovering\_High\_Level\_Patterns\_from\_Simulation\_Traces](https://www.researchgate.net/publication/400661907_Discovering_High_Level_Patterns_from_Simulation_Traces)  
25. Discovering High Level Patterns from Simulation Traces \- arXiv, accessed March 4, 2026, [https://arxiv.org/pdf/2602.10009](https://arxiv.org/pdf/2602.10009)  
26. Computer Science \- arXiv, accessed March 4, 2026, [https://www.arxiv.org/list/cs/new?skip=25\&show=1000](https://www.arxiv.org/list/cs/new?skip=25&show=1000)  
27. Daily Papers \- Hugging Face, accessed March 4, 2026, [https://huggingface.co/papers?q=Vision-Language-Navigation](https://huggingface.co/papers?q=Vision-Language-Navigation)  
28. CVPR 2025 Awards \- The Computer Vision Foundation, accessed March 4, 2026, [https://cvpr.thecvf.com/virtual/2025/awards\_detail](https://cvpr.thecvf.com/virtual/2025/awards_detail)  
29. Clutter-Resistant Vision–Language–Action Models through Object-Centric and Geometry Grounding \- arXiv, accessed March 4, 2026, [https://arxiv.org/html/2512.22519v1](https://arxiv.org/html/2512.22519v1)  
30. VLM-DEWM: Dynamic External World Model for Verifiable and Resilient Vision-Language Planning in Manufacturing \- arXiv, accessed March 4, 2026, [https://arxiv.org/html/2602.15549v1](https://arxiv.org/html/2602.15549v1)  
31. Prompt-based Adaptation in Large-scale Vision Models: A Survey \- OpenReview, accessed March 4, 2026, [https://openreview.net/notes/edits/attachment?id=ZRbqjnzjcu\&name=pdf](https://openreview.net/notes/edits/attachment?id=ZRbqjnzjcu&name=pdf)  
32. CLIP Goes 3D: Leveraging Prompt Tuning for Language Grounded 3D Recognition \- CVF Open Access, accessed March 4, 2026, [https://openaccess.thecvf.com/content/ICCV2023W/OpenSUN3D/papers/Hegde\_CLIP\_Goes\_3D\_Leveraging\_Prompt\_Tuning\_for\_Language\_Grounded\_3D\_ICCVW\_2023\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2023W/OpenSUN3D/papers/Hegde_CLIP_Goes_3D_Leveraging_Prompt_Tuning_for_Language_Grounded_3D_ICCVW_2023_paper.pdf)  
33. 3DAxisPrompt: Promoting the 3D Grounding and Reasoning in GPT-4o \- arXiv, accessed March 4, 2026, [https://arxiv.org/html/2503.13185v1](https://arxiv.org/html/2503.13185v1)  
34. SplatTalk: 3D VQA with Gaussian Splatting \- arXiv.org, accessed March 4, 2026, [https://arxiv.org/html/2503.06271v2](https://arxiv.org/html/2503.06271v2)  
35. SplatTalk: 3D VQA with Gaussian Splatting \- arXiv, accessed March 4, 2026, [https://arxiv.org/html/2503.06271v1](https://arxiv.org/html/2503.06271v1)  
36. SplatTalk: 3D VQA with Gaussian Splatting \- CVF Open Access, accessed March 4, 2026, [https://openaccess.thecvf.com/content/ICCV2025/papers/Thai\_SplatTalk\_3D\_VQA\_with\_Gaussian\_Splatting\_ICCV\_2025\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2025/papers/Thai_SplatTalk_3D_VQA_with_Gaussian_Splatting_ICCV_2025_paper.pdf)  
37. SplatTalk: 3D VQA with Gaussian Splatting | Semantic Scholar, accessed March 4, 2026, [https://www.semanticscholar.org/paper/SplatTalk%3A-3D-VQA-with-Gaussian-Splatting-Thai-Peng/32545b681bcf6b80cd837458f08ca7b4f41f51d5](https://www.semanticscholar.org/paper/SplatTalk%3A-3D-VQA-with-Gaussian-Splatting-Thai-Peng/32545b681bcf6b80cd837458f08ca7b4f41f51d5)  
38. SplatTalk: 3D VQA with Gaussian Splatting \- arXiv, accessed March 4, 2026, [https://arxiv.org/pdf/2503.06271](https://arxiv.org/pdf/2503.06271)  
39. GaussianVLM: Scene-centric 3D Vision-Language Models using Language-aligned Gaussian Splats for Embodied Reasoning and Beyond \- arXiv, accessed March 4, 2026, [https://arxiv.org/html/2507.00886v1](https://arxiv.org/html/2507.00886v1)  
40. VLMFusionOcc3D: VLM Assisted Multi-Modal 3D Semantic Occupancy Prediction \- arXiv.org, accessed March 4, 2026, [https://arxiv.org/html/2603.02609v1](https://arxiv.org/html/2603.02609v1)  
41. PocketGS: On-Device Training of 3D Gaussian Splatting for High Perceptual Modeling, accessed March 4, 2026, [https://arxiv.org/html/2601.17354v3](https://arxiv.org/html/2601.17354v3)  
42. Efficient Density Control for 3D Gaussian Splatting \- arXiv.org, accessed March 4, 2026, [https://arxiv.org/html/2411.10133v2](https://arxiv.org/html/2411.10133v2)  
43. Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting | alphaXiv, accessed March 4, 2026, [https://www.alphaxiv.org/overview/2403.15530v1](https://www.alphaxiv.org/overview/2403.15530v1)  
44. Pixel-GS, accessed March 4, 2026, [https://pixelgs.github.io/](https://pixelgs.github.io/)  
45. qqqqqqy0227/awesome-3DGS: 3D Gaussian Splatting: Survey, Technologies, Challenges, and Opportunities \- GitHub, accessed March 4, 2026, [https://github.com/qqqqqqy0227/awesome-3DGS](https://github.com/qqqqqqy0227/awesome-3DGS)  
46. Density Control with Pixel-aware Gradient for 3D Gaussian Splatting \- ECVA, accessed March 4, 2026, [https://www.ecva.net/papers/eccv\_2024/papers\_ECCV/papers/02926.pdf](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)  
47. 3D-LLM: Injecting the 3D World into Large Language Models \- OpenReview, accessed March 4, 2026, [https://openreview.net/forum?id=YQA28p7qNz](https://openreview.net/forum?id=YQA28p7qNz)  
48. Discovering High Level Patterns from Simulation Traces \- arXiv, accessed March 4, 2026, [https://arxiv.org/html/2602.10009v1](https://arxiv.org/html/2602.10009v1)  
49. apple/ml-sharp: Sharp Monocular View Synthesis in Less Than a Second \- GitHub, accessed March 4, 2026, [https://github.com/apple/ml-sharp](https://github.com/apple/ml-sharp)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAXCAYAAAAyet74AAAAf0lEQVR4XmNgGAUDCqYCsTEUg4AyFJ+DKWCB4k9ALArFIJAExeuhfOIV2kAx3AoomA7FxTCBSiieBBOAgttQbAET2AzFIMUwALLhKxTLwASJUgjyxAcoPgXEhVCcDsSLoRjkTrD9l6EYLyBaYSkQT4ZivGAREPtBMV5AtEKiAADHrSOAr1codQAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADcAAAAWCAYAAABkKwTVAAAB6ElEQVR4XuWXzyulURjHj5BSExYkRnPzo6SUDWEzNSyMIqyQUqxYSMnPBTMlNTXT+FVqJhY2FAtkIQuWimYxdvwFpGRrIb5P53veexyXe6+7cnzq073nOee9Pee9z3nee5V6Z2TCauod3m4uFe7BAuoV3+GwG/QFLzeXRi9ghjP35mmhZ+6ED0zTdXcCfIArdB6uwUE6AX+Hl0YlV+myFw9gvjXXDVetcSxIDldUqm4DbtMks8jrzZnkx03A4if8RIUjOEBls5uMx8KYCp/vG1hhzUliI9Y4XhrgKQzRgL+0xw6SkPVeHvC3sJjGSwg20n+MpdBrWMVYPJgfHHKjIzbDISql+RKS1LkTy3PG0VimUtJCDZVvMtksipFK2E4N9TTA6819pdIsXE5gJ51V+mwa6mCTNc6C/XwVI3FI2zj+QbeCFZpSFfmYGOSG7MNJWghHlb7m0XWmYRzbQSJdbYZOwf9wiboN6Au8h700Eh10V+lzfkklMRtJWD6rhLr84msrlY65oHSXDDql4PXmDFIadnt+DeVKl5wpOxtJ0nRL4SO8o7VmkUUfzKYJk6P0AzARvqnws8zlj9J/qURBqmGHuqSrp5WRMM2wi8aLJCSl+RyflS5tcQ4uqudvRBkscoOJ4vXmvOEBAHGCPveQYS4AAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAaCAYAAABsONZfAAAAmUlEQVR4XmNgGBnACIpXAXE3FM8G4gooDkAoRQCSNWkA8V4oFkQS5wTip1DMjyQOBhOBuAqK0UE/FGMAsjT1AfErKM4GYikoxguEgXgNFH8H4v9QPANZETogSxMy4AHiHCgGaQSFLAjDwQQoxgXeAbE0FIMBMwPC8wowQShIgOIiVGEyNZlCBUG4hwES7CA8BYiToXgUjBQAAAyOLK/1PO/5AAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEoAAAAVCAYAAADhCHhTAAACIUlEQVR4Xu2YwYtNURzHT8NQUkZpTKY0sptRlEZMomYlLIRMYoMVGyWahsa8QSklYqOEmCZpZkFZTCnSWFhZ2vgHrOzYWPD9ut/fvHN/Gfe+N2dhcT71qdv53Xu/95137jnnvRAybdMFd8rMP8gdVYNOOAd7ZGYRGvCCb8w0WSm/wjWulonIHVWTQ/KzL2TK3JDTviBWyavyvlwXn5SIs3IC3oG7ZWosI86p5Ikc9QVxWfLm5IPcu3BGGviwzyW5Ca/IlFiOYTmV5I6qmfFQnvIFwS0D/QGfwe0yNYPwp3wDD5fLybAcZrSUc1Fe8wWwOjQ3oCfhXfhdcoOaimVwE9wjOXd8g0MyBcyIc2yOspxK9su/Teaz8JEkvfCLbIW18FwofiJRz/lQvudy+AlulnWwDMvxMCPOYUacU0mf/Fhu/sNReElyxZuB/ZJwpNmIHIcH5QvVjWH4C56WngF4HY7JKXgsqq+AJ+BjSSZDcR0llmE5HjvXcpjhcwjv8zoUn4GesUKfzB1VsGhHGS/hNt9YQTyp3wvF9dSHk62hubltlY1wF3wgyXvYIQ1mWM5S4Ip+W3JeK9ENX/nGCnaEYm6g70LzwbfEJ4lGKEYGbQeO6gNyfSi+WE9DtptBRkJ5q7QhOl7gCDwu68BV0Hbs7GSbTP0eizt7DumlwAXHXk1OAf6fDstoN8fehnn4FN6SuaMcLXXU/8y+6PhtqLmkp+A3BtuRyHTJfzsAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYCAYAAAAlBadpAAAAwUlEQVR4Xu3QoQrCUBTG8Ru0iQZBjMNqVTBYfAODWAS7weADWCx2X0GZSbuIsmZSg8U3EJtVDPod98HdzlZkTfaHX7lnZ1yuMWk12MGbTrClPZyhR7HVjV2uqFknMHPCI79EyyO4km4Idyqq2bc1zChYATbQpEhZeIBLY2N/dIGy/TRaouWW8R+jRFKGbtDnWWwTOKozuY14wkDNQnkwVWcNkht1IU/V4EeSZ35clmdf0gsO0CYpR3NYwIocztP+vA/ZZDUDh1hnmwAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAF0AAAAWCAYAAACi7pBsAAADMElEQVR4Xu2YW4hNURjHPzTkGspdLuUueSQltwfX5C4TGbfcSkqEMCP3hEjxgI7ILQ9CLklTIsqDezzQTCHEi0dJfP/2f81e8509zhm7c8bD+tWvZn17tffsb33rso9I4L+gnTqMBopESHqRKVFvqZ1poAhsV9faYKCwhKQXkWb0s9rcXAsUiOn0lb0QKBy76Dl7gUylGXWnupHe8PqkZYh6Wt1Ly9Wrag+aFpzKDtJD6gF1Py31+v0rjek69YRE94ebJHpmFiHp6al30jMUHSzj1K+0C2MYHJhhOw2d6Bd1vBdfqVZ57bTclbi4wGD1N+3jOqWgglZKPADwo7q4ppfHSVpm4uCBxBXieE4Tb1ZPXGU/NvGjkj2o3Uw7XyapP9UOFMxVq2laMIt+0GmMuW+dOgd1PcWR0aeJ+kudSAGq0q8Q5yJ1jMRLUb7coXtM/KlEg9qVLlRf1OqRP1vV+yZ2ROIZjvfcoG6hu9UWrmMejJY4J60Ym02r2Z5Hx7Idki4NkPTJ9KwLeLyXaJODYLPUnpYXqFvv71G8CGhPV6ltGfM5Q1ezPZLiBfq6TgRLWhKN1KVqP2pZJtH/6MAgvlOX0OHqde86Dgv2IxFFh6KCFjwT3zgQlKjXKA4HYAXtzrb0pg9dwANV69Z0rL2otgwFb2hrtl3S3W83GFmIJGI2WAbQY+oO9Tat8juRupKONfWbeopaUH24/zaK04U/UzFoNclQLks8sx2YjW9pEu7e5RKdWr5Tu++tcX+EpDdA0h1X1EE2aHgi0XSFwJ1k2rCN0w50y41jqMQ7+984TLHkWHKt6XhhmItZEi2bljkUZ+0kXGJzgROSW+P7m2tZdFQv2aAHKgZHr4EUnKfuA6aSujXdUaE2NbEksOHB5faC8tIGPLBJoWByFQ3Apm33L6zVMyiY4F0DmI3zaS4wSz5QgN+z8L/DxBzMlOgLzf9KG0FvSjR6+yimZS9aJtFShGr2K7olTdqAfLC5HZe4Qi6qPSU6RUBMU2xUUyg2Kx/EcoHlC+I+j9RRFOf/1xKfpCAq3geDgGfa51oWqM/UT7SMcfeFitNRFiHpDZD0QIH5A65pDuxSbl88AAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHUAAAAYCAYAAADEbrI4AAADaUlEQVR4Xu2YWaiNURTHl1nmMkehTJllSggPeJBSItQlw4OSEC+KupQkc6aSmRKFSApFyfCgJMOLqBuSJEIoEuvf/q/Ovuucc6dznZvb/tWvzrf3951vf3vtWSSRSJSjBb2mblBv0/O8RjoslAnqKYr/fa3upLg+lLk1USgpqPWQObQDr6/T4bxeSgtlefS7mfpKbURB1jt6qT/pWwmR30xPqDfVbxT3NAyP1Qlb1ecU5bkR+VC9pPamHpS7lKLF71D3UrT4wXZjxDB6Tt0moUdAVPIutSUFrdUX1GhDPYMklPkPfcxr85m6lgJ7B5gmIS4xud4hKylesMDlARtm0Aq7u7xiY8PQRZ+hXJZMxXiOqsuoMYTGgTCGqrdoJ5d3Vl3k0mZIJuhVoZ9kgjrS5Y2J8ia7PDSuEpeWkxTUbP77oBoo7Be1D/Ug4GN9YpEpo/E8YzxQD1OjL/0h5ech0JhiCPag4aynnk2SHeg96mxaFRar76if0mZKKC9E8GMeqZ1dWoVgXsB4fp/GFQDaqu1dmmc1xdxcFVFoX/B8IDjWgvE7BnMNVp34BmjY/+OZC3S6hAVHRZxUP9BVknuetgaD73gjmRUpen9l4P/NmOYSvgM9H4IpklnnfJLwvnzzdU4GSOitsFrdvAiskPBR9mHwKt0f3ZcL9IynFAH+LmHkyTXdgC4SKhdiUYZn7tIqV2Ye0FnQQ62R4Tu2UywC+2durR1SUAP1Kqgj1GPUj/UYfm1PVhdcUY9TYz79qjaJ0isCQybWDx9pZWBInCXhHRCNqxCwLkEjibdfqGv4RF3HtFqhh4Q9UFPqQc8d7xMdmH+gnzvzOZBWBioW++SF1LANPioJi54YHAxMpR6sKlGB0BMvtGJsZT3XZ1QTfPdLn0jK1I0+sSbgAAIeUVu5PGBBvqN2c3nFYpKEwHWlhq12P/N6NMUQh/RSGoO8MxK2DtBAo4bv1Y5ROkBjPk39CFZd0DD2ubSeFN+4RsKoAzFy1ogU1EC9CWo7CdsYiH1efFwFsbD4TX9J9jbnX9JAwp4Q4igNw+9uaoGdSO+pW6T8nIuDgCX0oGSOBXHcOIr3xJRQHCzgHXb/AQnbNGyDKtsK5WOchIYEsf9EXc+jwIKIcmLxhDkf1vW5QCKRSCQSiUQikUgkEgXxF44iGbdgdzPwAAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAWCAYAAAAW5GZjAAAAfElEQVR4XmNgGAX0BGxQ3AXEC6C4BIh7gZgDigWgaolXzAjEq6EYJIkMVgBxERQXggTsgPgrFPMg1IFBExBvhOJmkEAtEG+HYnRQCcRPoVgEJECS4hAgXg/FyIAFiJcC8SYoBgOQB5dBcRsQ10MxyFRtID4AxbNIVjysAQC3BiXMVkx8ZAAAAABJRU5ErkJggg==>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAYCAYAAAAh8HdUAAAAl0lEQVR4XmNgGAU4AR8U1wPxUSg+BMS7oFgXoRQCrID4OhQnATEjFIMAzLDzUD4ckKRJD4pfALEFFOMCLTDGdiiegpDDCcJhDJI1qQHxfyh2RJHGAwIZEJoE0eSQAScUG4A4ZGmyZEBokkJShA6yoFgeJrAfiithAmggCIgjoRgOyNIEi+1OIJ4KxaC0VwfFLgilo4AyAAB6hiZS2s0+gAAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAWCAYAAAAfD8YZAAAAuElEQVR4XmNgGAUaQLwLiP9D8QlUaQZRIL4NxX/R5CjTDAJGQDwbikEGmEMxDEhBcSuSGByUALEwFJ8C4ulQDAMRUOyMJAYH7UjseCD+BsVCULGJUMwFU4QMyNbMCsRNSHw2IH4KxUVQsWlQjAEcgdgbTawKiu8DsQIQ10IxBqgHYl40MREo/grE+4HYFooxAEWap6ILIIEZQPwRiFmgGAxcgHgrFINMxxr5QKAFxFvQBSnSPApIAAC4iDN4V/ZzEgAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHAAAAAVCAYAAACe2WqiAAAC9ElEQVR4Xu2Yy6tNURzHf0hek4uU58TgFpmYeA04XknJO4nS9cizpFDKaydMDCQlFHdAmCjCSJQimXmEUkwY+RMY8Pva3197nWXtbTunc1un1qc+nWutfc/5uGufs9c+IolEIpFIJFxG0XvqK/UMjR232bpjZ7r6hL5W1zZPt8ZW+kzdre6gseM2W3fsXFdP00vqrObp1kgLOHB0ZAFv0GP+ROR0Y/NXdQFtm2nqQ/UHxXXkgTqYxog1W7c1W3eMrKEv1F/qY3rWPahVRkv+pHC8NxcraLbubmkGq9R3/mC7pAUcODqygIvUz7RbQLN1d5pJ/kAbZGq/P9guh9S7tIyl9Js/UYPh6j6KC/e25umWQLN1V5Gpt/3BGkygfeqH5qm2uK8e8Ac9Msmba3ffVE/SMmbQW/5EDbC930LBObVBDftI3KuOccbLQLN1V4Hn2+MP/ifv/QEHt7lON94A+OSowpprd+MMw7cBoW8EsEvCvRXuXSB+HiTFO6BPfa6OpCEuq6spOC5/32fathrXtF3OeBlotm4XfDNzTV1HsUPtVXtoJvm94yNah6prlttc1T2Z4tix3pw1W7c1W3cmxX16sDktYDVRLyA+An7yERrr6WH+2+6zpqqb1RMUbOJjGReluA8Cmbqd+uAeb6M/6GHN1u1yQZ2tDqX2x8dJBBdKfqKtpHX41zXQmqu6sfuEL/0JKZqt2z1h3GbrlinqF4oL6h072qGfzpR8E/KW4otYjM+lAPNV4LPcziBwXp1DfTIJPx+arduaQ91v+NigVyRv/kiHcB6vEXqdEPi9KjIpfz4s2if1FMVGzseaQUOKZuu2ZvDnNSZKsSBH1XHOAcZyinfbTvUqxQ5yvuTvTNhQl0hxxofAOP7oEL8b2oWNoHiuEGi2bmsOdR9Ul6lHKE4WbBr20w2Sn9HzaBnWg//vd3UFHeYeJNXNAH/Dp5J3QVx+fKzZuq3Zuq3ZutMCSpcvYCfAwkDbDLnGjN8KFzcdERm/AbKb4HksNLBgAAAAAElFTkSuQmCC>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAYCAYAAADKx8xXAAAApklEQVR4XmNgGBlAA4j/Q/FjIN6Nhk9BMUh+DhSDAdkaQWAyFH8EYnkoRgcuQLwDiuGAF4ofAPFeKGZEVgAFs6AYA3gzIJxdgCYHAkpQjAHI1ggCy6H4MxBLoMnhBSDFIPwAiMVRpfADsjVWQbEFugQQZEAxBggD4lwoxgYWQTEGIFmjIRSjJCc0EM6ASBxgAAqAO1B8hAEzrT6AYlDcLoDiUUATAACirDkFRGNXqQAAAABJRU5ErkJggg==>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAaCAYAAABsONZfAAAAhUlEQVR4XmNgGP7AEojXQ/F/IF4HxcnIitABWZpAoACKz6FL4AMroLgPXQIfIEvTKygOQpfABXQZIAEAwiJocjgBWZrygfgaFKMDfig2QJdYA8RzoBgdVECxJLrEMyCOhGJkIAvEK6EYAxCtyQmIN0IxKACmQTEonnZA8TcgLobiUTCcAQDYnifGLgkipQAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAAYCAYAAAD+vg1LAAAA3klEQVR4Xu3UMQsBcRzG8T8WFKtBLFabUax4DTYDg1lJJlkM5BUwy2BDKQZlUcpbYDdaJJ5/90j9Dme4Y7lvfaan+3PdoZTbr0vCkm6whQWt4ARt8vCar8uQPjgitpB6flBDbJY5dnCddnJgJTrIwaoZdeXAqqTvyEuWBeBMBbE9GtBeDp9y7OAcXCksNp2+7SPVxNaHGJnqwJpeVVbGQ9WCYvvYBpokq8AcoqTzUV4Z3/htth6chRFdYEwtmMCUisr8M45TCnpisyX9zqchQbY1VMYfk59sy7GD3f7QHdnNOyUU0f/GAAAAAElFTkSuQmCC>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAYCAYAAAA20uedAAAAXklEQVR4XmNgGHgQCcSboRgDKAOxJxRjALySGIALiNugeDcQ60IxGGQBMQ8UnwJiVygmLCkHxNpQ/AKIWaEYDjqhuB+IZaGYESb5DIp1gHgaFBMneRKKQcZ6QPGIAQBxaRU+6eWvxwAAAABJRU5ErkJggg==>

[image16]: <data:image/png;base64,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>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAABJUlEQVR4Xu3TMUsDURAE4FViJYgQGwWxsBZMZ6FgJ4pYiFUs0wiBEJJC1B+gCCIWFoJga6MEBBvBIo2g+A8s7e1sLHQmb8R9OcET7iBFBj7I3uax3O2dWT+9lnG4hXd5jduJbMEnPMpB3P49Y3AtPDwctzuZkmML/ylKquQ+oAzLwsMzcbuTbTm08Gj+lVMYEu5gNW7bCkzIA+zH7b9z7n63oebqSdiAEfmARddPlVwHlKDu6gs4cnUTBmBN3qDg+swSVCWRHZh19R7cWDhE07rOoXSlOnUuYdDVm/AC6/KdJ2m4a8w8nMGoJJLLAN76vXBpd/bzms5ZWDSfO3Eg+/w+6Nnil4A7bFnYS/duMgk/zl0Le/S7zCwVOIEFyTy5D+inB/MFKgJBDqqWisQAAAAASUVORK5CYII=>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAYCAYAAAAs7gcTAAAAjElEQVR4XmNgGLpAHYp3A/FjKCYISoF4CRQTBCQp3gvEqVCME/BA8U8gVoViCyCuAuLVUAwHHlB8D4hdodgNiBuAeA8UwwFJinuh+AUQT4BiIWQFyOAiFPsCcSAUX4LKiUMxGEgA8S8o5gJiLSh+CcQsDAibSFcM8tgOKAYBTig+A8QTgVgXikcBnQAAUCgm1JFcweMAAAAASUVORK5CYII=>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAYCAYAAACfpi8JAAABLklEQVR4Xu3ULUgEQRjG8dcPBMOBIBYRTOeBwWDRJGKzKYJy3qFgVVDEIDaLwSBYDAYNBosYjFrFZBNMRjEJdpPPwzxzO7fcgeCAezB/+JXd2dth990zS6VSqb83DKdyD1XYlys4lCXYgjvZ48Ux4w0H5MvczXzj8C3zOrYoT35RrAqzkVGYkk/oDs5twKP4joSvLd9t/kDQgblrWl3XaFeuc8fPLZsR34ssmJuv31aDdWmbH0A+gbA3mBHGV/UuJTiDEanDpta16hIqsgPHzaddhdnIh3Be2JBwZnqF8Uee5QbKMCFrsKJ1YV3yCicyaNnwR+/B3Ib6xDcpHHD+9xDnZCxYE6Ue4RPil+G7gGXYljnLvk4+ldnGykgVZiPt6ofV/MH/aNqaZyXVmf0AgYJH40MCMLgAAAAASUVORK5CYII=>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAZCAYAAAAv3j5gAAABDUlEQVR4Xu3UMUsCYRzH8cdarEEIJ8E0NIQWQXEqfAGCOOgciItLU9QQuDWrKC0RTToKNrjlIvYKpNk5EPIl6O9//ryeDk3J86b7wme454F77rnnOKXc3NzsLgsvUNNUyNb2vtAljeFYGx9CmWzpgx54fUhTiJNUhzOydgEjilrmzBxZyAczSnAsSd9wQNv0TmubUJDXT9RVP7vLQJPzq4qpxY5FDp4hQmb39AYNtfgoxC2ESHYoN1nXDfQpDXnwkpljC+nJeXxRShuvwhWck7UOBGgAJ+AnozbckVSAT9I/gld4hCMqqd87lPNc1oJr8JCRvJ7l4cufoAdh2lTROvBXji30307Vdg/jtltzz5JFb5AAosMAAAAASUVORK5CYII=>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAZCAYAAADnstS2AAAAeklEQVR4XmNgGAX0BPFQvBCIV0JxORDnAPEGKOYnWbESELdDMQgkQ/EZILYG4ndQLAiSFGCA6IJZMwuKW6F8FECSYnRwE4qdoHwJKAaDcCA+DMVqQPwfijmAmAeIm6AYDPyBeBcUdwLxaiieAcTLGSDOBGHSFY8C2gMA+f8j/p5C6QwAAAAASUVORK5CYII=>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAaCAYAAACO5M0mAAAAoElEQVR4Xu3RMQ4BQRiG4VHYhk6nIBqVE4jSFZxAg55CIUqNRCOh5hJUDrHFJttp3UAi8f3jNdlyNKLwJk8xk6+ZXef+fb1EpjhJC9YMCzsMpYJMurAuGHw0bEgHuZRQlTuaNrTmWL8vVF+uCO0xLtxt5IhQ9LCHs+xwkwl8Nfd6lX+ZquMhbfiihyNJYS1x4Byy37WCfZ4tysWRFT389Z4aZiqNpMysaQAAAABJRU5ErkJggg==>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAZCAYAAADNAiUZAAAA5ElEQVR4Xu3SvQ7BYBTG8ddHQoLdBfiIhMlms/lYRIRBYheDWMRgMZkwGAxWrBaTXXAprsHAc9KnadKKGKol6T/5DU1Pc+hbpby8vP6hEMzoDg+TGwX0B+zIlaVjKJK0IVuXvCsMR3IsV5ZWYUGOtYI26dVoDR0YkHwDURpCE+bkhxTUSeYjZOkECdLL015pCwokP3BJWc5uKQ092NEIgmTJlaWvklclDryekJz/mSQfZ0QMWso4mgtnPi5H8mBDGeckTUmuKxCnpNL+ocyLMuc/rksl841v5vjSDFypb7rn9Zs9Acc/Px12Av9gAAAAAElFTkSuQmCC>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGgAAAAYCAYAAAAWPrhgAAAC2ElEQVR4Xu2YWahNURjHl3nK9CKZkpnrQYhCxqKQF6UUL4QnQ5kznQdDKMlMhgwRJR5MxX3xItODJCUPkkTKowce+P+t/+rsvnvOufvc0713n9P61e9hf2vv1dn723t93zrORSLNwBA50g5UIXVwuKwZYoIyzAD4WHYwY9UI7+Gu7GzGGjAQPoR/5DuXfxj0K7wEe8rWgL9jqqwVpsjjdqAQ/eFfOcOMtYPf4QHZ0oyDb2ywhngLe9igJSao9UiVoGXwh2xvxnjxb7hdtjSH4EUbrCGuwbk2aDkPb0rLTvgKdpPF2CB3pXSCbIx6uM4GwRh5C56GR+RmeBKukmnoC7fIJ3BQYmwpvJM4TsN4+F7ymbaFq6VtcrbC3SbWgM/wtgwPkBdS3nCjnUYz8gFOt0FwQ3Zy/veFJXoyfObK++K3wY7yJ5yUGLsK9yWO03AYzpJ8hqdc8bZ6nvNNWkligjKcIC4zvLHRMmtwmRhhgy6/cSVz4BfJ5aRcBsP5kkWbtJHf4ELFyiFczyaLL1ExxsKPNpiE6y73OpWyXtpaU0y+6bQxuP7zwZUiBy9L0tWVv2+7IPfomN0j5cvbO5yUEtbqfjIwSloWwZc2mOS+859xVtnvfOKTDIOf5FD4HK6RZJPzXwUlfDArJN/qQtTLJTrOyRc6DoS5whdi6QNPOF/4KZe1jc4vk4WWSpaQKzaYJCbIk6kETYPX5S/n193kDWaJ5fCYiXFj/VrulU8lX7bF+VP/s8Plm4hidZaJoffgWedrDz2YPMnl5ypWs9mYMHGsW5Sb/DPO7y/tHpNwjAmvWrhRZlfWXTaVtTJZFwL8IhdIwn0RN+d0djgpQZin0Fxp6SXZBJVqIqoC/qnI1pU2BRbtsJEtxDn4SJIcfCDtMhbmqhSuCnSlHahGYoKqgFBr6uxAClgr2EzQQsx0+fmPOp+wLtIS5qqEic6386Glj0QikUgkE/wDWXjFyH/MbMkAAAAASUVORK5CYII=>

[image25]: <data:image/png;base64,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>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGgAAAAYCAYAAAAWPrhgAAADRklEQVR4Xu2ZW4hNYRTHl/v9WuTajHEb8uBJbrnmRXLJPUqU2wuRS0iNcUmk3EoiihAPPEiSkEmIUCJFygMSL0p4Eus/3/+z91lnH2efOdOcc8b+1a9pf99uzfnO+va31p4RyU53tSlNKDIGqRfUM7RV6nRCoUkSVKS0oSf4czRdH74pISETZbTSTpQos2mjoLd6k7Ywcx40Nr65qQ960PogKs5CusxOlCL/dYL6qtfVX/SlBF8G/CCuWehECwE+xxgahW9sfHOTb2MzQ71CN5i5XKlWr6qTqOWaWmEHLeXqbzo+dUqaiUvSPtrQDFWf28EQ4cbGNzf5NDYj1U2h69Xq5NB1XFbQKWpz9Ti1m3yeesCMpbFE/UyRkDDt1R/qNtrQ7FVP28FGBJ7+e3bQkiSocMRK0Cn1IrVsVZ+oHWgmZtHtMZ1Ds3FbXWsHxR198JJ6TD1EUTOOqstpHJqI+x0QdWdmaK6/uCM+F9Cs3KWo6zfUGjo/dJ/niwRHdBr4cO/Vy9R/gZvpRrX137sbnteSXhfBOYqGoKUENXSUel/cxoJxWCpBF4jvIFyD0GXdCV3nCuoPkr6LRoH4WGPUOmWEuIUNocXGK3HHgKUfBSjin2hd2uxyCbrUb2qv0Bw2wY7QdVzwOSAagOlmznJe3EaIbLmTBBV5graoH+1gHcC5DW2tySTO4qjz2ILjYaodNFSrZynAWd6ZxmUxvcVrHP0QSY96f/kXSAxafdiNY9hksMzfFOKhBDU8Dbyk+oUVI7vVNWasQn1LsehH6ioK1ol7KiDoIu59BnblmAVJhkd4jaTAn5Jag30sxImKheSg6TpJ0QgNF9fMQNRLy3d1MK1lrLjHCqKFfiGpCywmFqmHzRiOoGfUF98ais02N7i1lnESNBErzZxnIEWDcVB9Sv0T5fGxECcq1jR1gtqO4gR4I8F/Byw9xTVCKSQJSqeoElRKYJEPJPt7WDYq6QI7oXSU9CL9mFaFxjw+TlSsXMGrTL5/7ys42Hn7aV2pouF64sFT+pViE0xU31HscEuVuDhRseLSh2ITYIOUPL6ID7MTMWgrrhXP9EdPHJl76E5xxRzdVlTH5WPlC9pvOMBOlCpJghKKgz9LONgzQk9DwQAAAABJRU5ErkJggg==>

[image27]: <data:image/png;base64,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>

[image28]: <data:image/png;base64,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>

[image29]: <data:image/png;base64,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>

[image30]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAXCAYAAACS5bYWAAABZklEQVR4Xu3VPShGURzH8eM1g0HJIMvjbbDJYhJFXhYGi5JsFCZlkAwo7yyeQSkDBhkpCya7spFB2WyMMvD7u79z3Hu69z4PdeXW+danp+ec89Q53adzlXK5XInXTEewBpfU7V/0Bw3BIe3AGdSRKVWbfaR+ft+mByjQixKuCZ6gkaQb2CfTPU3x+zI9mxXR9SjvQFGHaqMye8KqHt7U91OWruGEQiuBW5qw5sLqgCzJb/31wQr9tAy8QiuFlprNDsIenFNVcDqyatqAUhqAYf+iPJumC+UdspgimyH5w9dYc3HJhvV/bNya+02ncEWF1pzp325Wrol5quWiTvqAUY7lk9wms3QA5cHp2LpgTgUf+4Ly9iAyMiB3qx5o56JJeocGjsU1Rr2+sQpYh0rK1Sa8KO+A+pDHcEfmalykJX7qt4icNlcj0EJ2cjusUpE1ZyeHk7eWfspbsKuCL4mvUrVZl8uVQJ+s1l8MLiIGUgAAAABJRU5ErkJggg==>