[License: arXiv.org perpetual non-exclusive license](https://info.arxiv.org/help/license/index.html#licenses-available)

arXiv:2512.10685v1 [cs.CV] 11 Dec 2025

Lars MeschederWei DongShiwei LiXuyang BaiMarcel SantosPeiyun HuBruno LecouatMingmin ZhenAmaël DelaunoyTian FangYanghai TsinStephan R. RichterVladlen KoltunApple

###### Abstract

We present SHARP 1, an approach to photorealistic view synthesis from a single image. Given a single photograph, SHARP regresses the parameters of a 3D Gaussian representation of the depicted scene. This is done in less than a second on a standard GPU via a single feedforward pass through a neural network. The 3D Gaussian representation produced by SHARP can then be rendered in real time, yielding high-resolution photorealistic images for nearby views. The representation is metric, with absolute scale, supporting metric camera movements. Experimental results demonstrate that SHARP delivers robust zero-shot generalization across datasets. It sets a new state of the art on multiple datasets, reducing LPIPS by 25–34% and DISTS by 21–43% versus the best prior model, while lowering the synthesis time by three orders of magnitude.

### 1Introduction

![Refer to caption](https://arxiv.org/html/2512.10685v1/x1.png)

Figure 1:Synthesis time on a single GPU versus image fidelity on the ScanNet++ dataset.

|   |   |   |   |
|---|---|---|---|
|Input|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/teaser/UnsplashCurated/ground_truth/source/_5wkyNA2BPc_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/teaser/UnsplashCurated/ground_truth/source/_5IPpJvbByo_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/teaser/UnsplashCurated/ground_truth/source/_5_gyzzeNPw_0000_0001.jpg)|
|SHARP (ours)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/teaser/UnsplashCurated/sharp_public_aligned/inlet_target/_5wkyNA2BPc_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/teaser/UnsplashCurated/sharp_public_aligned/inlet_target/_5IPpJvbByo_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/teaser/UnsplashCurated/sharp_public_aligned/inlet_target/_5_gyzzeNPw_0000_0001.jpg)|
|Gen3C|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/teaser/UnsplashCurated/gen3c_aligned/inlet_target/_5wkyNA2BPc_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/teaser/UnsplashCurated/gen3c_aligned/inlet_target/_5IPpJvbByo_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/teaser/UnsplashCurated/gen3c_aligned/inlet_target/_5_gyzzeNPw_0000_0001.jpg)|
|ViewCrafter|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/teaser/UnsplashCurated/view_crafter_aligned/inlet_target/_5wkyNA2BPc_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/teaser/UnsplashCurated/view_crafter_aligned/inlet_target/_5IPpJvbByo_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/teaser/UnsplashCurated/view_crafter_aligned/inlet_target/_5_gyzzeNPw_0000_0001.jpg)|
|Flash3D|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/teaser/UnsplashCurated/flash3d_aligned/inlet_target/_5wkyNA2BPc_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/teaser/UnsplashCurated/flash3d_aligned/inlet_target/_5IPpJvbByo_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/teaser/UnsplashCurated/flash3d_aligned/inlet_target/_5_gyzzeNPw_0000_0001.jpg)|
|TMPI|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/teaser/UnsplashCurated/tmpi_aligned/inlet_target/_5wkyNA2BPc_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/teaser/UnsplashCurated/tmpi_aligned/inlet_target/_5IPpJvbByo_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/teaser/UnsplashCurated/tmpi_aligned/inlet_target/_5_gyzzeNPw_0000_0001.jpg)|

Figure 2:SHARP synthesizes a photorealistic 3D representation from a single photograph in less than a second. The synthesized representation supports high-resolution rendering of nearby views, with sharp details and fine structures, at more than 100 frames per second on a standard GPU. We illustrate on photographs from Unsplash ([2022](https://arxiv.org/html/2512.10685v1#bib.bib54)).

Imagine revisiting a precious memory captured on camera. What if technology could lift the scene out of the image plane, recreating the three-dimensional world as it was then, putting you back in the scene? High-resolution low-latency AR/VR headsets can convincingly present spatial content. 3D representations can also be rendered on handheld displays. Can these surfaces be used to reconnect us with our memories in new ways?

Recent advances in neural rendering (Tewari et al., [2022](https://arxiv.org/html/2512.10685v1#bib.bib51)) have demonstrated remarkable success in synthesizing photorealistic views, but many of the most impressive results leverage multiple input images and conduct time-consuming per-scene optimization. We are interested in view synthesis from a single photograph, to support real-time photorealistic rendering from nearby views. Specifically, our application setting yields the following desiderata. (a) Fast synthesis of a 3D representation from a single photograph, to support interactive browsing of personal photo collections. (b) Real-time photorealistic rendering of the resulting 3D representation from nearby views. We wish to support natural posture shifts in AR/VR headsets, providing the experience of looking at a stable 3D scene from different perspectives, but need not support substantial travel (“walking around”) within the photograph. (c) The 3D representation should be metric, with absolute scale, to accurately couple the virtual camera with a physical headset or another physical device.

In this paper, we present SHARP (Single-image High-Accuracy Real-time Parallax), our approach to meeting these desiderata. Given a photograph, SHARP produces a 3D Gaussian representation (Kerbl et al., [2023](https://arxiv.org/html/2512.10685v1#bib.bib21)) of the depicted scene via a single forward pass through a neural network. This representation can then be rendered in real time from nearby views. Though the high-level approach (single image in, 3D Gaussian representation out) echoes prior work, SHARP delivers state-of-the-art visual fidelity while keeping the generation time under one second on an A100 GPU. (See Figure [1](https://arxiv.org/html/2512.10685v1#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Sharp Monocular View Synthesis in Less Than a Second").) The key ingredients are scale and a number of technical choices whose importance we validate via controlled experiments.

First, we design a neural network that regresses a high-resolution 3D Gaussian representation from a single photograph. While our network comprises multiple modules, it is trained end-to-end to optimize view synthesis fidelity. Second, we introduce a carefully designed loss configuration that prioritizes the accuracy of synthesized views while regularizing away common artifacts. Third, we introduce a learned depth adjustment module that is used during training to facilitate view synthesis supervision in the presence of inaccurate depth estimates. Figure [2](https://arxiv.org/html/2512.10685v1#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Sharp Monocular View Synthesis in Less Than a Second") shows some views synthesized by SHARP and a number of baselines.

We conduct a thorough experimental evaluation on multiple datasets that were not used during training, using powerful perceptual metrics such as LPIPS (Zhang et al., [2018](https://arxiv.org/html/2512.10685v1#bib.bib69)) and DISTS (Ding et al., [2022](https://arxiv.org/html/2512.10685v1#bib.bib7)) to assess image fidelity. SHARP improves image fidelity by substantial factors versus prior feed-forward methods. In comparison to diffusion-based systems, SHARP delivers higher fidelity while reducing synthesis time by two to three orders of magnitude. Compared to the strongest prior method (Ren et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib37)), SHARP reduces LPIPS by 25–34% and DISTS by 21–43% across the test datasets (in the zero-shot regime), while accelerating synthesis by three orders of magnitude and producing a 3D representation that supports high-resolution rendering of nearby views at 100 frames per second.

In summary, our contributions are as follows

- • 
    
    End-to-end architecture: we design a novel network architecture that can be trained end-to-end to predict high-resolution 3D Gaussian representations.
    
- • 
    
    Robust and effective loss configuration: we carefully choose a series of loss functions to prioritize view synthesis quality while maintaining training stability and suppressing common visual artifacts.
    
- • 
    
    Depth alignment module: we introduce a simple module that can effectively resolve depth ambiguities during training, a fundamental challenge for regression-based view synthesis methods.
    

Using our insights, we demonstrate that state-of-the-art high-resolution view synthesis is feasible in a purely regression based framework.

### 2Related Work

View synthesis from multiple images. Early image-based rendering approaches synthesized new views with minimal 3D modeling. Chen & Williams ([1993](https://arxiv.org/html/2512.10685v1#bib.bib6)) introduced view interpolation, enabling transitions between captured viewpoints. QuickTime VR (Chen, [1995](https://arxiv.org/html/2512.10685v1#bib.bib5)) created navigable environments from panoramic images. Layered Depth Images (Shade et al., [1998](https://arxiv.org/html/2512.10685v1#bib.bib45)) addressed occlusions by storing multiple depth values per pixel. Kang et al. ([2007](https://arxiv.org/html/2512.10685v1#bib.bib19)) survey the early years of image-based rendering.

More recently, deep learning and GPU acceleration transformed view synthesis from multiple images. Free View Synthesis (Riegler & Koltun, [2020](https://arxiv.org/html/2512.10685v1#bib.bib38)) combined geometric scaffolds with learned features to synthesize novel views from distributed viewpoints. Stable View Synthesis (Riegler & Koltun, [2021](https://arxiv.org/html/2512.10685v1#bib.bib39)) improved on this by enhancing stability and consistency across views. Neural radiance fields (NeRF) (Mildenhall et al., [2020](https://arxiv.org/html/2512.10685v1#bib.bib29)) introduced continuous implicit representations that support remarkable levels of photorealism (Barron et al., [2023](https://arxiv.org/html/2512.10685v1#bib.bib1)). 3D Gaussian Splatting (Kerbl et al., [2023](https://arxiv.org/html/2512.10685v1#bib.bib21)) significantly accelerated rendering while maintaining visual fidelity through explicit 3D primitives. We use the 3D Gaussian representation developed by Kerbl et al. ([2023](https://arxiv.org/html/2512.10685v1#bib.bib21)), but apply it in the context of view synthesis from a single image.

A number of works develop feed-forward prediction models for view synthesis from a small number of nearby views. IBRNet (Wang et al., [2021](https://arxiv.org/html/2512.10685v1#bib.bib56)) generalized image-based rendering across scenes using learned features and ray transformers. MVSNeRF (Chen et al., [2021](https://arxiv.org/html/2512.10685v1#bib.bib3)) reconstructed neural radiance fields from a few input images via cost volume processing. LaRa (Chen et al., [2024](https://arxiv.org/html/2512.10685v1#bib.bib4)) regressed an object-level radiance field from sparse input images. GS-LRM (Zhang et al., [2024](https://arxiv.org/html/2512.10685v1#bib.bib68)) leveraged a transformer to predict a 3D Gaussian representation from posed sparse images. Our work focuses on view synthesis from a single image.

View synthesis from a single image. The introduction of larger datasets (Zhou et al., [2018](https://arxiv.org/html/2512.10685v1#bib.bib73); Tongue et al., [2024](https://arxiv.org/html/2512.10685v1#bib.bib52)) enabled the transition from scene-specific multi-view optimization to learning-based pipelines that can infer a plausible 3D representation from a single image. Zhou et al. ([2016](https://arxiv.org/html/2512.10685v1#bib.bib72)) synthesized novel views from a single image through appearance flow. Subsequent work developed variants of depth-based warping (Wiles et al., [2020](https://arxiv.org/html/2512.10685v1#bib.bib61); Jampani et al., [2021](https://arxiv.org/html/2512.10685v1#bib.bib16)) and multiplane images (MPI) (Zhou et al., [2018](https://arxiv.org/html/2512.10685v1#bib.bib73); Tucker & Snavely, [2020](https://arxiv.org/html/2512.10685v1#bib.bib53)). AdaMPI (Han et al., [2022](https://arxiv.org/html/2512.10685v1#bib.bib14)) adapted multiplane images to diverse scene layouts through plane depth adjustment and depth-aware color prediction, trained using a warp-back strategy on single-view image collections. Khan et al. ([2023](https://arxiv.org/html/2512.10685v1#bib.bib22)) proposed Tiled Multiplane Images (TMPI), which splits an MPI into many small tiled regions with fewer depth planes per tile, reducing computational overhead while maintaining quality. Several recent methods have drawn inspiration from the success of transformers in modeling long-range dependencies, leveraging large transformer-based encoder-decoder architectures to infer scene structure and appearance implicitly for novel views (Hong et al., [2023](https://arxiv.org/html/2512.10685v1#bib.bib15); Jin et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib17)).

PixelNeRF (Yu et al., [2021](https://arxiv.org/html/2512.10685v1#bib.bib65)) trained convolutional networks to predict an object-level neural radiance field from a single image. Splatter image (Szymanowicz et al., [2024](https://arxiv.org/html/2512.10685v1#bib.bib48)) introduced direct prediction of per-pixel Gaussians via a U-Net. Flash3D (Szymanowicz et al., [2025a](https://arxiv.org/html/2512.10685v1#bib.bib49)) incorporated a pre-trained depth prediction network for generalization to more complex scenes. Schwarz et al. ([2025](https://arxiv.org/html/2512.10685v1#bib.bib44)) proposed a recipe for generating 3D worlds from a single image by decomposing this task into a number of steps and leveraging diffusion models.

Diffusion models have emerged as powerful tools for novel view synthesis with sparse input, offering high-quality results through iterative denoising processes (Po et al., [2023](https://arxiv.org/html/2512.10685v1#bib.bib32)). Watson et al. ([2022](https://arxiv.org/html/2512.10685v1#bib.bib59)) developed an early application to view synthesis. Zero-1-to-3 (Liu et al., [2023](https://arxiv.org/html/2512.10685v1#bib.bib26)) demonstrated zero-shot view synthesis by fine-tuning diffusion models on 3D object datasets. iNVS (Kant et al., [2023](https://arxiv.org/html/2512.10685v1#bib.bib20)) repurposed diffusion inpainters for view synthesis by combining monocular depth estimation with inpainting. NerfDiff (Gu et al., [2023](https://arxiv.org/html/2512.10685v1#bib.bib13)) distilled a 3D-aware diffusion model into NeRF by synthesizing virtual views to improve rendering under occlusion. These early attempts focused on object-level view synthesis.

At the scene level, ViewCrafter (Yu et al., [2024](https://arxiv.org/html/2512.10685v1#bib.bib67)), ZeroNVS (Sargent et al., [2024](https://arxiv.org/html/2512.10685v1#bib.bib41)), CAT3D (Gao et al., [2024](https://arxiv.org/html/2512.10685v1#bib.bib10)), SplatDiff (Zhang et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib70)), Stable Virtual Camera (Zhou et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib71)), Bolt3D (Szymanowicz et al., [2025b](https://arxiv.org/html/2512.10685v1#bib.bib50)), Wonderland (Liang et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib25)), WonderWorld (Yu et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib66)), See3D (Ma et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib28)) and Gen3C (Ren et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib37)) all applied diffusion models to view synthesis from sparse image sets or a single image. The diffusion-based approach supports impressive image quality from faraway viewpoints, leveraging diffusion priors to synthesize plausible appearance even for views that have no overlap with the input. On the other hand, image quality from nearby views (corresponding to natural head motion or posture shifts) can be noticeably less sharp and photorealistic than the input, while the synthesis time can sometimes stretch into minutes. (Although Bolt3D (Szymanowicz et al., [2025b](https://arxiv.org/html/2512.10685v1#bib.bib50)) makes impressive progress on the latter front.) In contrast, we aim for real-time rendering of maximally photorealistic high-resolution images from nearby views, supporting a headbox that allows for natural posture shifts while maintaining photographic quality. Our approach generates a high-resolution 3D representation that provides such experiences from single-image input in less than a second on a single GPU, supporting conversion of pre-existing photographs to photorealistic 3D during interactive browsing of a photo collection.

### 3Method

#### 3.1Overview

Our approach, SHARP, generates a 3D Gaussian representation from a single image via a forward pass through a neural network. The input to the network is a single monocular RGB image 𝐈∈ℝC×H×W, where C=3 denotes the number of color channels, H is the height, and W is the width of the image. The output is a set of 3D Gaussians 𝐆∈ℝK×N, which can be rendered to arbitrary views using a differentiable renderer. Here K=14 is the number of Gaussian attributes (3 for the position, 3 for scale, 4 for orientation, 3 for color and 1 for opacity) and N is the number of output Gaussians. In practice, SHARP outputs 2×768×768≈1.2 million Gaussians per image, parameterized over a 768×768 grid with two layers. We do not use spherical harmonics (Kerbl et al., [2023](https://arxiv.org/html/2512.10685v1#bib.bib21)), because the number of spherical harmonic coefficients grows quadratically with the order of the spherical harmonics and would lead to a large increase in output size. Figure [3](https://arxiv.org/html/2512.10685v1#S3.F3 "Figure 3 ‣ 3.1 Overview ‣ 3 Method ‣ Sharp Monocular View Synthesis in Less Than a Second") provides an overview of our method. The following paragraphs describe the modules in more detail.

![Refer to caption](https://arxiv.org/html/2512.10685v1/x2.png)

Figure 3:Our model consists of four learnable modules (Section [3.1](https://arxiv.org/html/2512.10685v1#S3.SS1 "3.1 Overview ‣ 3 Method ‣ Sharp Monocular View Synthesis in Less Than a Second")): a pretrained encoder for feature extraction, a depth decoder that produces two distinct depth layers, a depth adjustment module, and a Gaussian decoder that refines all Gaussian attributes. The differentiable Gaussian initializer and composer assemble the Gaussians for the resulting 3D representation. The predicted Gaussians are rendered to the input and novel views for loss computation (Section [3.4](https://arxiv.org/html/2512.10685v1#S3.SS4 "3.4 Training Objectives ‣ 3 Method ‣ Sharp Monocular View Synthesis in Less Than a Second")).

##### Monodepth backbone.

The input image 𝐈∈ℝ3×H×W is fed into a pretrained Depth Pro image encoder φenc to produce 4 intermediate features maps (𝐟i)i∈1,…,4=φenc​(𝐈). As in Depth Pro (Bochkovskii et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib2)), we resize the input image so that H=W=1536. We then feed the intermediate feature maps into the Depth Pro decoder φdec to produce a monocular depth map 𝐃^=φdec​((𝐟i)i∈1,…,4). Similar to Flynn et al. ([2019](https://arxiv.org/html/2512.10685v1#bib.bib9)), we duplicate the last convolutional layer of the decoder to produce a two-channel depth map 𝐃^∈ℝ2×H×W.

One of our key observations is that depth is ill-defined and using a frozen monodepth model can degrade view synthesis fidelity, particularly for transparent or reflective surfaces (Wen et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib60)). During training, we therefore unfreeze both φdec and the low-resolution encoder part of φenc. This enables the full view synthesis training to adapt the depth prediction modules via backpropagation, in conjunction with downstream modules, for the end-to-end view synthesis objectives.

##### Depth adjustment.
AIzaSyD 6 rU 9 nx 7 poOiymiO 1 GB 6 fHUGVO_NZgxyc
Although monocular depth estimation has made impressive advances in recent years, the depth estimator still needs to deal with the inherent ambiguity of the task. In monocular depth estimation, the network might just resolve the problem by predicting outputs at the mean scale of possible outcomes (Poggi et al., [2020](https://arxiv.org/html/2512.10685v1#bib.bib33)). When depth estimates are used for view synthesis, however, this ambiguity can lead to visual artifacts.

To address this, we take inspiration from the line of work on Conditional Variational Autoencoders (C-VAE) (Sohn et al., [2015](https://arxiv.org/html/2512.10685v1#bib.bib46)), which addresses the ambiguity by designing a posterior model. In a traditional C-VAE, the posterior would take ground-truth depth 𝐃∈ℝH×W as input and produce a latent representation 𝐳. During training this latent vector would be passed through an information bottleneck in the form of a KL divergence. This ensures that the latent represents the smallest amount of information required to resolve the ambiguity of the task. We simplify this scheme and adapt it to our setting by interpreting 𝐳 as a scale map 𝐒∈ℝH×W and replacing the KL divergence with a task-specific regularizer. More details are given in Section [3.4](https://arxiv.org/html/2512.10685v1#S3.SS4 "3.4 Training Objectives ‣ 3 Method ‣ Sharp Monocular View Synthesis in Less Than a Second"). The output of this module is an adjusted two-layer depth map 𝐃¯=𝐒​(𝐃^,𝐃)⊙𝐃^.

##### Gaussian initializer.

We use this adjusted two-layer depth map 𝐃¯∈ℝ2×H×W and the input image 𝐈∈ℝ3×H×W to initialize a set of base Gaussians 𝐆0∈ℝK×2×H′×W′, where H′=H/2 and W′=W/2.

To compute 𝐆0​(𝐈,𝐃¯), we first subsample 𝐈 and 𝐃¯ by a factor of 2, using average and min-pooling, respectively. This yields a downsampled depth map 𝐃¯′ and input image 𝐈′. We then unproject the resulting depth map 𝐃¯′ to produce mean vectors μ​(i,j)=[i⋅𝐃¯′​(i,j),j⋅𝐃¯′​(i,j),𝐃¯′​(i,j)]T. Note that we deliberately do not use the intrinsics matrix of the input image here. This enables the network to reason about Gaussian attributes in a normalized space without having to adapt its predictions to the field of view of the image. We set the scale proportional to depth: s​(i,j)=s0⋅𝐃¯′​(i,j) with a fixed scale factor s0. The color is initialized directly from the downsampled input image c​(i,j)=𝐈′​(i,j). The rotation and opacity are initialized to a unit quaternion [1,0,0,0]T and a fixed value of 0.5, respectively.

##### Gaussian decoder.

While the initial Gaussians provide a reasonable starting point, they require substantial refinement to achieve high-fidelity rendering. The Gaussian decoder φgauss takes as input the feature maps (𝐟i)i∈1,…,4 and the input image 𝐈, and outputs refinements Δ​𝐆∈ℝK×2×H′×W′ for all Gaussian attributes:

|   |   |   |   |
|---|---|---|---|
||Δ​𝐆=φgauss​((𝐟i)i∈1,…,4,𝐈).||(3.1)|

These refinements include deltas for position Δ​𝐆pos∈ℝ3×2×H′×W′, scale Δ​𝐆scale∈ℝ3×2×H′×W′, rotation Δ​𝐆rot∈ℝ4×2×H′×W′, color Δ​𝐆color∈ℝ3×2×H′×W′, and opacity Δ​𝐆alpha∈ℝ1×2×H′×W′. The ability to refine Gaussians across all attributes is crucial for creating a coherent 3D representation that models detailed geometry and appearance.

##### Gaussian composer.

The Gaussian composer takes the base Gaussians 𝐆0∈ℝK×2×H′×W′ and Gaussian refinements Δ​𝐆∈ℝK×2×H′×W′ as input and produces the final Gaussian attributes 𝐆∈ℝK×2×H′×W′. Instead of directly adding the values, we compose them with an attribute-specific activation function γattr:

|   |   |   |   |
|---|---|---|---|
||𝐆attr=γattr​(γattr−1​(𝐆0,attr)+ηattr​Δ​𝐆attr),||(3.2)|

The supplement provides the details on the activation functions γattr and scale factors ηattr.

##### Gaussian renderer.

The resulting Gaussian representation can be rendered from arbitrary viewpoints using an in-house differentiable renderer ℛ. The rendering process can be expressed as 𝐈^=ℛ​(𝐆,𝐏), where 𝐈^ is the rendered image, 𝐆 are the final Gaussian attributes, and 𝐏 represents the camera projection parameters for the desired viewpoint. Since we predict the Gaussians in normalized space, we would theoretically need to transform them using the extrinsics and intrinsics of the source view. However, we can alternatively incorporate this transformation directly into the projection matrix for the target view: 𝐏=𝐊tgt​𝐄tgt​𝐄src−1​𝐊src−1, where 𝐊src and 𝐄src are the intrinsic and extrinsic matrices of the source view, and 𝐊tgt and 𝐄tgt are those of the target view. In contrast to image diffusion models, the inference cost is amortized: once a 3D representation is synthesized, it can be rendered in real time from new viewpoints.

#### 3.2Network Architecture

Our architecture includes a number of trainable modules, as illustrated in Figure [3](https://arxiv.org/html/2512.10685v1#S3.F3 "Figure 3 ‣ 3.1 Overview ‣ 3 Method ‣ Sharp Monocular View Synthesis in Less Than a Second"). The complete network has approximately 340M trainable parameters (702M parameters in total). It processes a single 1536×1536 image and produces approximately 1.2 million Gaussians in under one second on a single GPU.

Feature encoder. We base our feature encoder on the Depth Pro backbone (Bochkovskii et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib2)). This encoder processes the input image 𝐈∈ℝC×H×W and produces four feature maps (𝐟i)i∈1,…,4 at different resolutions. The Depth Pro backbone consists of two Vision Transformers (ViTs) (Dosovitskiy et al., [2021](https://arxiv.org/html/2512.10685v1#bib.bib8)), one applied to a downscaled version of the input image and one applied to various image patches. The low-resolution image encoder and patch encoder each has 326M parameters. During training, we unfreeze the low-resolution image encoder to allow adaptation to the view synthesis task, while keeping the patch encoder and normalization layers frozen to preserve the pretrained feature extraction capabilities.

Depth decoder. Our depth decoder is based on the Dense Prediction Transformer (DPT) (Ranftl et al., [2021](https://arxiv.org/html/2512.10685v1#bib.bib35)). We modify the original DPT decoder by duplicating the final convolutional layer to output two depth channels instead of one. Our encoder thus takes the feature maps (𝐟i)i∈1,…,4 from the encoder and produces a two-layer depth map 𝐃^∈ℝ2×H×W. The first layer represents the primary visible surfaces, while the second layer may represent occluded regions and view-dependent effects. The depth decoder consists of multiple convolutional blocks with approximately 20M parameters. This module is fully unfrozen during training to optimize depth prediction for view synthesis.

Gaussian decoder. The Gaussian decoder predicts refinements for all Gaussian attributes. It has the same DPT architecture as the depth decoder but we replace the last upsampling block with a custom prediction head. The decoder takes as input the feature maps (𝐟i)i∈1,…,4, the input image 𝐈, and the predicted depth maps 𝐃^. It outputs a tensor Δ​𝐆∈ℝK×2×H′×W′ that contains deltas for all Gaussian attributes: position (3 channels), scale (3 channels), rotation (4 channels), color (3 channels), and opacity (1 channel). This decoder has approximately 7.8M parameters and is trained from scratch. The high dimensionality of the output (approximately 16.5M values) enables fine-grained control over the Gaussian representation.

Depth adjustment. For the depth adjustment network we use a small U-Net (Ronneberger et al., [2015](https://arxiv.org/html/2512.10685v1#bib.bib40)) with 2M parameters that takes both the predicted inverse depth 𝐃^−1 and the corresponding ground truth 𝐃−1 as inputs and produces a scale map 𝐒∈ℝH×W. During inference we replace the depth adjustment module with the identity function.

#### 3.3Training Strategy

##### Supervision.

We supervise predicted 3D Gaussians in image space through differentiable rendering. Each training sample consists of two views: the input view and the novel view. We predict Gaussians from the input view, render in both views, and evaluate losses on these renderings. The losses are defined in Section [3.4](https://arxiv.org/html/2512.10685v1#S3.SS4 "3.4 Training Objectives ‣ 3 Method ‣ Sharp Monocular View Synthesis in Less Than a Second"). We use a two-stage curriculum.

##### Stage 1: Synthetic training.

We first train on synthetic data with perfect image and depth ground truth for both the input view and the novel view, allowing the network to learn fundamental principles of 3D reconstruction without real-world ambiguities. The synthetic data is further described in the supplement.

##### Stage 2: Self-supervised finetuning (SSFT).

We fine-tune the model on real images that have no ground truth for view synthesis. To this end, we use our trained model to generate pseudo ground truth on single-view real images from OpenScene ([2023](https://arxiv.org/html/2512.10685v1#bib.bib30)) and online resources, detailed in the supplement. For each real image, we generate a 3D Gaussian representation and render a pseudo-novel view. We then use the pseudo-novel view as the input view, and the real input image as the novel view. The swapping of input and novel views forces the network to adapt to real images, enhancing its ability to generate coherent novel views.

Unlike AdaMPI (Han et al., [2022](https://arxiv.org/html/2512.10685v1#bib.bib14)), which constructs stereo pairs from single-view collections using a warp-back strategy, our approach leverages the 3D representation generated by our model to create pseudo-novel views. This maintains geometric consistency while adapting to real images without requiring stereo pairs.

#### 3.4Training Objectives

We train our network using a combination of loss functions:

Rendering losses. We apply an L1 loss between the rendered image 𝐈^ and the ground truth 𝐈 on both input and novel views:

|   |   |   |   |
|---|---|---|---|
||ℒcolor=∑view∈{input, novel}𝔼p∼Ω​[\|𝐈^view​(p)−𝐈view​(p)\|],||(3.3)|

where Ω denotes the set of all pixels p. We further use a perceptual loss (Johnson et al., [2016](https://arxiv.org/html/2512.10685v1#bib.bib18); Gatys et al., [2016](https://arxiv.org/html/2512.10685v1#bib.bib11); Suvorov et al., [2021](https://arxiv.org/html/2512.10685v1#bib.bib47)) on novel views to encourage plausible inpainting:

|   |   |   |   |
|---|---|---|---|
||ℒpercep=∑l=14λlfeat⋅‖ϕl​(𝐈^novel)−ϕl​(𝐈novel)‖2+λlGram⋅‖Ml​(𝐈^novel)−Ml​(𝐈novel)‖2,||(3.4)|

where ϕl and Ml are the l-th layer of our feature extractor and its Gram matrix, respectively. We apply a Binary Cross Entropy (BCE) loss to penalize rendered alpha on the input view to discourage spurious transparent pixels:

|   |   |   |   |
|---|---|---|---|
||ℒalpha=∑view∈{input, novel}𝔼p∼Ω​[ℒBCE​(𝐀^view​(p),1)],||(3.5)|

where 𝐀^view is the rendered alpha image.

Depth losses. We apply an L1 loss between the predicted and ground-truth disparity, only on the input view, exclusively on the first depth layer:

|   |   |   |   |
|---|---|---|---|
||ℒdepth=𝔼p∼Ω​[\|𝐃¯(1)−1​(p)−𝐃−1​(p)\|],||(3.6)|

where 𝐃¯(1) and 𝐃 are the first predicted depth layer and the ground-truth depth, respectively.

Regularizers. We apply a total variation regularizer on the second depth layer to promote smoothness:

|   |   |   |   |
|---|---|---|---|
||ℒtv=𝔼p∼Ω​[\|∇x𝐃¯(2)−1​(p)\|+\|∇y𝐃¯(2)−1​(p)\|],||(3.7)|

where 𝐃¯(2) is the second predicted depth layer. Additionally, we apply a regularizer to suppress floaters with large disparity gradients:

|   |   |   |   |
|---|---|---|---|
||ℒgrad=𝔼i∼ℐ​[𝐆alpha​(i)⋅(1−exp⁡(−1σ​max⁡{0,\|∇𝐃¯−1​(π​(𝐆0​(i)))\|−ϵ}))],||(3.8)|

where ℐ is the index set for the Gaussians and π​(⋅) computes the projection of the Gaussian position onto the 2D image plane. We use σ=ϵ=10−2. We further constrain Gaussian offset magnitudes Δ​𝐆x,Δ​𝐆y to discourage extreme deviations from the base Gaussians:

|   |   |   |   |
|---|---|---|---|
||ℒdelta=𝔼i∼ℐ​[max⁡{\|Δ​𝐆x​(i)\|−δ,0}+max⁡{\|Δ​𝐆y​(i)\|−δ,0}]||(3.9)|

with δ=400.0. In screen space, we regularize the variance of projected Gaussians:

|   |   |   |   |
|---|---|---|---|
||ℒsplat=𝔼i∼ℐ​[max⁡{σ​(𝐆​(i))−σmax,0}+max⁡{σmin−σ​(𝐆​(i)),0}],||(3.10)|

where σ​(⋅) computes the projected Gaussian variance and σmin=10−1,σmax=102.

Depth adjustment. We regularize the depth adjustment with an MAE loss and a multiscale total variation regularizer:

|   |   |   |   |
|---|---|---|---|
||ℒscale=𝔼p∼Ω​[\|𝐒​(p)\|] and ℒ∇scale=∑k=16𝔼p∼Ω↓k​[\|∇𝐒↓k​(p)\|].||(3.11)|

Here 𝐒↓k denotes a scale map downsampled by a factor 2k on the downsampled image domain Ω↓k. The depth adjustment losses act as an information bottleneck, encouraging the network to learn the most compact representation to resolve depth ambiguities.

The final loss is a composition of all the loss terms:

|   |   |   |   |
|---|---|---|---|
||ℒ=∑d∈𝒟λd​ℒd+∑r∈ℛλr​ℒr+∑s∈𝒮λs​ℒs,||(3.12)|

where 𝒟={color,alpha,depth,percep}, ℛ={tv,grad,delta,splat}, 𝒮={scale,∇scale} are the attribute sets for the data terms and regularizers. The hyperparameters are specified in the supplement.

### 4Experiments

We first train our model for 100K steps on 128 A100 GPUs using synthetic data only (Stage 1). We then fine-tune our model using self-supervision for 60K steps on 32 A100 GPUs (Stage 2).

##### Datasets.

We evaluate our approach on multiple datasets with metric poses: Middlebury (Scharstein et al., [2014](https://arxiv.org/html/2512.10685v1#bib.bib42)), Booster (Ramirez et al., [2024](https://arxiv.org/html/2512.10685v1#bib.bib34)), ScanNet++ (Yeshwanth et al., [2023](https://arxiv.org/html/2512.10685v1#bib.bib63)), WildRGBD (Xia et al., [2024](https://arxiv.org/html/2512.10685v1#bib.bib62)), ETH3D (Schöps et al., [2017](https://arxiv.org/html/2512.10685v1#bib.bib43)), and Tanks and Temples (Knapitsch et al., [2017](https://arxiv.org/html/2512.10685v1#bib.bib24)). The sampling choices are discussed in the supplement. We do not include non-metric datasets such as RealEstate10K (Zhou et al., [2018](https://arxiv.org/html/2512.10685v1#bib.bib73)).

##### Evaluation metrics.

We employ LPIPS (Zhang et al., [2018](https://arxiv.org/html/2512.10685v1#bib.bib69)) and DISTS (Ding et al., [2022](https://arxiv.org/html/2512.10685v1#bib.bib7)) to quantitatively assess the quality of novel view synthesis. We focus primarily on these perceptual metrics, since older pointwise metrics such as PSNR and SSIM can be overly sensitive to small translations, where even a 1% shift can lead to catastrophic drops in scores despite visually similar results. (See the supplement for an illustration. We also list PSNR and SSIM numbers in the supplement for completeness.) Since we are interested in sharp high-resolution view synthesis, we evaluate all methods on the full-resolution ground truth. If a method generates results at lower resolution, we resize the output image to the input resolution before evaluation. If a method crops the input and generates cropped results (Ren et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib37); Yu et al., [2024](https://arxiv.org/html/2512.10685v1#bib.bib67); Jin et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib17); Zhou et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib71)), we evaluate against correspondingly cropped ground-truth images.

##### Baselines.

We compare SHARP to the following state-of-the-art methods: Flash3D (Szymanowicz et al., [2025a](https://arxiv.org/html/2512.10685v1#bib.bib49)), which is based on 3D Gaussians; TMPI (Khan et al., [2023](https://arxiv.org/html/2512.10685v1#bib.bib22)), which uses multi-plane images; LVSM (Jin et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib17)), which is based on image-to-image regression; and Stable Virtual Camera (SVC) (Zhou et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib71)), ViewCrafter (Yu et al., [2024](https://arxiv.org/html/2512.10685v1#bib.bib67)), and Gen3C (Ren et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib37)), which employ diffusion models.

##### Quantitative evaluation.

Table [1](https://arxiv.org/html/2512.10685v1#S4.T1 "Table 1 ‣ Quantitative evaluation. ‣ 4 Experiments ‣ Sharp Monocular View Synthesis in Less Than a Second") presents a quantitative evaluation of SHARP and the baselines in the zero-shot regime. (Cross-dataset generalization to datasets that were not used during training.) For each metric, we report the mean value over all test samples. SHARP achieves the highest accuracy on all metrics across all datasets. Additional experimental results are provided in the supplement.

Table 1:Quantitative evaluation. Lower is better. Best, second-best, and third-best in each column are highlighted.

|   |   |   |   |   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||Middlebury|   |Booster|   |ScanNet++|   |WildRGBD|   |Tanks and Temples|   |ETH3D|   |
||DISTS↓|LPIPS↓|DISTS↓|LPIPS↓|DISTS↓|LPIPS↓|DISTS↓|LPIPS↓|DISTS↓|LPIPS↓|DISTS↓|LPIPS↓|
|Flash3D|0.359|0.581|0.409|0.370|0.374|0.572|0.159|0.345|0.382|0.683|0.535|0.651|
|TMPI|0.158|0.436|0.232|0.409|0.128|0.309|0.114|0.327|0.309|0.693|0.396|0.720|
|LVSM|0.274|0.555|0.307|0.404|0.145|0.302|0.095|0.257|0.227|0.575|0.555|0.664|
|SVC|0.208|0.629|0.283|0.448|0.201|0.596|0.157|0.531|0.230|0.733|0.420|0.708|
|ViewCrafter|0.373|0.751|0.318|0.523|0.176|0.526|0.148|0.386|0.295|0.759|0.454|0.748|
|Gen3C|0.164|0.545|0.207|0.384|0.090|0.227|0.106|0.285|0.177|0.566|0.408|0.734|
|SHARP (ours)|0.097|0.358|0.119|0.270|0.071|0.154|0.069|0.190|0.122|0.421|0.258|0.554|

##### Qualitative results.

Figure [2](https://arxiv.org/html/2512.10685v1#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Sharp Monocular View Synthesis in Less Than a Second") shows novel views synthesized by SHARP and a number of baselines. Additional qualitative results, including on images from all evaluation datasets, can be found in the supplement. SHARP consistently produces higher-fidelity renderings from nearby views.

##### Ablation studies.

We conduct extensive ablation studies and controlled experiments on the losses, training curriculum, depth adjustment, and more. The perceptual loss brings substantial improvement in visual quality, while the regularizers address some classes of artifacts. The learned depth adjustment boosts image sharpness and enhances details. The SSFT likewise yields crisper synthesized views. Detailed results and sample images are provided in the supplement.

### 5Conclusion

We presented SHARP, an approach to real-time photorealistic rendering of nearby views from a single photograph. SHARP synthesizes a 3D Gaussian representation via a single forward pass through a neural network in less than a second on a standard GPU. This 3D representation can then be rendered in real time at high resolution from nearby views. Our experiments demonstrate that SHARP delivers state-of-the-art image fidelity for nearby view synthesis, outperforming recent approaches that are in some cases two to three orders of magnitude more computationally intensive.

One clear opportunity for future work is to extend the methodology to support photorealistic synthesis of faraway views without compromising the fidelity of nearby views or the benefits of fast interactive synthesis. This may call for judicious integration of diffusion models (Po et al., [2023](https://arxiv.org/html/2512.10685v1#bib.bib32)), possibly with the aid of distillation for reducing synthesis latency (Yin et al., [2024](https://arxiv.org/html/2512.10685v1#bib.bib64)). With diffusion models, a unified view synthesis routine for single view, multi-view, and video input (Ren et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib37); Ma et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib28)) may emerge as a versatile generalization. Another interesting avenue is a principled treatment of view-dependent and volumetric effects (Verbin et al., [2024](https://arxiv.org/html/2512.10685v1#bib.bib55)).

### References

- Barron et al. (2023)Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman.Zip-NeRF: Anti-aliased grid-based neural radiance fields.In _ICCV_, 2023.
- Bochkovskii et al. (2025)Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, and Vladlen Koltun.Depth Pro: Sharp monocular metric depth in less than a second.In _ICLR_, 2025.
- Chen et al. (2021)Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su.MVSNeRF: Fast generalizable radiance field reconstruction from multi-view stereo.In _ICCV_, 2021.
- Chen et al. (2024)Anpei Chen, Haofei Xu, Stefano Esposito, Siyu Tang, and Andreas Geiger.LaRa: Efficient large-baseline radiance fields.In _ECCV_, 2024.
- Chen (1995)Shenchang Eric Chen.QuickTime VR: An image-based approach to virtual environment navigation.In _SIGGRAPH_, 1995.
- Chen & Williams (1993)Shenchang Eric Chen and Lance Williams.View interpolation for image synthesis.In _SIGGRAPH_, 1993.
- Ding et al. (2022)Keyan Ding, Kede Ma, Shiqi Wang, and Eero P. Simoncelli.Image quality assessment: Unifying structure and texture similarity._PAMI_, 44(5), 2022.
- Dosovitskiy et al. (2021)Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby.An image is worth 16x16 words: Transformers for image recognition at scale.In _ICLR_, 2021.
- Flynn et al. (2019)John Flynn, Michael Broxton, Paul Debevec, Matthew DuVall, Graham Fyffe, Ryan Overbeck, Noah Snavely, and Richard Tucker.DeepView: View synthesis with learned gradient descent.In _CVPR_, 2019.
- Gao et al. (2024)Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul P. Srinivasan, Jonathan T. Barron, and Ben Poole.CAT3D: Create anything in 3D with multi-view diffusion models.In _NeurIPS_, 2024.
- Gatys et al. (2016)Leon A Gatys, Alexander S Ecker, and Matthias Bethge.Image style transfer using convolutional neural networks.In _CVPR_, 2016.
- Godard et al. (2017)Clémeant Godard, Oisin Mac Aodha, and Gabriel J. Brostow.Unsupervised monocular depth estimation with left-right consistency.In _CVPR_, 2017.
- Gu et al. (2023)Jiatao Gu, Alex Trevithick, Kai-En Lin, Joshua M Susskind, Christian Theobalt, Lingjie Liu, and Ravi Ramamoorthi.NerfDiff: Single-image view synthesis with NeRF-guided distillation from 3D-aware diffusion.In _ICML_, 2023.
- Han et al. (2022)Yuxuan Han, Ruicheng Wang, and Jiaolong Yang.Single-view view synthesis in the wild with learned adaptive multiplane images.In _SIGGRAPH_, 2022.
- Hong et al. (2023)Yuan Hong, Kai Zhang, Jianfeng Gu, Sai Bi, Yibing Zhou, Ding Liu, Fangcheng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan.LRM: Large reconstruction model for single image to 3D._arXiv:2311.04400_, 2023.
- Jampani et al. (2021)Varun Jampani, Huiwen Chang, Kyle Sargent, Abhishek Kar, Richard Tucker, Michael Krainin, Dominik Kaeser, William T Freeman, David Salesin, Brian Curless, et al.Slide: Single image 3D photography with soft layering and depth-aware inpainting.In _ICCV_, 2021.
- Jin et al. (2025)Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang Xu.LVSM: A large view synthesis model with minimal 3d inductive bias.In _ICLR_, 2025.
- Johnson et al. (2016)Justin Johnson, Alexandre Alahi, and Li Fei-Fei.Perceptual losses for real-time style transfer and super-resolution.In _ECCV_, 2016.
- Kang et al. (2007)Sing Bing Kang, Yin Li, Xin Tong, and Heung-Yeung Shum.Image-based rendering._Foundations and Trends in Computer Graphics and Vision_, 2(3), 2007.
- Kant et al. (2023)Yash Kant, Aliaksandr Siarohin, Mikhail Vasilkovsky, Riza Alp Guler, Jian Ren, Sergey Tulyakov, and Igor Gilitschenski.iNVS: Repurposing diffusion inpainters for novel view synthesis.In _SIGGRAPH Asia_, 2023.
- Kerbl et al. (2023)Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis.3D Gaussian splatting for real-time radiance field rendering.In _SIGGRAPH_, 2023.
- Khan et al. (2023)Numair Khan, Eric Penner, Douglas Lanman, and Lei Xiao.Tiled multiplane images for practical 3D photography.In _ICCV_, 2023.
- Kingma & Ba (2015)Diederik P. Kingma and Jimmy Ba.Adam: A method for stochastic optimization.In _ICLR_, 2015.
- Knapitsch et al. (2017)Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun.Tanks and temples: Benchmarking large-scale scene reconstruction.In _ICCV_, 2017.
- Liang et al. (2025)Hanwen Liang, Junli Cao, Vidit Goel, Guocheng Qian, Sergei Korolev, Demetri Terzopoulos, Konstantinos N. Plataniotis, Sergey Tulyakov, and Jian Ren.Wonderland: Navigating 3D scenes from a single image.In _CVPR_, 2025.
- Liu et al. (2023)Ruoshi Liu, Raymond Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick.Zero-1-to-3: Zero-shot one image to 3D object.In _ICCV_, 2023.
- Loshchilov & Hutter (2017)Ilya Loshchilov and Frank Hutter.SGDR: stochastic gradient descent with warm restarts.In _ICLR_, 2017.
- Ma et al. (2025)Baorui Ma, Huachen Gao, Haoge Deng, Zhengxiong Luo, Tiejun Huang, Lulu Tang, and Xinlong Wang.You see it, you got it: Learning 3d creation on pose-free videos at scale.In _CVPR_, 2025.
- Mildenhall et al. (2020)Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng.NeRF: Representing scenes as neural radiance fields for view synthesis.In _ECCV_, 2020.
- OpenScene (2023)OpenScene.Openscene: The largest up-to-date 3D occupancy prediction benchmark in autonomous driving.[https://github.com/OpenDriveLab/OpenScene](https://github.com/OpenDriveLab/OpenScene), 2023.
- Piccinelli et al. (2024)Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Ghoul, and Fisher Yu.UniDepth: Universal monocular metric depth estimation.In _CVPR_, 2024.
- Po et al. (2023)Ryan Po, Wang Yifan, Vladislav Golyanik, Kfir Aberman, Jonathan T. Barron, Amit H. Bermano, Eric Ryan Chan, Tali Dekel, Aleksander Holynski, Angjoo Kanazawa, C. Karen Liu, Lingjie Liu, Ben Mildenhall, Matthias NieSSner, Björn Ommer, Christian Theobalt, Peter Wonka, and Gordon Wetzstein.State of the art on diffusion models for visual computing._arXiv:2310.07204_, 2023.
- Poggi et al. (2020)Matteo Poggi, Filippo Aleotti, Fabio Tosi, and Stefano Mattoccia.On the uncertainty of self-supervised monocular depth estimation.In _CVPR_, 2020.
- Ramirez et al. (2024)Pierluigi Zama Ramirez, Alex Costanzino, Fabio Tosi, Matteo Poggi, Samuele Salti, Stefano Mattoccia, and Luigi Di Stefano.Booster: A benchmark for depth from images of specular and transparent surfaces._PAMI_, 46(1), 2024.
- Ranftl et al. (2021)René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun.Vision transformers for dense prediction.In _ICCV_, 2021.
- Reda et al. (2022)Fitsum Reda, Janne Kontkanen, Eric Tabellion, Deqing Sun, Caroline Pantofaru, and Brian Curless.FILM: Frame interpolation for large motion.In _ECCV_, 2022.
- Ren et al. (2025)Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, and Jun Gao.GEN3C: 3D-informed world-consistent video generation with precise camera control.In _CVPR_, 2025.
- Riegler & Koltun (2020)Gernot Riegler and Vladlen Koltun.Free view synthesis.In _ECCV_, 2020.
- Riegler & Koltun (2021)Gernot Riegler and Vladlen Koltun.Stable view synthesis.In _CVPR_, 2021.
- Ronneberger et al. (2015)Olaf Ronneberger, Philipp Fischer, and Thomas Brox.U-Net: Convolutional networks for biomedical image segmentation.In _MICCAI_, 2015.
- Sargent et al. (2024)Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann, Hong-Xing Yu, Yunzhi Zhang, Eric R. Chan, Dmitry Lagun, Li Fei-Fei, Deqing Sun, and Jiajun Wu.ZeroNVS: Zero-shot 360-degree view synthesis from a single real image.In _CVPR_, 2024.
- Scharstein et al. (2014)Daniel Scharstein, Heiko Hirschmüller, York Kitajima, Greg Krathwohl, Nera Nešić, Xi Wang, and Porter Westling.High-resolution stereo datasets with subpixel-accurate ground truth.In _GCPR_, 2014.
- Schöps et al. (2017)Thomas Schöps, Johannes L Schönberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger.A multi-view stereo benchmark with high-resolution images and multi-camera videos.In _CVPR_, 2017.
- Schwarz et al. (2025)Katja Schwarz, Denis Rozumny, Samuel Rota Bulo, Lorenzo Porzi, and Peter Kontschieder.A recipe for generating 3D worlds from a single image._arXiv:2503.16611_, 2025.
- Shade et al. (1998)Jonathan Shade, Steven Gortler, Li-wei He, and Richard Szeliski.Layered depth images.In _SIGGRAPH_, 1998.
- Sohn et al. (2015)Kihyuk Sohn, Honglak Lee, and Xinchen Yan.Learning structured output representation using deep conditional generative models.In _NeurIPS_, 2015.
- Suvorov et al. (2021)Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky.Resolution-robust large mask inpainting with fourier convolutions._arXiv preprint arXiv:2109.07161_, 2021.
- Szymanowicz et al. (2024)Stanislaw Szymanowicz, Christian Rupprecht, and Andrea Vedaldi.Splatter image: Ultra-fast single-view 3d reconstruction.In _CVPR_, 2024.
- Szymanowicz et al. (2025a)Stanislaw Szymanowicz, Eldar Insafutdinov, Chuanxia Zheng, Dylan Campbell, Joao F. Henriques, Christian Rupprecht, and Andrea Vedaldi.Flash3D: Feed-forward generalisable 3D scene reconstruction from a single image.In _3DV_, 2025a.
- Szymanowicz et al. (2025b)Stanislaw Szymanowicz, Jason Y. Zhang, Pratul Srinivasan, Ruiqi Gao, Arthur Brussee, Aleksander Holynski, Ricardo Martin-Brualla, Jonathan T. Barron, and Philipp Henzler.Bolt3D: Generating 3D scenes in seconds._arXiv:2503.14445_, 2025b.
- Tewari et al. (2022)A. Tewari, J. Thies, B. Mildenhall, P. Srinivasan, E. Tretschk, W. Yifan, C. Lassner, V. Sitzmann, R. Martin-Brualla, S. Lombardi, T. Simon, C. Theobalt, M. NieSSner, J. T. Barron, G. Wetzstein, M. Zollhöfer, and V. Golyanik.Advances in neural rendering._Computer Graphics Forum_, 41(2), 2022.
- Tongue et al. (2024)Joseph Tongue, Gene Chou, Ruojin Cai, Guandao Yang, Kai Zhang, Gordon Wetzstein, Bharath Hariharan, and Noah Snavely.MegaScenes: Scene-level view synthesis at scale.In _ECCV_, 2024.
- Tucker & Snavely (2020)Richard Tucker and Noah Snavely.Single-view view synthesis with multiplane images.In _CVPR_, 2020.
- Unsplash (2022)Unsplash.Unsplash image collection.[https://unsplash.com](https://unsplash.com/), 2022.
- Verbin et al. (2024)Dor Verbin, Pratul P. Srinivasan, Peter Hedman, Ben Mildenhall, Benjamin Attal, Richard Szeliski, and Jonathan T. Barron.NeRF-Casting: Improved view-dependent appearance with consistent reflections.In _SIGGRAPH Asia_, 2024.
- Wang et al. (2021)Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P. Srinivasan, Howard Zhou, Jonathan T. Barron, Ricardo Martin-Brualla, Noah Snavely, and Thomas Funkhouser.IBRNet: Learning multi-view image-based rendering.In _CVPR_, 2021.
- Wang et al. (2024a)Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang.MoGe: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision._arXiv:2410.19115_, 2024a.
- Wang et al. (2024b)Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud.DUSt3R: Geometric 3D vision made easy.In _CVPR_, 2024b.
- Watson et al. (2022)Daniel Watson, William Chan, Ricardo Martin-Brualla, Jonathan Ho, Andrea Tagliasacchi, and Mohammad Norouzi.Novel view synthesis with diffusion models._arXiv:2210.04628_, 2022.
- Wen et al. (2025)Hongyu Wen, Yiming Zuo, Venkat Subramanian, Patrick Chen, and Jia Deng.Seeing and seeing through the glass: Real and synthetic data for multi-layer depth estimation._arXiv:2503.11633_, 2025.
- Wiles et al. (2020)Olivia Wiles, Georgia Gkioxari, Richard Szeliski, and Justin Johnson.SynSin: End-to-end view synthesis from a single image.In _CVPR_, 2020.
- Xia et al. (2024)Hongchi Xia, Yang Fu, Sifei Liu, and Xiaolong Wang.RGBD objects in the wild: Scaling real-world 3D object learning from RGB-D videos._arXiv:2401.12592_, 2024.
- Yeshwanth et al. (2023)Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai.ScanNet++: A high-fidelity dataset of 3D indoor scenes.In _ICCV_, 2023.
- Yin et al. (2024)Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman, and Taesung Park.One-step diffusion with distribution matching distillation.In _CVPR_, 2024.
- Yu et al. (2021)Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa.pixelNeRF: Neural radiance fields from one or few images.In _CVPR_, 2021.
- Yu et al. (2025)Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T. Freeman, and Jiajun Wu.Wonderworld: Interactive 3D scene generation from a single image.In _CVPR_, 2025.
- Yu et al. (2024)Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian.ViewCrafter: Taming video diffusion models for high-fidelity novel view synthesis._arXiv:2409.02048_, 2024.
- Zhang et al. (2024)Kai Zhang, Ruojin Cai, Chao Xu, Zexiang Xu, Hao Su, and Yuan Hong.GS-LRM: Large reconstruction model for 3D Gaussian splatting.In _ECCV_, 2024.
- Zhang et al. (2018)Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang.The unreasonable effectiveness of deep features as a perceptual metric.In _CVPR_, 2018.
- Zhang et al. (2025)Xiang Zhang, Yang Zhang, Lukas Mehl, Markus Gross, and Christopher Schroers.High-fidelity novel view synthesis via splatting-guided diffusion.In _SIGGRAPH_, 2025.
- Zhou et al. (2025)Jensen (Jinghao) Zhou, Hang Gao, Vikram Voleti, Aaryaman Vasishta, Chun-Han Yao, Mark Boss, Philip Torr, Christian Rupprecht, and Varun Jampani.Stable virtual camera: Generative view synthesis with diffusion models._arXiv:2503.14489_, 2025.
- Zhou et al. (2016)Tinghui Zhou, Shubham Tulsiani, Weilun Sun, Jitendra Malik, and Alexei A. Efros.View synthesis by appearance flow.In _ECCV_, 2016.
- Zhou et al. (2018)Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely.Stereo magnification: Learning view synthesis using multiplane images.In _SIGGRAPH_, 2018.

## Supplementary Material

### Appendix AImplementation Details

#### A.1Attribute Specific Activation

Activation functions γ and their correspondent scale factors η in Eq. [3.2](https://arxiv.org/html/2512.10685v1#S3.E2 "Equation 3.2 ‣ Gaussian composer. ‣ 3.1 Overview ‣ 3 Method ‣ Sharp Monocular View Synthesis in Less Than a Second") are specified below:

|   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|
||position (x/z,y/z)|position (z−1)|color|rotation|scale|alpha|
|γ|identity|softplus|sigmoid|identity|sigmoid|sigmoid|
|η|10−3|10−3|10−1|1|1|1|

For the position, we apply the activation function in NDC space, i.e. we first map [x,y,z]→[x/z,y/z,1/z] before applying the activation function and adding the delta. After the operation, we transform the result back to world coordinates.

#### A.2Training Objectives

In the loss configuration, we choose λcolor=1.0,λalpha=1.0,λpercep=3.0, λdepth=0.2, λtv=1.0, λgrad=0.5, λdelta=1.0, λsplat=1.0, λscale=0.1, λ∇scale=5.0 in Eq. [3.12](https://arxiv.org/html/2512.10685v1#S3.E12 "Equation 3.12 ‣ 3.4 Training Objectives ‣ 3 Method ‣ Sharp Monocular View Synthesis in Less Than a Second").

For the perceptual loss in Eq. [3.4](https://arxiv.org/html/2512.10685v1#S3.E4 "Equation 3.4 ‣ 3.4 Training Objectives ‣ 3 Method ‣ Sharp Monocular View Synthesis in Less Than a Second") we use λlfeat=1Dl⋅Hl⋅Wl and λlGram=10Dl2, where Dl×Hl×Wl denotes the shape of the l-th feature map ϕl​(⋅)∈ℝDl×Hl×Wl.

We trained the network using the Adam optimizer (Kingma & Ba, [2015](https://arxiv.org/html/2512.10685v1#bib.bib23)) with a cosine learning rate schedule (Loshchilov & Hutter, [2017](https://arxiv.org/html/2512.10685v1#bib.bib27)). The learning rate was linearly warmed up for 10,000 iterations to an initial value of 1.6×10−4, after which it decayed to a final value of 1.6×10−5.

#### A.3View Frustum Masking

We implement a view frustum masking technique to address ambiguity in view synthesis – regions occluded in the original view have multiple plausible reconstructions. By using depth information to determine which regions in the new view correspond to points visible in the original view, we apply supervision only where ground truth is reliable.

To calculate this mask, we project points from the target view back to the source view:

|   |   |   |   |
|---|---|---|---|
||[x′⋅z′,y′⋅z′,z′,1]T→𝐓novel→source[x⋅z,y⋅z,z,1]T.||(A.1)|

The mask is then defined as

|   |   |   |   |
|---|---|---|---|
||M​(x′,y′)={1,if −1.05≤x≤1.05​ and −1.05≤y≤1.050,otherwise||(A.2)|

This mask is applied to all image-based losses on the target view.

#### A.4Perceptual Loss

Here we detail the challenges and our solutions in incorporating the perceptual loss. We employ the perceptual loss aimed at improving inpainting (Suvorov et al., [2021](https://arxiv.org/html/2512.10685v1#bib.bib47)). Similar to its application in the image domain, we initially applied the loss only in the occluded image patches of the novel view; however, through experiments, we observed that applying the loss to the entire rendered image resulted in more plausible details and fewer artifacts in general, even in the non-occluded foreground regions, as seen in Figure [9](https://arxiv.org/html/2512.10685v1#A4.F9 "Figure 9 ‣ D.8 Additional Qualitative Results ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second").

However, this formulation of loss imposes two major challenges: (a) heavy memory overhead, and (b) compromised sharpness.

Memory. The perceptual loss maximizes feature similarity between the rendering and the ground truth. It is constructed from a combination of MSE losses on layer-wise feature maps of deep neural networks (in our case, a ResNet-50). Since the loss itself is computed through a deep neural network, when applied to full images, it adds a significant memory overhead to the already large computation graph during backpropagation. Furthermore, when the loss is applied to both the reconstruction and synthesized views, the accumulated computation graph can lead to out-of-memory conditions even on an A100 with a generous memory pool (40GB) with a batch size of one.

To address the problem, one potential workaround would be simply reducing the activation precision to BF16; however, this does not address the fundamental problem of computation graph accumulation, prevents scaling the loss to more novel view supervisions, and causes training instability, especially when predicted 3DGS contains properties (_e.g._, singular values) that are prone to precision changes. Gradient checkpointing is another option, but it can drastically impair training efficiency.

To address this problem, we propose a novel computation graph surgery mechanism. We implement a _surgery operator_ to accept and cache gradients along with the inputs during the forward pass, and to inject cached gradients during the backward pass. Then, at the perceptual loss node in the graph, we _eagerly pre-compute_ the gradients with respect to the features via an explicit _autograd_ call, release the partial computation graph involving ResNet, and override the node with the surgery-operated one. This strategy avoids accumulating the computation graph and leads to a compact graph that is agnostic to the number of pixels or views. As a result, we are able to continue training at the full FP32 precision with perceptual loss on both reconstruction and novel views, without compromising training throughput. It is worth noting that the surgery operator is a general operator and can be integrated into any training framework with similar memory concerns regarding the computation graph.

Sharpness. Since the perceptual loss is applied to the latent feature space, while it offers the benefit of more plausible inpainting, the output renderings often tend to be blurry in the pixel space. Through backpropagation, this translates to large and blobby 3D Gaussians, whose renderings are simultaneously less detailed and more time-consuming.

To encourage sharpness, we explored losses that reduce feature space distance and revived the Gram matrix loss (Reda et al., [2022](https://arxiv.org/html/2512.10685v1#bib.bib36)) that was originally designed for style transfer. This loss matches the auto-correlation of the latent features, further enhancing feature space similarity and boosting image sharpness. We introduce this loss in Eq. [3.4](https://arxiv.org/html/2512.10685v1#S3.E4 "Equation 3.4 ‣ 3.4 Training Objectives ‣ 3 Method ‣ Sharp Monocular View Synthesis in Less Than a Second"). As mentioned above, the original Gram matrix loss was applied to VGG features targeted at style transfer, and cannot be directly transferred to the ResNet-50 features pre-trained for inpainting. We conducted a series of controlled experiments with λlGram=jDl2, j∈{1,10,100,500}, along with λlfeat=kDl⋅Hl⋅Wl, k∈{0.1,0.3,1.0,10.0}, and identified the most promising combination, as reported in Section [A.2](https://arxiv.org/html/2512.10685v1#A1.SS2 "A.2 Training Objectives ‣ Appendix A Implementation Details ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second"), through extensive quantitative metric validation and qualitative human inspection. This carefully tuned perceptual loss improves the DISTS metrics by 62% and 47% on benchmarks (as seen in Table [8](https://arxiv.org/html/2512.10685v1#A4.T8 "Table 8 ‣ Losses. ‣ D.5 Ablation Studies ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second")), and reduces rendering latency by 49% and 36% respectively (Table [9](https://arxiv.org/html/2512.10685v1#A4.T9 "Table 9 ‣ Losses. ‣ D.5 Ablation Studies ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second")).

### Appendix BTraining Data

#### B.1Synthetic data

In Stage 1 of training (Section [3.3](https://arxiv.org/html/2512.10685v1#S3.SS3 "3.3 Training Strategy ‣ 3 Method ‣ Sharp Monocular View Synthesis in Less Than a Second")), we use a large-scale synthetic dataset generated using an in-house procedural content generation system. This system operates by sampling from a large collection of artist-made environments, comprising over 2K outdoor and 5K indoor scenes, and augmenting them procedurally. For each sampled environment, the framework populates the scene with high-quality digital human characters featuring realistic hair grooms and garments, along with a variety of additional objects. This approach enhances the structural and visual diversity of the dataset while preserving the underlying artistic quality of the base environments.

To further enhance scene diversity and complexity, the framework supports random placement of various object types, including thin structures, transparent materials, and reflective surfaces, across a wide range of spatial configurations. It also offers fine-grained control over camera parameters such as position, orientation, and focal length, as well as detailed illumination settings. Lighting setups include both physically-based direct light sources, with variations in direction, intensity, and color temperature. We also use high-dynamic-range (HDR) environment maps, which are sampled from a curated collection off high-resolution HDRIs. This combination enables realistic global illumination effects under diverse and physically plausible lighting conditions.

For each scene, we identify one object of interest and position a ring containing 10 virtual cameras around it at varying distances and angles. The cameras are arranged in concentric circles such that the cameras no more than 60cm apart, simulating a multi-view capture setup. This allows the dataset to capture the same object or scene element from diverse perspectives. All images are rendered using the V-Ray physically based rendering engine, ensuring photorealistic lighting and material interactions. The final dataset consists of approximately 700K unique rendered scene instances, each with 11 rendered views, totaling around 8M images at 1536×1536 or 2048×2048 resolutions.

#### B.2Real-world data

In Stage 2 of training (Section [3.3](https://arxiv.org/html/2512.10685v1#S3.SS3 "3.3 Training Strategy ‣ 3 Method ‣ Sharp Monocular View Synthesis in Less Than a Second")), we use OpenScene ([2023](https://arxiv.org/html/2512.10685v1#bib.bib30)) as well as a collection of high-quality photographs from Shutterstock, Getty Images, and Flickr, all with commercial licenses. The dataset contains 2.65M images in total.

### Appendix CExplanatory Figures

#### C.1Image Fidelity Metrics

To determine which metrics are most suitable for evaluating view synthesis quality, we conducted an experiment analyzing how different metrics respond to simple image translations.

As shown in Table [2](https://arxiv.org/html/2512.10685v1#A3.T2 "Table 2 ‣ C.1 Image Fidelity Metrics ‣ Appendix C Explanatory Figures ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second") and Figure [4](https://arxiv.org/html/2512.10685v1#A3.F4 "Figure 4 ‣ C.1 Image Fidelity Metrics ‣ Appendix C Explanatory Figures ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second"), we observe that older pointwise metrics such as PSNR and SSIM are highly sensitive to small spatial misalignments. A mere 1% translation causes PSNR to drop to 11.2 and SSIM to 0.375 – values that are surprisingly close to those obtained when comparing with a mean image (PSNR 10.7, SSIM 0.351).

![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/metrics_comparison/reference_crop.jpeg)

(a)Reference image

![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/metrics_comparison/translated_crop_10.jpeg)

(b)Translated by 1%

![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/metrics_comparison/mean_image.jpg)

(c)Mean image

Figure 4:Effect of small translations on metrics. A 1% translation (b) of the reference image (a) appears nearly identical to human observers, yet dramatically affects PSNR and SSIM values. The mean image (c), despite being unrecognizable compared to the reference, produces PSNR and SSIM values remarkably similar to those of the 1% translation.

Table 2:Sensitivity of metrics to small image shifts. Different metrics exhibit varying sensitivity to small spatial misalignments.

|   |   |   |   |   |
|---|---|---|---|---|
||DISTS (↓)|LPIPS (↓)|PSNR (↑)|SSIM (↑)|
|Comparison|||||
|Translated (0.1%)|0.008|0.059|21.3|0.623|
|Translated (1.0%)|0.079|0.491|11.2|0.375|
|Translated (5.0%)|0.121|0.723|8.1|0.249|
|Mean Image|0.859|0.970|10.7|0.351|

Perceptual metrics, such as DISTS and LPIPS, demonstrate better robustness to these small translations. DISTS shows exceptional stability with a value of 0.079 for a 1% translation compared to 0.859 for the mean image. This characteristic is especially relevant for evaluating view synthesis, where geometric inaccuracies can manifest as small shifts between synthesized and ground truth views. Since novel view synthesis must address both geometric and appearance errors, metrics that can accommodate minor geometric misalignments while still reflecting perceptual quality provide evaluations that correspond more closely to human perception. Based on these findings, we adopted DISTS and LPIPS as our primary evaluation metrics.

#### C.2Depth Estimation Uncertainty

Monocular depth estimation is fundamentally ill-posed, as multiple 3D configurations can produce the same 2D image (Poggi et al., [2020](https://arxiv.org/html/2512.10685v1#bib.bib33)). Figure [5](https://arxiv.org/html/2512.10685v1#A3.F5 "Figure 5 ‣ C.2 Depth Estimation Uncertainty ‣ Appendix C Explanatory Figures ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second") illustrates this ambiguity by comparing depth predictions for an image and its mirror image – a technique similar to that used in left-right consistency for monocular depth training (Godard et al., [2017](https://arxiv.org/html/2512.10685v1#bib.bib12)). The uncertainty map reveals that depth estimators struggle most at object boundaries and in regions with complex geometric structures, such as foliage. When these ambiguous depth estimates are used directly for view synthesis, the resulting images can exhibit visual artifacts as the network attempts to average across multiple plausible depth configurations. Our depth adjustment module, inspired by Conditional Variational Autoencoders (Sohn et al., [2015](https://arxiv.org/html/2512.10685v1#bib.bib46)), addresses this issue by learning a scale map that refines the predicted depth during training, addressing these ambiguities in a way that optimizes for view synthesis quality rather than depth accuracy alone.

![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/depth_uncertainty/original.jpg)

(a)Original image

![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/depth_uncertainty/depth.jpg)

(b)Predicted depth

![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/depth_uncertainty/flipped_depth.jpg)

(c)Flipped prediction

![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/depth_uncertainty/uncertainty.jpg)

(d)Uncertainty map

Figure 5:Ambiguity in depth estimation. We demonstrate the inherent ambiguity in monocular depth estimation by (a) taking an original image, (b) predicting its depth using Depth Pro, (c) horizontally flipping the image, applying Depth Pro, and flipping the result back, and (d) computing the relative absolute error between the two predictions to generate an uncertainty map. Higher values (brighter regions) indicate greater inconsistency between predictions.

### Appendix DExperiments

#### D.1Evaluation Dataset Setup

For stereo datasets (Middlebury, Booster), we apply SHARP and the baselines to the left frame and predict the right frame.

For multi-view datasets (ScanNet++, WildRGBD, Tanks and Temples, ETH3D), we proceed as follows:

- • 
    
    For each sequence/scene, we split them into 10-view sets.
    
- • 
    
    Within each 10-view set, we compute pairwise depth overlap and select pairs with overlap >60%. For datasets with sparse depth (e.g., ETH3D), we predict monodepth via Depth Pro (Bochkovskii et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib2)), apply a global scale alignment from dense monodepth to sparse depth (Eq. [D.2](https://arxiv.org/html/2512.10685v1#A4.E2 "Equation D.2 ‣ D.4 Evaluation with Privileged Depth Information ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second")), and compute the monodepth overlap.
    
- • 
    
    We select min⁡(512,#​p​a​i​r​s) pairs to evaluate per dataset. For each pair, we predict target image from the source image.
    

The reason for limiting the number of pairs is the slow inference speed of diffusion-based baselines. For instance, Gen3C takes 15 minutes to synthesize a new view (as a byproduct of synthesizing a video). 512 pairs already take roughly 5 days to evaluate on an A100, any larger set becomes less tractable to evaluate.

Figure [6](https://arxiv.org/html/2512.10685v1#A4.F6 "Figure 6 ‣ D.1 Evaluation Dataset Setup ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second") shows the distribution of pairwise camera baseline size across datasets.

![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/baseline_histograms.jpg)

Figure 6:The pairwise camera baseline size distribution across datasets.

ScanNet++. We sample from the _nvs test_ split with DLSR images.

WildRGBD. We sample from the validation split from _nvs list_ in each scene.

Tanks and Temples. We take the training set for Tanks And Temples. We composite SfM poses and SfM-to-LiDAR transformation, both provided by the authors, to create camera matrices in the metric space, then backproject the LiDAR points to the associated images to form ground truth depth maps. The depth maps were only used for experiments with privileged depth information.

ETH3D. We use the ETH3D high-resolution multi-view training set. As with Tanks and Temples, we only use sparse depth for experiments with privileged depth information.

#### D.2Baselines

We report model sizes of SHARP and baselines in Table [3](https://arxiv.org/html/2512.10685v1#A4.T3 "Table 3 ‣ D.2 Baselines ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second"). The numbers are based on reported numbers in the publication and source code. The original TMPI paper utilizes DPT depth (Ranftl et al., [2021](https://arxiv.org/html/2512.10685v1#bib.bib35)) as the monodepth backbone; we replace it with the latest Depth Pro (Bochkovskii et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib2)) for better quality and hence report the total parameters with Depth Pro backbone.

Table 3:Parameter counts across models. Trainable parameters are estimated by subtracting frozen module parameter counts from total counts. ∗: finetuning diffusion models.

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
||Flash3D|TMPI|LVSM|SVC|ViewCrafter|Gen3C|SHARP (ours)|
|# total|399M|957M|314M|2.33B|3.17B|7.7B|702M|
|# trainable|52M|6M|314M|1.26B∗|2.6B∗|7.4B∗|340M|

To verify that the in-house synthetic data (Section [B](https://arxiv.org/html/2512.10685v1#A2 "Appendix B Training Data ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second")) is not the dominant factor in the view synthesis fidelity demonstrated by SHARP, we retrain Flash3D on the same in-house synthetic data.

We trained on 24K (3%) and 216K (28%) scenes from our data for 100K steps and 150K steps, respectively. We do not further scale up the number of scenes because we do not find a consistent positive signal of scaling data with Flash3D, and more scenes trigger data loader crashes in the reference implementation2. As shown in Table [1](https://arxiv.org/html/2512.10685v1#S4.T1 "Table 1 ‣ Quantitative evaluation. ‣ 4 Experiments ‣ Sharp Monocular View Synthesis in Less Than a Second"), we do not observe a distinct improvement when training Flash3D with our synthetic data. This implies that our in-house data quality is not the principal factor in the reported view synthesis performance.

Table 4:Training Flash3D on in-house synthetic data.

|   |   |   |   |   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|data|Middlebury|   |Booster|   |ScanNet++|   |WildRGBD|   |Tanks and Temples|   |ETH3D|   |
|DISTS↓|LPIPS↓|DISTS↓|LPIPS↓|DISTS↓|LPIPS↓|DISTS↓|LPIPS↓|DISTS↓|LPIPS↓|DISTS↓|LPIPS↓|
|internal (3%)|0.325|0.599|0.335|0.403|0.398|0.630|0.235|0.417|0.453|0.756|0.506|0.673|
|internal (28%)|0.433|0.647|0.442|0.415|0.488|0.696|0.255|0.448|0.553|0.815|0.570|0.686|
|public (RE10K)|0.359|0.581|0.409|0.370|0.374|0.572|0.159|0.345|0.382|0.683|0.535|0.651|

#### D.3Additional Quantitative Experiments

##### PSNR and SSIM.

For completeness, we report PSNR and SSIM in Table [5](https://arxiv.org/html/2512.10685v1#A4.T5 "Table 5 ‣ PSNR and SSIM. ‣ D.3 Additional Quantitative Experiments ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second"), but we discourage their use for evaluating view synthesis fidelity, as per the analysis in Section [C.1](https://arxiv.org/html/2512.10685v1#A3.SS1 "C.1 Image Fidelity Metrics ‣ Appendix C Explanatory Figures ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second").

Table 5:We report PSNR/SSIM metrics for completeness. See Section [C.1](https://arxiv.org/html/2512.10685v1#A3.SS1 "C.1 Image Fidelity Metrics ‣ Appendix C Explanatory Figures ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second") for analysis of the metrics.

|   |   |   |   |   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||Middlebury|   |Booster|   |ScanNet++|   |WildRGBD|   |Tanks and Temples|   |ETH3D|   |
||PSNR↑|SSIM↑|PSNR↑|SSIM↑|PSNR↑|SSIM↑|PSNR↑|SSIM↑|PSNR↑|SSIM↑|PSNR↑|SSIM↑|
|Flash3D|15.88|0.683|22.40|0.873|18.14|0.641|18.09|0.616|15.80|0.518|15.21|0.682|
|TMPI|16.42|0.688|19.44|0.833|16.16|0.712|16.44|0.559|12.41|0.368|12.61|0.540|
|LVSM|15.53|0.681|20.16|0.843|20.25|0.775|18.04|0.594|15.95|0.519|16.72|0.722|
|SVC|12.72|0.613|17.65|0.781|11.71|0.624|12.20|0.410|11.76|0.413|13.36|0.662|
|ViewCrafter|10.33|0.569|14.18|0.692|13.30|0.645|14.43|0.437|11.49|0.423|11.94|0.621|
|Gen3C|13.89|0.624|20.19|0.837|20.82|0.792|16.54|0.504|14.83|0.499|13.09|0.642|
|SHARP (ours)|17.12|0.693|22.19|0.864|22.63|0.833|19.57|0.655|16.33|0.528|14.51|0.610|

##### Runtime.

Runtimes are reported in Table [6](https://arxiv.org/html/2512.10685v1#A4.T6 "Table 6 ‣ Runtime. ‣ D.3 Additional Quantitative Experiments ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second"). SHARP synthesizes the 3D representation in less than a second on an A100 GPU. The representation can then be rendered in real time (100 FPS or higher on most datasets). We always render the results to the native resolution of the datasets, which explains the variability between datasets (e.g. ETH3D has native resolution 6048×4032).

Table 6:Runtime (in seconds) on an A100 GPU. Note that the SVC/TMPI runtime is lower on ETH3D, since they encountered memory issues and we had to rerun them on an H100.

|   |   |   |   |   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||Middlebury|   |Booster|   |ScanNet++|   |WildRGBD|   |Tanks and Temples|   |ETH3D|   |
||Inference↓|Render↓|Inference↓|Render↓|Inference↓|Render↓|Inference↓|Render↓|Inference↓|Render↓|Inference↓|Render↓|
|Flash3D|0.154|0.025|0.154|0.047|0.155|0.004|0.154|0.003|0.154|0.004|0.153|0.041|
|TMPI|0.328|0.249|0.333|0.248|0.315|0.247|0.183|0.294|0.272|0.218|0.222|0.157|
|LVSM|0.121|-|0.120|-|0.120|-|0.120|-|0.120|-|0.121|-|
|SVC|62.687|-|57.598|-|62.670|-|57.456|-|78.846|-|32.610|-|
|ViewCrafter|119.718|-|118.679|-|119.385|-|119.590|-|119.859|-|119.922|-|
|Gen3C|830.225|-|831.775|-|836.455|-|838.695|-|841.418|-|838.143|-|
|SHARP (ours)|0.912|0.010|0.911|0.016|0.911|0.006|0.912|0.004|0.912|0.005|0.910|0.022|

#### D.4Evaluation with Privileged Depth Information

Table [7](https://arxiv.org/html/2512.10685v1#A4.T7 "Table 7 ‣ D.4 Evaluation with Privileged Depth Information ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second") evaluates all view synthesis methods when privileged ground-truth depth maps are used for scale adjustment. We again report PSNR/SSIM metrics for completeness but discourage their use for view synthesis fidelity.

For approaches where a depth proxy is available (Flash3D uses UniDepth (Piccinelli et al., [2024](https://arxiv.org/html/2512.10685v1#bib.bib31)), ViewCrafter uses Dust3r (Wang et al., [2024b](https://arxiv.org/html/2512.10685v1#bib.bib58)), TMPI uses DepthPro (Bochkovskii et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib2)), Gen3C uses MoGe (Wang et al., [2024a](https://arxiv.org/html/2512.10685v1#bib.bib57)), and SHARP uses DepthPro (Bochkovskii et al., [2025](https://arxiv.org/html/2512.10685v1#bib.bib2))), we align the intermediate depth representation 𝐃^ to the ground truth 𝐃 to derive an approximate global scale factor:

|   |   |   |   |   |
|---|---|---|---|---|
||s|=medianp∼Ω​{𝐃​(p)𝐃^​(p)},||(D.1)|
||𝐃¯​(p)|=s⋅𝐃^​(p).||(D.2)|

For other approaches (LVSM and SVC), for each pair, we apply a linear scale sweep to find the best scale that minimizes the DISTS score.

Table 7:View synthesis fidelity with privileged depth information.

|   |   |   |   |   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||Middlebury|   |Booster|   |ScanNet++|   |WildRGBD|   |Tanks and Temples|   |ETH3D|   |
||DISTS↓|LPIPS↓|DISTS↓|LPIPS↓|DISTS↓|LPIPS↓|DISTS↓|LPIPS↓|DISTS↓|LPIPS↓|DISTS↓|LPIPS↓|
|Flash3D|0.333|0.510|0.412|0.361|0.283|0.395|0.181|0.368|0.399|0.666|0.474|0.595|
|TMPI|0.155|0.426|0.232|0.404|0.128|0.310|0.108|0.279|0.356|0.736|0.345|0.697|
|LVSM|0.243|0.564|0.294|0.428|0.125|0.236|0.088|0.229|0.219|0.558|0.456|0.668|
|SVC|0.181|0.518|0.257|0.381|0.146|0.459|0.120|0.407|0.199|0.653|0.410|0.700|
|ViewCrafter|0.163|0.410|0.223|0.310|0.111|0.232|0.102|0.159|0.184|0.476|0.339|0.594|
|Gen3C|0.124|0.347|0.192|0.291|0.085|0.196|0.078|0.118|0.149|0.434|0.283|0.568|
|SHARP (ours)|0.081|0.262|0.110|0.214|0.068|0.137|0.057|0.117|0.112|0.374|0.187|0.381|

|   |   |   |   |   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||Middlebury|   |Booster|   |ScanNet++|   |WildRGBD|   |Tanks and Temples|   |ETH3D|   |
||PSNR↑|SSIM↑|PSNR↑|SSIM↑|PSNR↑|SSIM↑|PSNR↑|SSIM↑|PSNR↑|SSIM↑|PSNR↑|SSIM↑|
|Flash3D|18.61|0.719|23.16|0.879|21.98|0.803|19.45|0.679|16.40|0.567|17.06|0.674|
|TMPI|16.70|0.696|19.69|0.838|16.11|0.709|17.54|0.600|11.85|0.329|13.39|0.578|
|LVSM|15.22|0.672|19.27|0.823|23.42|0.826|19.29|0.627|16.31|0.529|16.38|0.719|
|SVC|15.73|0.671|20.27|0.841|15.19|0.696|14.77|0.483|13.47|0.465|13.83|0.671|
|ViewCrafter|17.11|0.703|21.55|0.860|19.73|0.788|20.10|0.672|16.84|0.566|18.81|0.721|
|Gen3C|18.46|0.720|23.12|0.875|22.11|0.822|22.45|0.745|17.23|0.557|18.93|0.716|
|SHARP (ours)|19.18|0.742|23.57|0.880|23.67|0.865|23.62|0.780|16.92|0.543|19.09|0.715|

#### D.5Ablation Studies

We summarize the results from extensive ablation studies in Tables [8](https://arxiv.org/html/2512.10685v1#A4.T8 "Table 8 ‣ Losses. ‣ D.5 Ablation Studies ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second")–[13](https://arxiv.org/html/2512.10685v1#A4.T13 "Table 13 ‣ Unfreezing Backbone. ‣ D.5 Ablation Studies ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second") and Figures [9](https://arxiv.org/html/2512.10685v1#A4.F9 "Figure 9 ‣ D.8 Additional Qualitative Results ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second")–[12](https://arxiv.org/html/2512.10685v1#A4.F12 "Figure 12 ‣ D.8 Additional Qualitative Results ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second").

Datasets. We report metrics on ScanNet++ (small scale scenes) and Tanks and Temples (large scale scenes), and display results on the real-world dataset Unsplash (Unsplash, [2022](https://arxiv.org/html/2512.10685v1#bib.bib54)).

Models. For losses, depth adjustment, and unfreezing experiments, we train multiple variants of our model for 60K steps on 32 A100 GPUs only on Stage 1, without Stage 2 SSFT. For the SSFT experiment, we compare Stage 1 and Stage 2 models discussed in Section [4](https://arxiv.org/html/2512.10685v1#S4 "4 Experiments ‣ Sharp Monocular View Synthesis in Less Than a Second") of the main paper.

##### Losses.

We always incorporate color and alpha losses for appearance reconstruction. Our ablation of loss terms (Table [8](https://arxiv.org/html/2512.10685v1#A4.T8 "Table 8 ‣ Losses. ‣ D.5 Ablation Studies ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second") and Figure [9](https://arxiv.org/html/2512.10685v1#A4.F9 "Figure 9 ‣ D.8 Additional Qualitative Results ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second")) show that the depth loss reduces geometry distortion, and perceptual loss brings significant improvement in inpainting quality and image sharpness; both losses result in improved metrics. While our regularizers do not move the metrics on the datasets used for ablation analysis, they qualitatively improve scenes with challenging geometry and faraway backgrounds (Figure [9](https://arxiv.org/html/2512.10685v1#A4.F9 "Figure 9 ‣ D.8 Additional Qualitative Results ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second")). We also observe that our regularizers boost rendering speed (Table [9](https://arxiv.org/html/2512.10685v1#A4.T9 "Table 9 ‣ Losses. ‣ D.5 Ablation Studies ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second")), which we attribute to the fact that they prevent degenerate or very large Gaussians.

Because of the importance of the perceptual loss, we separately evaluated the performance improvements from the Gram matrix component (Table [10](https://arxiv.org/html/2512.10685v1#A4.T10 "Table 10 ‣ Losses. ‣ D.5 Ablation Studies ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second")). Our results show that adding the Gram-matrix loss significantly improves results.

Table 8:Ablation study on loss components. The perceptual loss significantly enhances image quality; regularizer losses (ℒreg≜∑r∈ℛλr​ℒr in Eq. [3.12](https://arxiv.org/html/2512.10685v1#S3.E12 "Equation 3.12 ‣ 3.4 Training Objectives ‣ 3 Method ‣ Sharp Monocular View Synthesis in Less Than a Second")) do not have a strong effect on the metrics but yield qualitative improvements. (See Figure [9](https://arxiv.org/html/2512.10685v1#A4.F9 "Figure 9 ‣ D.8 Additional Qualitative Results ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second").)

|   |   |   |   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|---|---|---|
|ℒcolor+ℒalpha|ℒdepth|ℒpercep|ℒreg|ScanNet++|   |   |   |Tanks and Temples|   |   |   |
|DISTS↓|LPIPS↓|PSNR↑|SSIM↑|DISTS↓|LPIPS↓|PSNR↑|SSIM↑|
|✓|✗|✗|✗|0.229|0.414|18.18|0.768|0.301|0.656|14.75|0.520|
|✓|✓|✗|✗|0.162|0.270|22.95|0.844|0.239|0.548|16.23|0.550|
|✓|✓|✓|✗|0.063|0.143|23.65|0.843|0.126|0.421|16.29|0.531|
|✓|✓|✓|✓|0.064|0.147|22.61|0.829|0.126|0.419|16.19|0.523|

Table 9:Effect of loss terms on rendering speed. Median rendering latency per frame for different loss combinations. Loss terms improve rendering speed.

|   |   |   |   |   |   |
|---|---|---|---|---|---|
|ℒcolor+ℒalpha|ℒdepth|ℒpercep|ℒreg|ScanNet++|Tanks and Temples|
|Latency↓|Latency↓|
|✓|✗|✗|✗|22.2 ms|15.5 ms|
|✓|✓|✗|✗|12.2 ms|8.8 ms|
|✓|✓|✓|✗|6.2 ms|5.6 ms|
|✓|✓|✓|✓|5.5 ms|4.9 ms|

Table 10:Ablation study on perceptual loss. Adding the Gram matrix loss improves performance.

|   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|
|Gram loss|ScanNet++|   |   |   |Tanks and Temples|   |   |   |
|DISTS↓|LPIPS↓|PSNR↑|SSIM↑|DISTS↓|LPIPS↓|PSNR↑|SSIM↑|
|✗|0.070|0.153|22.26|0.827|0.130|0.441|15.89|0.517|
|✓|0.064|0.147|22.61|0.829|0.127|0.420|16.19|0.522|

##### Depth Adjustment.

Table [11](https://arxiv.org/html/2512.10685v1#A4.T11 "Table 11 ‣ Depth Adjustment. ‣ D.5 Ablation Studies ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second") evaluates the contribution of learned depth adjustment during training. The depth adjustment consistently improves perceptual image fidelity metrics. This can also be seen in the qualitative examples in Figure [10](https://arxiv.org/html/2512.10685v1#A4.F10 "Figure 10 ‣ D.8 Additional Qualitative Results ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second"), where the use of the depth adjustment during training yields a model that synthesizes sharper views.

Table 11:Ablation study on depth adjustment. Using the learned depth adjustment module consistently improves image quality. See also Figure [10](https://arxiv.org/html/2512.10685v1#A4.F10 "Figure 10 ‣ D.8 Additional Qualitative Results ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second").

|   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|
|Learned|ScanNet++|   |   |   |Tanks and Temples|   |   |   |
|DISTS↓|LPIPS↓|PSNR↑|SSIM↑|DISTS↓|LPIPS↓|PSNR↑|SSIM↑|
|✗|0.077|0.154|22.89|0.838|0.148|0.444|16.04|0.519|
|✓|0.064|0.147|22.61|0.829|0.126|0.419|16.19|0.523|

##### Self-supervised Fine-tuning.

Table [12](https://arxiv.org/html/2512.10685v1#A4.T12 "Table 12 ‣ Self-supervised Fine-tuning. ‣ D.5 Ablation Studies ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second") evaluates the contribution of self-supervised fine-tuning on real images (Stage 2 in Section [3.3](https://arxiv.org/html/2512.10685v1#S3.SS3 "3.3 Training Strategy ‣ 3 Method ‣ Sharp Monocular View Synthesis in Less Than a Second")). The metrics on the ablation datasets are on par, but qualitative analysis in Figure [11](https://arxiv.org/html/2512.10685v1#A4.F11 "Figure 11 ‣ D.8 Additional Qualitative Results ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second") indicates that self-supervised fine-tuning yields sharper images. We hypothesize that these improvements are due to the limited presence of complex view-dependent effects in synthetic data.

Table 12:Ablation study on self-supervised fine-tuning. While SSFT does not yield consistent metric improvement across datasets, we found it helpful in qualitative studies. (See Figure [11](https://arxiv.org/html/2512.10685v1#A4.F11 "Figure 11 ‣ D.8 Additional Qualitative Results ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second").)

|   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|
|SSL|ScanNet++|   |   |   |Tanks and Temples|   |   |   |
|DISTS↓|LPIPS↓|PSNR↑|SSIM↑|DISTS↓|LPIPS↓|PSNR↑|SSIM↑|
|✗|0.063|0.142|22.86|0.835|0.125|0.433|15.91|0.513|
|✓|0.071|0.154|22.63|0.833|0.122|0.421|16.33|0.528|

##### Unfreezing Backbone.

Unfreezing the monodepth backbone improves view synthesis fidelity, both quantitatively (Table [13](https://arxiv.org/html/2512.10685v1#A4.T13 "Table 13 ‣ Unfreezing Backbone. ‣ D.5 Ablation Studies ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second")) and qualitatively (Figure [12](https://arxiv.org/html/2512.10685v1#A4.F12 "Figure 12 ‣ D.8 Additional Qualitative Results ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second")). Qualitatively, we observe that unfreezing the monodepth backbone resolves boundary artifacts, improves reflections, and resolves artifacts in scenes with challenging geometry.

Table 13:Ablation study on unfreezing the monodepth backbone. See also Figure [12](https://arxiv.org/html/2512.10685v1#A4.F12 "Figure 12 ‣ D.8 Additional Qualitative Results ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second").

|   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|
|Unfreeze|ScanNet++|   |   |   |Tanks and Temples|   |   |   |
|DISTS↓|LPIPS↓|PSNR↑|SSIM↑|DISTS↓|LPIPS↓|PSNR↑|SSIM↑|
|✗|0.084|0.158|22.21|0.833|0.139|0.434|15.83|0.506|
|✓|0.064|0.147|22.61|0.829|0.126|0.419|16.19|0.523|

##### Number of Gaussians.

Table [14](https://arxiv.org/html/2512.10685v1#A4.T14 "Table 14 ‣ Number of Gaussians. ‣ D.5 Ablation Studies ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second") evaluates the contribution of the number of Gaussians that we output from our network. We compare the full 2×784×784≈1.2​M output to a 2× and 4× downsampled output. We see that performance of our method improves when we predict more Gaussians. This is confirmed by our qualitative results in Figure [13](https://arxiv.org/html/2512.10685v1#A4.F13 "Figure 13 ‣ D.8 Additional Qualitative Results ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second").

Table 14:Ablation study on number of predicted Gaussians. Increasing the number of Gaussians improves performance. See also Figure [13](https://arxiv.org/html/2512.10685v1#A4.F13 "Figure 13 ‣ D.8 Additional Qualitative Results ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second").

|   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|
|# Gaussians|ScanNet++|   |   |   |Tanks and Temples|   |   |   |
|DISTS↓|LPIPS↓|PSNR↑|SSIM↑|DISTS↓|LPIPS↓|PSNR↑|SSIM↑|
|2×196×196|0.110|0.199|20.46|0.799|0.181|0.458|16.27|0.525|
|2×392×392|0.077|0.160|22.00|0.822|0.140|0.425|16.23|0.525|
|2×784×784|0.064|0.147|22.61|0.829|0.126|0.419|16.19|0.523|

#### D.6Motion Range

While SHARP excels at generating high-quality nearby views (e.g. for AR/VR applications), it was not designed for synthesis of faraway views that have little overlap with the source image.

In Figure [7](https://arxiv.org/html/2512.10685v1#A4.F7 "Figure 7 ‣ D.6 Motion Range ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second") we study the the perceptual metrics trend against the motion values (measured by pairwise camera baseline size in meter) in our evaluation setup (see Section [D.1](https://arxiv.org/html/2512.10685v1#A4.SS1 "D.1 Evaluation Dataset Setup ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second")). Experiments show that while SHARP works well, as expected, on small camera motion (<0.5 meter), it retains its quality on larger motion and performs better than most other approaches on extended motion ranges. The SOTA diffusion-based approach Gen3C only outperforms SHARP on ETH3D with motion >3 meter and on ScanNet++ with motion >0.5 meter. We also see that with privileged info, the quality regression over motion range can be further alleviated. In summary, quantitative analysis show that while the desired motion range is around half a meter, our approach still works reasonably well on larger camera displacement.

|   |
|---|
|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/baseline_vs_dists_nonaligned.jpg)|
|(a) Plots of camera baseline size vs. DISTS metric.|
|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/baseline_vs_dists_aligned.jpg)|
|(b) Plots of camera baseline size vs. DISTS metric, with privileged info.|

Figure 7:Motion range analysis on the evaluation set. Shade indicates standard deviation. Unshaded data points indicate one single samples in the bin. Bins without samples are skipped, _c.f._ Figure [6](https://arxiv.org/html/2512.10685v1#A4.F6 "Figure 6 ‣ D.1 Evaluation Dataset Setup ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second"). SHARP works consistently the best with camera baseline sizes <0.5 meter, and maintains comparable results against diffusion-based approaches on larger motion ranges. It remains the best or the second best up to 3 meters.

In Figure [14](https://arxiv.org/html/2512.10685v1#A4.F14 "Figure 14 ‣ D.8 Additional Qualitative Results ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second") we deliberately extend the range of motion beyond SHARP’s intended operating regime. To ensure comparable visual quality of baselines, we provide monodepth from Depth Pro as privileged depth information to all methods in this analysis. Per discussion in Section [D.4](https://arxiv.org/html/2512.10685v1#A4.SS4 "D.4 Evaluation with Privileged Depth Information ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second"), we do not show SVC and LVSM results in Figure [14](https://arxiv.org/html/2512.10685v1#A4.F14 "Figure 14 ‣ D.8 Additional Qualitative Results ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second") since they cannot make use of privileged information and cannot perform a scale sweep due to a lack of ground truth novel view.

Qualitatively, we see that extending the range of motion reduces image fidelity in all regression-based approaches. On the other hand, diffusion-based approaches such as Gen3C can synthesize content even for far-away views. However, we also observe the tendency by diffusion models to alter the content of the image even for nearby views (e.g., the stirrups and horse’s tail in Figure [14](https://arxiv.org/html/2512.10685v1#A4.F14 "Figure 14 ‣ D.8 Additional Qualitative Results ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second")).

We believe it is an interesting research direction to combine the strengths of diffusion-based approaches (synthesis of faraway content) and feed-forward models such as SHARP (interactive generation of a 3D representation that can be rendered in real time).

#### D.7Failure Cases

Apart from the failure of excessive motion range that exceed the operation domain, like all machine learning models, SHARP may fail under challenging scenarios. In Figure [8](https://arxiv.org/html/2512.10685v1#A4.F8 "Figure 8 ‣ D.7 Failure Cases ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second") we show several such examples.

- • 
    
    In a _macro photo_, due to strong depth-of-field effect, the bee’s depth is incorrectly interpreted as behind the flowers, leading to detached wings and distorted tail in novel view synthesis.
    
- • 
    
    Due to the rich starry texture in a _night photo_, the sky is interpreted as a curvy surface instead of a plain surface faraway, causing heavily distorted rendering.
    
- • 
    
    The _complex reflection_ in water is interpreted by the network as a distant mountain, therefore the water surface is broken.
    

These failures are root caused by the depth model, and despite unfreezing the depth backbone, SHARP is unable to recover from the corrupted initialization. We regard this as a long-tail problem of depth prediction. Retraining the depth backbone with higher capacity through more data may alleviate the issue; involving diffusion models with richer priors may be an alternative solution in the future.

|   |   |   |
|---|---|---|
|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/failure_cases/_AeQepiyWgQ_original.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/failure_cases/_AeQepiyWgQ_rgb_frame2.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/failure_cases/_AeQepiyWgQ_depth_frame1.jpg)|
|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/failure_cases/_Bq3TeSBRdE_original.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/failure_cases/_Bq3TeSBRdE_rgb_frame2.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/failure_cases/_Bq3TeSBRdE_depth_frame1.jpg)|
|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/failure_cases/_8mdYBdLYj0_original.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/failure_cases/_8mdYBdLYj0_rgb_frame2.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/failure_cases/_8mdYBdLYj0_depth_frame1.jpg)|
|(a) Input|(b) Rendered novel view|(c) Rendered inverse depth|

Figure 8:Depth failures in challenging edge cases.

#### D.8Additional Qualitative Results

Here we provide extensive qualitative results of all approaches on all datasets in Figures [15](https://arxiv.org/html/2512.10685v1#A4.F15 "Figure 15 ‣ D.8 Additional Qualitative Results ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second")–[26](https://arxiv.org/html/2512.10685v1#A4.F26 "Figure 26 ‣ D.8 Additional Qualitative Results ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second"), both with and without privileged depth information. LVSM, SVC, ViewCrafter, and Gen3C operate at a fixed aspect ratio, therefore we pad their output to match the original image resolution. SHARP consistently produces high-fidelity results. Further video results can be found in [https://apple.github.io/ml-sharp](https://apple.github.io/ml-sharp).

|   |   |   |   |
|---|---|---|---|
|color + alpha|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_color_alpha_loss/target/_0eTYFd2pzM_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_color_alpha_loss/target/_0LRNICuIWY_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_color_alpha_loss/target/_2jnLE7CWsE_0000_0001.jpg)|
|+ depth|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_color_alpha_disparity_loss/target/_0eTYFd2pzM_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_color_alpha_disparity_loss/target/_0LRNICuIWY_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_color_alpha_disparity_loss/target/_2jnLE7CWsE_0000_0001.jpg)|
|+ perceptual|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_color_alpha_disparity_perceptual_loss/target/_0eTYFd2pzM_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_color_alpha_disparity_perceptual_loss/target/_0LRNICuIWY_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_color_alpha_disparity_perceptual_loss/target/_2jnLE7CWsE_0000_0001.jpg)|
|+ regularizer|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_public_ablation_base/target/_0eTYFd2pzM_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_public_ablation_base/target/_0LRNICuIWY_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_public_ablation_base/target/_2jnLE7CWsE_0000_0001.jpg)|

Figure 9:The effect of different loss terms.

|   |   |   |   |
|---|---|---|---|
|None|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_alignment_none/target/_0lImL6bFwQ_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_alignment_none/target/_2pk0ACDUVM_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_alignment_none/target/_3S74BwJ17w_0000_0001.jpg)|
|+ learned adjustment|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_public_ablation_base/target/_0lImL6bFwQ_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_public_ablation_base/target/_2pk0ACDUVM_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_public_ablation_base/target/_3S74BwJ17w_0000_0001.jpg)|

Figure 10:The effect of learned depth adjustment.

|   |   |   |   |
|---|---|---|---|
|Stage 1|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_public_large/target/__HUt9l_aK0_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_public_large/target/_3W4bwHiQRc_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_public_large/target/_4KRzexcFyA_0000_0001.jpg)|
|Stage 1+2|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_ssl_large/target/__HUt9l_aK0_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_ssl_large/target/_3W4bwHiQRc_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_ssl_large/target/_4KRzexcFyA_0000_0001.jpg)|

Figure 11:The effect of SSFT.

|   |   |   |   |
|---|---|---|---|
|Freeze|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_unfreeze_none/target/_1EeapItydo_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_unfreeze_none/target/_1h_NN3nqzI_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_unfreeze_none/target/_2WE1fsVNJg_0000_0001.jpg)|
|Unfreeze|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_public_ablation_base/target/_1EeapItydo_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_public_ablation_base/target/_1h_NN3nqzI_0000_0001.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_public_ablation_base/target/_2WE1fsVNJg_0000_0001.jpg)|

Figure 12:The effect of unfreezing the monodepth backbone.

|   |   |   |   |
|---|---|---|---|
|4× downsampled|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_stride_8/collage/_EesKmt5kN4_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_stride_8/collage/_0tX_xz9o5Q_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_stride_8/collage/_7e7bTO0msE_0000_0001_overlay.jpg)|
|2× downsampled|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_stride_4/collage/_EesKmt5kN4_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_stride_4/collage/_0tX_xz9o5Q_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_stride_4/collage/_7e7bTO0msE_0000_0001_overlay.jpg)|
|Full model|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_public_ablation_base/collage/_EesKmt5kN4_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_public_ablation_base/collage/_0tX_xz9o5Q_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/ablation/UnsplashCurated/sharp_public_ablation_base/collage/_7e7bTO0msE_0000_0001_overlay.jpg)|

Figure 13:The effect of the number of output Gaussians.

|   |   |   |   |   |
|---|---|---|---|---|
|SHARP (ours)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/videos/SharpPaperVideo/sharp_public_aligned/target/_B_lu05yfgE_0000_0001_0000.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/videos/SharpPaperVideo/sharp_public_aligned/target/_B_lu05yfgE_0000_0001_0005.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/videos/SharpPaperVideo/sharp_public_aligned/target/_B_lu05yfgE_0000_0001_0011.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/videos/SharpPaperVideo/sharp_public_aligned/target/_B_lu05yfgE_0000_0001_0016.jpg)|
|Gen3C|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/videos/SharpPaperVideo/gen3c_aligned/target/_B_lu05yfgE_0000_0001_0000.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/videos/SharpPaperVideo/gen3c_aligned/target/_B_lu05yfgE_0000_0001_0005.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/videos/SharpPaperVideo/gen3c_aligned/target/_B_lu05yfgE_0000_0001_0011.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/videos/SharpPaperVideo/gen3c_aligned/target/_B_lu05yfgE_0000_0001_0016.jpg)|
|ViewCrafter|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/videos/SharpPaperVideo/view_crafter_aligned/target/_B_lu05yfgE_0000_0001_0000.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/videos/SharpPaperVideo/view_crafter_aligned/target/_B_lu05yfgE_0000_0001_0005.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/videos/SharpPaperVideo/view_crafter_aligned/target/_B_lu05yfgE_0000_0001_0011.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/videos/SharpPaperVideo/view_crafter_aligned/target/_B_lu05yfgE_0000_0001_0016.jpg)|
|Flash3D|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/videos/SharpPaperVideo/flash3d_aligned/target/_B_lu05yfgE_0000_0001_0000.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/videos/SharpPaperVideo/flash3d_aligned/target/_B_lu05yfgE_0000_0001_0005.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/videos/SharpPaperVideo/flash3d_aligned/target/_B_lu05yfgE_0000_0001_0011.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/videos/SharpPaperVideo/flash3d_aligned/target/_B_lu05yfgE_0000_0001_0016.jpg)|
|TMPI|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/videos/SharpPaperVideo/tmpi_aligned/target/_B_lu05yfgE_0000_0001_0000.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/videos/SharpPaperVideo/tmpi_aligned/target/_B_lu05yfgE_0000_0001_0005.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/videos/SharpPaperVideo/tmpi_aligned/target/_B_lu05yfgE_0000_0001_0011.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/videos/SharpPaperVideo/tmpi_aligned/target/_B_lu05yfgE_0000_0001_0016.jpg)|
||0cm|50cm|100cm|150cm|

Figure 14:Extending the range of motion beyond nearby views, with monodepth as privileged depth information. We do not show LVSM and SVC as they cannot make use of privileged depth information, see discussions in Section [D.4](https://arxiv.org/html/2512.10685v1#A4.SS4 "D.4 Evaluation with Privileged Depth Information ‣ Appendix D Experiments ‣ Supplementary Material ‣ Sharp Monocular View Synthesis in Less Than a Second"). ViewCrafter and Gen3C operates at a fixed aspect ratio, therefore we pad their output to match the original image resolution.

|   |   |   |   |
|---|---|---|---|
|Flash3D|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/flash3d/collage/3a0f717d79_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/flash3d/collage/ea068642ad_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/flash3d/collage/fec6df7a40_000_0000_0001_overlay.jpg)|
|TMPI|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/tmpi/collage/3a0f717d79_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/tmpi/collage/ea068642ad_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/tmpi/collage/fec6df7a40_000_0000_0001_overlay.jpg)|
|LVSM|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/lvsm/collage/3a0f717d79_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/lvsm/collage/ea068642ad_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/lvsm/collage/fec6df7a40_000_0000_0001_overlay.jpg)|
|SVC|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/stable_virtual_camera/collage/3a0f717d79_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/stable_virtual_camera/collage/ea068642ad_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/stable_virtual_camera/collage/fec6df7a40_000_0000_0001_overlay.jpg)|
|ViewCrafter|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/view_crafter/collage/3a0f717d79_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/view_crafter/collage/ea068642ad_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/view_crafter/collage/fec6df7a40_000_0000_0001_overlay.jpg)|
|Gen3C|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/gen3c/collage/3a0f717d79_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/gen3c/collage/ea068642ad_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/gen3c/collage/fec6df7a40_000_0000_0001_overlay.jpg)|
|SHARP (ours)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/sharp_public/collage/3a0f717d79_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/sharp_public/collage/ea068642ad_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/sharp_public/collage/fec6df7a40_000_0000_0001_overlay.jpg)|
|Ground truth|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/ground_truth/collage/3a0f717d79_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/ground_truth/collage/ea068642ad_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/ground_truth/collage/fec6df7a40_000_0000_0001_overlay.jpg)|

Figure 15:Qualitative comparison on Middlebury.

|   |   |   |   |
|---|---|---|---|
|Flash3D|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/flash3d_aligned/collage/3a0f717d79_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/flash3d_aligned/collage/ea068642ad_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/flash3d_aligned/collage/fec6df7a40_000_0000_0001_overlay.jpg)|
|TMPI|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/tmpi_aligned/collage/3a0f717d79_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/tmpi_aligned/collage/ea068642ad_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/tmpi_aligned/collage/fec6df7a40_000_0000_0001_overlay.jpg)|
|LVSM|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/lvsm_sweep/collage/3a0f717d79_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/lvsm_sweep/collage/ea068642ad_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/lvsm_sweep/collage/fec6df7a40_000_0000_0001_overlay.jpg)|
|SVC|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/stable_virtual_camera_sweep/collage/3a0f717d79_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/stable_virtual_camera_sweep/collage/ea068642ad_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/stable_virtual_camera_sweep/collage/fec6df7a40_000_0000_0001_overlay.jpg)|
|ViewCrafter|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/view_crafter_aligned/collage/3a0f717d79_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/view_crafter_aligned/collage/ea068642ad_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/view_crafter_aligned/collage/fec6df7a40_000_0000_0001_overlay.jpg)|
|Gen3C|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/gen3c_aligned/collage/3a0f717d79_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/gen3c_aligned/collage/ea068642ad_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/gen3c_aligned/collage/fec6df7a40_000_0000_0001_overlay.jpg)|
|SHARP (ours)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/sharp_public_aligned/collage/3a0f717d79_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/sharp_public_aligned/collage/ea068642ad_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/sharp_public_aligned/collage/fec6df7a40_000_0000_0001_overlay.jpg)|
|Ground truth|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/ground_truth/collage/3a0f717d79_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/ground_truth/collage/ea068642ad_000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Middlebury/ground_truth/collage/fec6df7a40_000_0000_0001_overlay.jpg)|

Figure 16:Qualitative comparison on Middlebury with privileged depth information.

|   |   |   |   |
|---|---|---|---|
|Flash3D|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/flash3d/collage/train+balanced+Moka1+camera_00+im6.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/flash3d/collage/train+balanced+Canteen+camera_00+im9.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/flash3d/collage/train+balanced+Motorcycle+camera_00+im6.png_00000_0000_0001_overlay.jpg)|
|TMPI|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/tmpi/collage/train+balanced+Moka1+camera_00+im6.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/tmpi/collage/train+balanced+Canteen+camera_00+im9.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/tmpi/collage/train+balanced+Motorcycle+camera_00+im6.png_00000_0000_0001_overlay.jpg)|
|LVSM|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/lvsm/collage/train+balanced+Moka1+camera_00+im6.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/lvsm/collage/train+balanced+Canteen+camera_00+im9.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/lvsm/collage/train+balanced+Motorcycle+camera_00+im6.png_00000_0000_0001_overlay.jpg)|
|SVC|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/stable_virtual_camera/collage/train+balanced+Moka1+camera_00+im6.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/stable_virtual_camera/collage/train+balanced+Canteen+camera_00+im9.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/stable_virtual_camera/collage/train+balanced+Motorcycle+camera_00+im6.png_00000_0000_0001_overlay.jpg)|
|ViewCrafter|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/view_crafter/collage/train+balanced+Moka1+camera_00+im6.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/view_crafter/collage/train+balanced+Canteen+camera_00+im9.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/view_crafter/collage/train+balanced+Motorcycle+camera_00+im6.png_00000_0000_0001_overlay.jpg)|
|Gen3C|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/gen3c/collage/train+balanced+Moka1+camera_00+im6.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/gen3c/collage/train+balanced+Canteen+camera_00+im9.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/gen3c/collage/train+balanced+Motorcycle+camera_00+im6.png_00000_0000_0001_overlay.jpg)|
|SHARP (ours)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/sharp_public/collage/train+balanced+Moka1+camera_00+im6.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/sharp_public/collage/train+balanced+Canteen+camera_00+im9.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/sharp_public/collage/train+balanced+Motorcycle+camera_00+im6.png_00000_0000_0001_overlay.jpg)|
|Ground truth|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/ground_truth/collage/train+balanced+Moka1+camera_00+im6.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/ground_truth/collage/train+balanced+Canteen+camera_00+im9.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/ground_truth/collage/train+balanced+Motorcycle+camera_00+im6.png_00000_0000_0001_overlay.jpg)|

Figure 17:Qualitative comparison on Booster.

|   |   |   |   |
|---|---|---|---|
|Flash3D|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/flash3d_aligned/collage/train+balanced+Moka1+camera_00+im6.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/flash3d_aligned/collage/train+balanced+Canteen+camera_00+im9.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/flash3d_aligned/collage/train+balanced+Motorcycle+camera_00+im6.png_00000_0000_0001_overlay.jpg)|
|TMPI|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/tmpi_aligned/collage/train+balanced+Moka1+camera_00+im6.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/tmpi_aligned/collage/train+balanced+Canteen+camera_00+im9.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/tmpi_aligned/collage/train+balanced+Motorcycle+camera_00+im6.png_00000_0000_0001_overlay.jpg)|
|LVSM|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/lvsm_sweep/collage/train+balanced+Moka1+camera_00+im6.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/lvsm_sweep/collage/train+balanced+Canteen+camera_00+im9.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/lvsm_sweep/collage/train+balanced+Motorcycle+camera_00+im6.png_00000_0000_0001_overlay.jpg)|
|SVC|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/stable_virtual_camera_sweep/collage/train+balanced+Moka1+camera_00+im6.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/stable_virtual_camera_sweep/collage/train+balanced+Canteen+camera_00+im9.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/stable_virtual_camera_sweep/collage/train+balanced+Motorcycle+camera_00+im6.png_00000_0000_0001_overlay.jpg)|
|ViewCrafter|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/view_crafter_aligned/collage/train+balanced+Moka1+camera_00+im6.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/view_crafter_aligned/collage/train+balanced+Canteen+camera_00+im9.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/view_crafter_aligned/collage/train+balanced+Motorcycle+camera_00+im6.png_00000_0000_0001_overlay.jpg)|
|Gen3C|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/gen3c_aligned/collage/train+balanced+Moka1+camera_00+im6.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/gen3c_aligned/collage/train+balanced+Canteen+camera_00+im9.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/gen3c_aligned/collage/train+balanced+Motorcycle+camera_00+im6.png_00000_0000_0001_overlay.jpg)|
|SHARP (ours)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/sharp_public_aligned/collage/train+balanced+Moka1+camera_00+im6.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/sharp_public_aligned/collage/train+balanced+Canteen+camera_00+im9.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/sharp_public_aligned/collage/train+balanced+Motorcycle+camera_00+im6.png_00000_0000_0001_overlay.jpg)|
|Ground truth|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/ground_truth/collage/train+balanced+Moka1+camera_00+im6.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/ground_truth/collage/train+balanced+Canteen+camera_00+im9.png_00000_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/Booster/ground_truth/collage/train+balanced+Motorcycle+camera_00+im6.png_00000_0000_0001_overlay.jpg)|

Figure 18:Qualitative comparison on Booster with privileged depth information.

|   |   |   |   |
|---|---|---|---|
|Flash3D|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/flash3d/collage/c5439f4607_00016_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/flash3d/collage/09c1414f1b_00074_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/flash3d/collage/d755b3d9d8_00014_0000_0002_overlay.jpg)|
|TMPI|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/tmpi/collage/c5439f4607_00016_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/tmpi/collage/09c1414f1b_00074_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/tmpi/collage/d755b3d9d8_00014_0000_0002_overlay.jpg)|
|LVSM|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/lvsm/collage/c5439f4607_00016_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/lvsm/collage/09c1414f1b_00074_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/lvsm/collage/d755b3d9d8_00014_0000_0002_overlay.jpg)|
|SVC|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/stable_virtual_camera/collage/c5439f4607_00016_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/stable_virtual_camera/collage/09c1414f1b_00074_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/stable_virtual_camera/collage/d755b3d9d8_00014_0000_0002_overlay.jpg)|
|ViewCrafter|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/view_crafter/collage/c5439f4607_00016_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/view_crafter/collage/09c1414f1b_00074_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/view_crafter/collage/d755b3d9d8_00014_0000_0002_overlay.jpg)|
|Gen3C|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/gen3c/collage/c5439f4607_00016_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/gen3c/collage/09c1414f1b_00074_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/gen3c/collage/d755b3d9d8_00014_0000_0002_overlay.jpg)|
|SHARP (ours)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/sharp_public/collage/c5439f4607_00016_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/sharp_public/collage/09c1414f1b_00074_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/sharp_public/collage/d755b3d9d8_00014_0000_0002_overlay.jpg)|
|Ground truth|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/ground_truth/collage/c5439f4607_00016_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/ground_truth/collage/09c1414f1b_00074_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/ground_truth/collage/d755b3d9d8_00014_0000_0002_overlay.jpg)|

Figure 19:Qualitative comparison on ScanNet++.

|   |   |   |   |
|---|---|---|---|
|Flash3D|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/flash3d_aligned/collage/c5439f4607_00016_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/flash3d_aligned/collage/09c1414f1b_00074_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/flash3d_aligned/collage/d755b3d9d8_00014_0000_0002_overlay.jpg)|
|TMPI|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/tmpi_aligned/collage/c5439f4607_00016_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/tmpi_aligned/collage/09c1414f1b_00074_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/tmpi_aligned/collage/d755b3d9d8_00014_0000_0002_overlay.jpg)|
|LVSM|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/lvsm_sweep/collage/c5439f4607_00016_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/lvsm_sweep/collage/09c1414f1b_00074_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/lvsm_sweep/collage/d755b3d9d8_00014_0000_0002_overlay.jpg)|
|SVC|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/stable_virtual_camera_sweep/collage/c5439f4607_00016_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/stable_virtual_camera_sweep/collage/09c1414f1b_00074_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/stable_virtual_camera_sweep/collage/d755b3d9d8_00014_0000_0002_overlay.jpg)|
|ViewCrafter|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/view_crafter_aligned/collage/c5439f4607_00016_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/view_crafter_aligned/collage/09c1414f1b_00074_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/view_crafter_aligned/collage/d755b3d9d8_00014_0000_0002_overlay.jpg)|
|Gen3C|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/gen3c_aligned/collage/c5439f4607_00016_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/gen3c_aligned/collage/09c1414f1b_00074_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/gen3c_aligned/collage/d755b3d9d8_00014_0000_0002_overlay.jpg)|
|SHARP (ours)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/sharp_public_aligned/collage/c5439f4607_00016_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/sharp_public_aligned/collage/09c1414f1b_00074_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/sharp_public_aligned/collage/d755b3d9d8_00014_0000_0002_overlay.jpg)|
|Ground truth|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/ground_truth/collage/c5439f4607_00016_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/ground_truth/collage/09c1414f1b_00074_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ScanNetPP/ground_truth/collage/d755b3d9d8_00014_0000_0002_overlay.jpg)|

Figure 20:Qualitative comparison on ScanNet++ with privileged depth information.

|   |   |   |   |   |   |
|---|---|---|---|---|---|
|Flash3D|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/flash3d/collage/train+scene_055_00025_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/flash3d/collage/mouse+scene_004_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/flash3d/collage/apple+scene_014_00000_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/flash3d/collage/flower_pot+scene_014_00006_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/flash3d/collage/handbag+scene_017_00020_0000_0003_overlay.jpg)|
|TMPI|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/tmpi/collage/train+scene_055_00025_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/tmpi/collage/mouse+scene_004_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/tmpi/collage/apple+scene_014_00000_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/tmpi/collage/flower_pot+scene_014_00006_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/tmpi/collage/handbag+scene_017_00020_0000_0003_overlay.jpg)|
|LVSM|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/lvsm/collage/train+scene_055_00025_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/lvsm/collage/mouse+scene_004_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/lvsm/collage/apple+scene_014_00000_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/lvsm/collage/flower_pot+scene_014_00006_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/lvsm/collage/handbag+scene_017_00020_0000_0003_overlay.jpg)|
|SVC|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/stable_virtual_camera/collage/train+scene_055_00025_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/stable_virtual_camera/collage/mouse+scene_004_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/stable_virtual_camera/collage/apple+scene_014_00000_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/stable_virtual_camera/collage/flower_pot+scene_014_00006_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/stable_virtual_camera/collage/handbag+scene_017_00020_0000_0003_overlay.jpg)|
|ViewCrafter|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/view_crafter/collage/train+scene_055_00025_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/view_crafter/collage/mouse+scene_004_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/view_crafter/collage/apple+scene_014_00000_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/view_crafter/collage/flower_pot+scene_014_00006_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/view_crafter/collage/handbag+scene_017_00020_0000_0003_overlay.jpg)|
|Gen3C|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/gen3c/collage/train+scene_055_00025_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/gen3c/collage/mouse+scene_004_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/gen3c/collage/apple+scene_014_00000_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/gen3c/collage/flower_pot+scene_014_00006_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/gen3c/collage/handbag+scene_017_00020_0000_0003_overlay.jpg)|
|SHARP (ours)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/sharp_public/collage/train+scene_055_00025_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/sharp_public/collage/mouse+scene_004_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/sharp_public/collage/apple+scene_014_00000_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/sharp_public/collage/flower_pot+scene_014_00006_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/sharp_public/collage/handbag+scene_017_00020_0000_0003_overlay.jpg)|
|Ground truth|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/ground_truth/collage/train+scene_055_00025_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/ground_truth/collage/mouse+scene_004_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/ground_truth/collage/apple+scene_014_00000_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/ground_truth/collage/flower_pot+scene_014_00006_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/ground_truth/collage/handbag+scene_017_00020_0000_0003_overlay.jpg)|

Figure 21:Qualitative comparison on WildRGBD.

|   |   |   |   |   |   |
|---|---|---|---|---|---|
|Flash3D|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/flash3d_aligned/collage/train+scene_055_00025_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/flash3d_aligned/collage/mouse+scene_004_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/flash3d_aligned/collage/apple+scene_014_00000_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/flash3d_aligned/collage/flower_pot+scene_014_00006_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/flash3d_aligned/collage/handbag+scene_017_00020_0000_0003_overlay.jpg)|
|TMPI|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/tmpi_aligned/collage/train+scene_055_00025_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/tmpi_aligned/collage/mouse+scene_004_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/tmpi_aligned/collage/apple+scene_014_00000_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/tmpi_aligned/collage/flower_pot+scene_014_00006_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/tmpi_aligned/collage/handbag+scene_017_00020_0000_0003_overlay.jpg)|
|LVSM|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/lvsm_sweep/collage/train+scene_055_00025_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/lvsm_sweep/collage/mouse+scene_004_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/lvsm_sweep/collage/apple+scene_014_00000_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/lvsm_sweep/collage/flower_pot+scene_014_00006_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/lvsm_sweep/collage/handbag+scene_017_00020_0000_0003_overlay.jpg)|
|SVC|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/stable_virtual_camera_sweep/collage/train+scene_055_00025_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/stable_virtual_camera_sweep/collage/mouse+scene_004_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/stable_virtual_camera_sweep/collage/apple+scene_014_00000_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/stable_virtual_camera_sweep/collage/flower_pot+scene_014_00006_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/stable_virtual_camera_sweep/collage/handbag+scene_017_00020_0000_0003_overlay.jpg)|
|ViewCrafter|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/view_crafter_aligned/collage/train+scene_055_00025_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/view_crafter_aligned/collage/mouse+scene_004_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/view_crafter_aligned/collage/apple+scene_014_00000_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/view_crafter_aligned/collage/flower_pot+scene_014_00006_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/view_crafter_aligned/collage/handbag+scene_017_00020_0000_0003_overlay.jpg)|
|Gen3C|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/gen3c_aligned/collage/train+scene_055_00025_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/gen3c_aligned/collage/mouse+scene_004_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/gen3c_aligned/collage/apple+scene_014_00000_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/gen3c_aligned/collage/flower_pot+scene_014_00006_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/gen3c_aligned/collage/handbag+scene_017_00020_0000_0003_overlay.jpg)|
|SHARP (ours)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/sharp_public_aligned/collage/train+scene_055_00025_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/sharp_public_aligned/collage/mouse+scene_004_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/sharp_public_aligned/collage/apple+scene_014_00000_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/sharp_public_aligned/collage/flower_pot+scene_014_00006_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/sharp_public_aligned/collage/handbag+scene_017_00020_0000_0003_overlay.jpg)|
|Ground truth|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/ground_truth/collage/train+scene_055_00025_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/ground_truth/collage/mouse+scene_004_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/ground_truth/collage/apple+scene_014_00000_0000_0003_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/ground_truth/collage/flower_pot+scene_014_00006_0000_0002_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/WildRGBD/ground_truth/collage/handbag+scene_017_00020_0000_0003_overlay.jpg)|

Figure 22:Qualitative comparison on WildRGBD with privileged depth information.

|   |   |   |   |
|---|---|---|---|
|Flash3D|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/flash3d/collage/Meetingroom_00007_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/flash3d/collage/Church_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/flash3d/collage/Truck_00007_0000_0001_overlay.jpg)|
|TMPI|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/tmpi/collage/Meetingroom_00007_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/tmpi/collage/Church_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/tmpi/collage/Truck_00007_0000_0001_overlay.jpg)|
|LVSM|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/lvsm/collage/Meetingroom_00007_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/lvsm/collage/Church_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/lvsm/collage/Truck_00007_0000_0001_overlay.jpg)|
|SVC|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/stable_virtual_camera/collage/Meetingroom_00007_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/stable_virtual_camera/collage/Church_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/stable_virtual_camera/collage/Truck_00007_0000_0001_overlay.jpg)|
|ViewCrafter|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/view_crafter/collage/Meetingroom_00007_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/view_crafter/collage/Church_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/view_crafter/collage/Truck_00007_0000_0001_overlay.jpg)|
|Gen3C|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/gen3c/collage/Meetingroom_00007_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/gen3c/collage/Church_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/gen3c/collage/Truck_00007_0000_0001_overlay.jpg)|
|SHARP (ours)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/sharp_public/collage/Meetingroom_00007_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/sharp_public/collage/Church_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/sharp_public/collage/Truck_00007_0000_0001_overlay.jpg)|
|Ground truth|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/ground_truth/collage/Meetingroom_00007_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/ground_truth/collage/Church_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/ground_truth/collage/Truck_00007_0000_0001_overlay.jpg)|

Figure 23:Qualitative comparison on Tanks and Temples.

|   |   |   |   |
|---|---|---|---|
|Flash3D|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/flash3d_aligned/collage/Meetingroom_00007_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/flash3d_aligned/collage/Church_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/flash3d_aligned/collage/Truck_00007_0000_0001_overlay.jpg)|
|TMPI|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/tmpi_aligned/collage/Meetingroom_00007_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/tmpi_aligned/collage/Church_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/tmpi_aligned/collage/Truck_00007_0000_0001_overlay.jpg)|
|LVSM|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/lvsm_sweep/collage/Meetingroom_00007_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/lvsm_sweep/collage/Church_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/lvsm_sweep/collage/Truck_00007_0000_0001_overlay.jpg)|
|SVC|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/stable_virtual_camera_sweep/collage/Meetingroom_00007_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/stable_virtual_camera_sweep/collage/Church_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/stable_virtual_camera_sweep/collage/Truck_00007_0000_0001_overlay.jpg)|
|ViewCrafter|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/view_crafter_aligned/collage/Meetingroom_00007_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/view_crafter_aligned/collage/Church_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/view_crafter_aligned/collage/Truck_00007_0000_0001_overlay.jpg)|
|Gen3C|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/gen3c_aligned/collage/Meetingroom_00007_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/gen3c_aligned/collage/Church_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/gen3c_aligned/collage/Truck_00007_0000_0001_overlay.jpg)|
|SHARP (ours)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/sharp_public_aligned/collage/Meetingroom_00007_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/sharp_public_aligned/collage/Church_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/sharp_public_aligned/collage/Truck_00007_0000_0001_overlay.jpg)|
|Ground truth|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/ground_truth/collage/Meetingroom_00007_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/ground_truth/collage/Church_00004_0000_0004_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/TanksAndTemples/ground_truth/collage/Truck_00007_0000_0001_overlay.jpg)|

Figure 24:Qualitative comparison on Tanks and Temples with privileged depth information.

|   |   |   |   |
|---|---|---|---|
|Flash3D|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/flash3d/collage/terrains_00002_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/flash3d/collage/facade_00001_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/flash3d/collage/courtyard_00000_0000_0001_overlay.jpg)|
|TMPI|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/tmpi/collage/terrains_00002_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/tmpi/collage/facade_00001_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/tmpi/collage/courtyard_00000_0000_0001_overlay.jpg)|
|LVSM|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/lvsm/collage/terrains_00002_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/lvsm/collage/facade_00001_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/lvsm/collage/courtyard_00000_0000_0001_overlay.jpg)|
|SVC|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/stable_virtual_camera/collage/terrains_00002_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/stable_virtual_camera/collage/facade_00001_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/stable_virtual_camera/collage/courtyard_00000_0000_0001_overlay.jpg)|
|ViewCrafter|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/view_crafter/collage/terrains_00002_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/view_crafter/collage/facade_00001_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/view_crafter/collage/courtyard_00000_0000_0001_overlay.jpg)|
|Gen3C|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/gen3c/collage/terrains_00002_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/gen3c/collage/facade_00001_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/gen3c/collage/courtyard_00000_0000_0001_overlay.jpg)|
|SHARP (ours)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/sharp_public/collage/terrains_00002_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/sharp_public/collage/facade_00001_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/sharp_public/collage/courtyard_00000_0000_0001_overlay.jpg)|
|Ground truth|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/ground_truth/collage/terrains_00002_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/ground_truth/collage/facade_00001_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/ground_truth/collage/courtyard_00000_0000_0001_overlay.jpg)|

Figure 25:Qualitative comparison on ETH3D.

|   |   |   |   |
|---|---|---|---|
|Flash3D|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/flash3d_aligned/collage/terrains_00002_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/flash3d_aligned/collage/facade_00001_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/flash3d_aligned/collage/courtyard_00000_0000_0001_overlay.jpg)|
|TMPI|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/tmpi_aligned/collage/terrains_00002_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/tmpi_aligned/collage/facade_00001_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/tmpi_aligned/collage/courtyard_00000_0000_0001_overlay.jpg)|
|LVSM|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/lvsm_sweep/collage/terrains_00002_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/lvsm_sweep/collage/facade_00001_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/lvsm_sweep/collage/courtyard_00000_0000_0001_overlay.jpg)|
|SVC|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/stable_virtual_camera_sweep/collage/terrains_00002_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/stable_virtual_camera_sweep/collage/facade_00001_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/stable_virtual_camera_sweep/collage/courtyard_00000_0000_0001_overlay.jpg)|
|ViewCrafter|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/view_crafter_aligned/collage/terrains_00002_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/view_crafter_aligned/collage/facade_00001_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/view_crafter_aligned/collage/courtyard_00000_0000_0001_overlay.jpg)|
|Gen3C|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/gen3c_aligned/collage/terrains_00002_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/gen3c_aligned/collage/facade_00001_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/gen3c_aligned/collage/courtyard_00000_0000_0001_overlay.jpg)|
|SHARP (ours)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/sharp_public_aligned/collage/terrains_00002_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/sharp_public_aligned/collage/facade_00001_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/sharp_public_aligned/collage/courtyard_00000_0000_0001_overlay.jpg)|
|Ground truth|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/ground_truth/collage/terrains_00002_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/ground_truth/collage/facade_00001_0000_0001_overlay.jpg)|![Refer to caption](https://arxiv.org/html/2512.10685v1/figures/qualitative/ETH3D/ground_truth/collage/courtyard_00000_0000_0001_overlay.jpg)|

Figure 26:Qualitative comparison on ETH3D with privileged depth information.

### Appendix ELLM Usage Declaration

We used Claude Sonnet 4.5 to polish the writing (e.g. check grammar issues and find better synonyms), to help layout LaTeX tables and figures, and to build the front-end interface of the interactive video comparison.