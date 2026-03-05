# Adversarial Blind Epistemic Ensemble for Robotic Failure Prediction

## 1. System and Problem Framing

The target setting is a 4‑agent ensemble of NVIDIA Cosmos Reason 2 VLM/VLA instances (Cosmos‑Reason2) evaluating human–robot handoff videos frame‑by‑frame and deciding when to safely release an object. Cosmos Reason 2 provides video tokenization, spatio‑temporal reasoning, and trajectory outputs for physical AI tasks, with long‑context support and trajectory coordinate prediction for robot planning. Agents share a short‑term LiveKV memory and a long‑term ArchiveKV memory backed by FAISS, and operate under an epistemic budget: at each frame they either THINK (wait and observe) or ACT (predict release) in a stopping‑time POMDP.[^1][^2][^3][^4]

The goal of the "Adversarial Blind Epistemic Ensemble" is to:

- Inject physically grounded latent‑space plausibility scores and constraints into each agent’s context.
- Extract *epistemic* (not just softmax) uncertainty from VLMs and world models, especially about physical dynamics.
- Structure a multi‑agent ensemble with asymmetric parametrization and temporal masking to avoid consensus collapse and herding.
- Use all of the above to define an Act‑vs‑Think stopping rule that provably reduces catastrophic release errors.

The rest of this dossier is organized along the requested research vectors and ends each subsection with a concrete splice into the existing Act‑vs‑Think POMDP loop.

***

## 2. Latent Space Dynamics and JEPA‑Style World Models

### 2.1 V‑JEPA and Variational JEPA (VJEPA/BJEPA)

**Key papers/techniques**

- **V‑JEPA (Video Joint Embedding Predictive Architecture)** from Meta: a non‑generative video world model that predicts masked spatio‑temporal patches in an abstract latent space rather than pixels, improving sample efficiency by 1.5–6× vs generative baselines.[^5]
- **VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models** introduces a probabilistic JEPA that learns a predictive distribution over future latent states rather than a point estimate, unifying JEPA with predictive state representations and Bayesian filtering.[^6]
- **BJEPA (Bayesian JEPA)** in the same work factorizes belief into a learned dynamics expert and a modular prior expert, combined as a Product of Experts; constraints (including physics) can be injected via the prior expert.[^6]

**Core mathematical / architectural innovations**

- In classical JEPA, an encoder maps observations to latent embeddings \(z_t = f(o_t)\), and a predictor \(g_\theta\) learns to regress masked future latents \(\hat z_{t+k} = g_\theta(z_{\le t})\) by minimizing a deterministic regression loss \(\|z_{t+k} - \hat z_{t+k}\|^2\).[^5]
- VJEPA replaces point prediction with a variational family \(q_\theta(z_{t+k} \mid z_{\le t})\) and optimizes a variational objective against a prior \(p(z_{t+k})\), so that the latent belief state \(b_t\) is a predictive distribution over futures, not a single vector.[^6]
- The authors show that, under mild assumptions, the VJEPA latent belief is a sufficient statistic for optimal control (a predictive state representation), without ever reconstructing pixels, and that sampling from \(q_\theta\) enables credible intervals and uncertainty estimates in latent space.[^6]
- BJEPA factors the belief as \(q(z_{t+k}) \propto q_{dyn}(z_{t+k} \mid z_{\le t}) \cdot q_{prior}(z_{t+k})\); the prior can encode hard or soft constraints (e.g., gravitational consistency, contact constraints) and is combined via a Product of Experts energy \(E(z) = E_{dyn}(z) + E_{prior}(z)\).[^6]

**Physical plausibility / energy scores**

- In a JEPA or VJEPA world model trained on handoff videos (real or simulated), each candidate future latent trajectory \(z_{t+1:t+H}\) has an associated *energy* (negative log density or variational free energy) that measures how compatible it is with the model’s learned dynamics and prior constraints.[^5][^6]
- One can define a *physical plausibility score* for an observed short trajectory window \(o_{t-L:t}\) by (i) encoding it into latents, (ii) rolling the predictor forward for \(k\) steps, and (iii) computing the energy gap between the predicted latent and the actually observed latent continuation; large gaps indicate physically implausible or out‑of‑distribution behavior.[^5][^6]
- With BJEPA, additional physics priors (e.g., approximate gravity, non‑penetration, bounded joint velocities) can be encoded as an extra energy term, so that trajectories violating these constraints have sharply higher energy and lower plausibility.[^6]

**Splice into Act‑vs‑Think loop**

1. **World‑model sidecar for Cosmos**
   - Train a compact VJEPA/BJEPA model on egocentric manipulation/handoff datasets (e.g., CrossTask/COIN style procedures and any domain‑specific demos). For your Cookoff system, this can run as a sidecar, taking the same video frames as Cosmos but outputting: current latent \(z_t\), predicted latent \(\hat z_{t+1:t+H}\), and an energy \(E_t\) over the realized short trajectory window.[^7]

2. **Latent plausibility observables**
   - At each frame, compute:
     - \(E_t^{dyn}\): energy under dynamics only (how surprising is the motion pattern?).[^6]
     - \(E_t^{phys}\): additional energy from physics priors (e.g., interpenetration penalties between gripper and hand masks from SAM 2, see Section 3).[^6]
     - \(r_t = \|z_t - \hat z_t\|\): latent prediction error norm.[^5]
   - Normalize these by per‑scenario statistics to obtain dimensionless plausibility scores in \([0,1]\), e.g. \(\pi_t = \sigma(-\alpha E_t^{phys})\).

3. **Textual injection into Cosmos**
   - For each Cosmos agent, prepend to the system message a structured textual summary such as:

     > World‑model diagnostics: dynamics surprise=0.18, physics violation score=0.72, credible interval radius for next‑frame gripper pose=0.35 (high). Overall physical plausibility of current trajectory: low.

   - Cosmos Reason 2 is explicitly designed to consume auxiliary spatial and trajectory information (e.g., 2D/3D points and trajectories) in prompts, so this numeric summary can be combined with natural‑language instructions like "treat low plausibility as a warning that the visual motion may be atypical or unsafe".[^2]

4. **Stopping‑time modification**
   - In a POMDP formulation with belief state \(b_t\), augment \(b_t\) with \((E_t^{dyn}, E_t^{phys}, r_t)\) and define the ACT action as admissible only when
     \[
       E_t^{phys} < \tau_{phys}, \quad r_t < \tau_{pred},
     \]
     for calibrated thresholds \(\tau_{phys},\tau_{pred}\) learned from safe vs failure episodes.
   - The reward function can penalize ACT under high energy with a large negative terminal reward, effectively pushing the optimal policy to THINK when the world model flags physical implausibility.

***

### 2.2 VL‑JEPA: Joint Embedding for Vision–Language

**Key technique**

- **VL‑JEPA** extends JEPA to vision–language: instead of autoregressively generating text tokens, it predicts continuous embeddings of the *target* texts in a shared semantic space, with a lightweight decoder used only when text is actually needed.[^8][^9][^10]
- VL‑JEPA supports open‑vocabulary classification, text–video retrieval, and discriminative VQA directly in embedding space, and shows competitive performance vs CLIP/SigLIP2 and classical VLMs while using half the trainable parameters.[^9]

**Architectural and mathematical aspects**

- Given a video embedding \(z_v\) and an input text embedding \(z_{txt}\), VL‑JEPA learns a predictor that outputs a representation \(\hat z_{ans}\) in the same text space as ground‑truth answers \(z_{ans}\), and minimizes a contrastive or regression loss between them.[^8][^9]
- Because both modalities live in a common JEPA latent, distances \(\|\hat z_{ans} - z_{ans}\|\) and energy‑style scores can be used directly as plausibility metrics over *linguistic* explanations of physical scenes.[^9]

**Splice into Act‑vs‑Think loop**

- Use VL‑JEPA as a *calibration layer* around Cosmos:
  - For each candidate explanation or safety classification from a Cosmos agent (e.g., "The human’s hand is stably under the object; release is safe"), encode it into VL‑JEPA text space and compute its energy relative to a bank of physically validated texts describing the same frame.
  - Large distances indicate that the explanation is semantically atypical given the visual input, enabling a *linguistic plausibility score* that complements the dynamical plausibility \(E_t^{phys}\) from VJEPA.
- Expose this to Cosmos as an additional textual diagnostic ("JEPA‑language plausibility: 0.23; explanation is atypical for this visual pattern"), and treat low plausibility as a trigger to extend THINK (observe more frames) even if softmax confidence on the label is high.

***

### 2.3 GeoWorld: Hyperbolic JEPA and Energy‑Based Planning

**Key technique**

- **GeoWorld: Geometric World Models** introduces a hyperbolic JEPA (H‑JEPA) that maps latent representations onto hyperbolic manifolds and uses an energy‑based value function to perform multi‑step planning by minimizing energy along geodesics in latent space.[^7]
- A geometric reinforcement learning (GRL) procedure refines the predictor to produce geodesic‑consistent rollouts, improving long‑horizon stability and planning success by 2–3% over V‑JEPA baselines on procedural datasets.[^7]

**Core innovations**

- Latent states live on a hyperbolic manifold \(\mathbb{H}^d\) with metric \(g\), where distances better capture hierarchical and compositional structure of tasks and object interactions.[^7]
- An energy function \(E(z_t, a_{t:t+H})\) encodes cumulative cost; low energy corresponds to plausible, high‑reward futures, and planning reduces to minimizing \(E\) along geodesic paths using methods like CEM.[^7]
- Triangle‑inequality regularization and geodesic consistency constraints enforce smooth latent trajectories and discourage physically impossible jumps.[^7]

**Splice into Act‑vs‑Think loop**

- For handoff, define a small discrete action set for the world model (e.g., maintain grasp, micro‑adjust pose, release now), even if the real robot is controlled by a separate low‑level policy.
- Use GeoWorld as a *latent critic* over short candidate futures: starting from current \(z_t\), roll out short geodesic trajectories that include "release" at different future times, compute energy for each, and treat the minimum energy over "release" futures as a scalar release‑plausibility score.[^7]
- If all release futures have high energy compared to "hold" futures, the Act‑vs‑Think policy should continue THINK; only when a low‑energy release trajectory emerges does ACT become epistemically justified.
- This critic can be summarized textually for Cosmos as "Geometric world model energy: releasing now has 3.5× higher latent energy than holding for two more frames; recommend waiting" and fed to each agent’s prompt as context.

***

### 2.4 Physically Grounded Evaluation and Invariant Violations

**Key works**

- **Physics‑RW**: a real‑world video‑language benchmark covering mechanics, thermodynamics, electromagnetism, and optics, used to probe whether general world models can infer physical phenomena.[^11]
- **How Far is Video Generation from World Model: A Physical Law Perspective** shows that visually plausible video generation models can systematically violate physical invariants (e.g., objects passing through obstacles, inconsistent lighting) and that visual realism is not a reliable indicator of physical correctness.[^12]
- **From Generative Engines to Actionable Simulators** argues that world models must encode causal structure and domain‑specific constraints, and proposes a reframing toward constraint‑aware, closed‑loop simulators evaluated by their ability to respect invariants under intervention rather than visual fidelity.[^13]

**Implication for "proving" physics hallucinations**

- These works suggest a *constraint‑based* notion of physical hallucination: a model is hallucinating physics when its predicted or explained trajectories systematically violate invariants (e.g., non‑penetration, approximate energy/momentum bounds, monotone gravity) that hold in the data distribution.[^11][^13][^12]
- One can formalize this as a hypothesis test over a sequence of predicted world states \(s_{t:t+H}\): under \(H_0\) (physically consistent), violations of a constraint set \(C\) should be rare and bounded; under \(H_1\) (hallucinating), the expected violation measure \(\mathbb{E}[v(s_{1:T})]\) exceeds a calibrated threshold with high probability.[^13][^12]

**Splice into Act‑vs‑Think loop**

- Derive a small set of task‑specific constraints for handoff: e.g., no interpenetration of object with human hand, bounded vertical acceleration of the object, non‑increasing potential energy after release except due to bounce, etc.
- Use the JEPA/GeoWorld latent rollouts to approximate object and hand trajectories, map them back to approximate 3D or 2D kinematics via a simple decoder or pose‑estimator, and compute violation statistics per frame window.
- Expose to Cosmos a binary "physics OK / violation" flag and a scalar violation margin; if the model’s *linguistic* explanation claims safety while latent‑space constraints are violated with high confidence, label this as a *physics hallucination event* and force THINK plus an epistemic penalty in the ensemble aggregation (see Section 4).

***

## 3. SAM 2 and Spatio‑Temporal Physical Constraints

### 3.1 SAM 2: Streaming Memory and Zero‑Shot Video Masks

**Key technique**

- **Segment Anything Model 2 (SAM 2)** is a unified foundation model for image and video segmentation with a streaming memory module that stores object‑specific information over time, enabling fast, interactive tracking from a single prompt (point/box/mask) across an entire video.[^14][^15][^16][^17]
- SAM 2 achieves state‑of‑the‑art video object segmentation with 3× fewer human interactions than prior methods and runs in real time on standard hardware.[^16][^17][^18]

**Architectural details relevant for constraints**

- A frame encoder produces embeddings that are conditioned on both the current frame and past predictions/prompted frames stored in a session‑level memory, enabling long‑range temporal coherence.[^17][^19][^14]
- Promptable heads output one or multiple masks per frame, with losses combining focal, dice, IoU, and object‑presence/occlusion heads to handle frames without visible masks.[^19][^17]

**Splice into Act‑vs‑Think loop**

1. **Object‑centric tracks for handoff entities**
   - Initialize SAM 2 with prompts for: robot gripper, object, human hand, and optionally torso/face for collision constraints.
   - Run SAM 2 in streaming mode over the handoff video to obtain per‑frame binary masks \(M_t^{grip}, M_t^{obj}, M_t^{hand}\) with associated confidence and occlusion flags.[^14][^16][^17]

2. **Derive geometric physical constraints**
   - For each frame, compute geometric features:
     - Centroids, principal axes, and extents for each mask.
     - Pairwise distances between centroids and minimum distance between mask boundaries (approximate contact gaps).
     - Overlap fractions \(\text{IoU}(M_t^{obj}, M_t^{hand})\), \(\text{IoU}(M_t^{obj}, M_t^{grip})\).
   - Differentiate centroids over time to approximate velocities and accelerations of the object relative to hand and gripper.
   - These features define *semantic physical constraints*, e.g.:
     - Non‑penetration: \(\text{IoU}(M_t^{obj}, M_t^{hand})\) should increase smoothly before contact, not jump from 0 to a large value; similarly for gripper–object.[^17]
     - Support: release is unsafe if the object’s vertical centroid is above the hand’s support centroid and downward velocity exceeds a threshold.

3. **Textual constraint summaries for Cosmos**
   - Compress these features into a short natural‑language diagnostic inserted into each agent’s context:

     > Segmentation–derived constraints: object–gripper IoU=0.82, object–hand IoU=0.12, vertical gap=3.1 cm, relative vertical velocity (object vs hand) = −0.05 m/s, occlusion=low. Current contact graph suggests the hand is *not yet* stably supporting the object.

   - Optionally, include structured JSON‑like text that Cosmos can parse (e.g., key–value pairs) in addition to prose.

4. **Constraint‑augmented stopping rule**
   - ACT is permitted only when both world‑model plausibility (Section 2) and SAM‑derived contact constraints indicate safety, e.g.:
     - Hand–object IoU \(> \tau_{support}\).
     - Object–gripper IoU decreasing while hand–object IoU increasing over a fixed window.
     - Estimated acceleration after hypothetical release (using finite differences) below a safe bound.
   - Violations of these constraints increment an epistemic cost and push the ensemble toward THINK, even if individual agents express high verbal confidence.

***

### 3.2 SAM2Grasp: Temporal Prompting for Grasping and Handoffs

**Key technique**

- **SAM2Grasp** uses a frozen SAM 2 as a temporal‑aware perception backbone and attaches a lightweight Action Chunking with Transformers (ACT) head that predicts future action chunks conditioned on object‑centric features and robot proprioception.[^20][^21][^22][^23]
- An initial prompt (e.g., a bounding box) selects a specific object; SAM 2’s internal memory and tracking then provide a sequence of object‑aware features \(F_t\), while ACT predicts a chunk of future actions \(A_t = [a_t, \dots, a_{t+K-1}]\) using a regression loss.[^21][^20]
- The framework achieves 87.8% success in simulated multi‑object grasping and 97% success in real cluttered bin picking, with remarkable robustness under occlusions (e.g., maintaining 77% success under 40% occlusion where baselines collapse).[^20][^21]

**Architectural details**

- SAM 2 is frozen and used purely as a temporal feature extractor, decoupling perception from control and avoiding catastrophic forgetting.[^21][^20]
- An asynchronous execution strategy runs policy inference and robot control threads at different frequencies and ensembles overlapping action chunk predictions to produce a final control command.[^20]

**Splice into Act‑vs‑Think loop**

- Even if the Cookoff entry does not directly use SAM2Grasp for low‑level control, its architecture suggests a design for *action‑conditioned constraint prediction*:
  - Use SAM 2 to extract object–hand–gripper features \(F_t\) as above.
  - Train a small temporal head (not necessarily full ACT) to predict *counterfactual* feature evolution under the ACT vs THINK choices, using recorded trajectories (e.g., what happens if release occurs now vs 3 frames later).
  - For each frame, infer short‑horizon rollouts of contact features and estimate whether a release now would violate safety constraints (e.g., object falling outside the hand, excessive relative velocity).
- Summarize the result as a scalar "counterfactual release risk" and textual message:

  > If released now, object–hand IoU is predicted to drop to 0.02 in 200 ms with high confidence; predicted failure probability estimated at 0.91 based on SAM‑derived futures.

- Feed this into Cosmos and the ensemble’s stopping rule; THINK is chosen when counterfactual risk is above a threshold, even if the current frame appears visually safe.

***

### 3.3 SAM2Plus and Physics‑Aware Tracking

**Key technique**

- **SAM2Plus** augments SAM 2 with a Kalman filter and dynamic thresholds to improve long‑term tracking under fast motion, occlusion, and crowded environments.[^24][^25]
- The Kalman filter models object motion using physical constraints (e.g., approximate constant‑velocity or constant‑acceleration models) to predict trajectories and refine segmentation states, reducing positional drift and tracking instability.[^25][^24]

**Splice into Act‑vs‑Think loop**

- In a handoff video, SAM2Plus‑style tracking can be adapted as follows:
  - Fit a simple linear dynamical model to centroids and principal axes of object/hand masks, yielding state estimates \(x_t\) and covariances \(P_t\).
  - Define an innovation term \(\nu_t\) between predicted and observed positions; large \(\|\nu_t\|\) indicates either abrupt motion (potentially unsafe) or segmentation failure.
- Use innovation magnitude and covariance traces as additional observables:
  - Large innovations, especially when combined with high world‑model energy, indicate that the current frame belongs to an out‑of‑distribution dynamic regime; this should consume more of the epistemic budget and bias toward THINK.
  - These values can be summarized textually for Cosmos ("tracking innovation norm=0.19 m (2.7× nominal), suggesting unexpected acceleration").

***

## 4. Epistemic Uncertainty in VLMs and Physics Hallucinations

### 4.1 ViLU: Vision–Language Uncertainties for Failure Prediction

**Key technique**

- **ViLU (Vision‑Language Uncertainties)** learns a post‑hoc uncertainty head operating on vision and text embeddings to predict whether a VLM’s output is correct or incorrect.[^26][^27][^28][^29]
- ViLU constructs an uncertainty‑aware multi‑modal representation by integrating the visual embedding \(z_v\), the predicted textual embedding \(z_{txt}\), and an image‑conditioned textual representation via cross‑attention, then trains a binary classifier with weighted cross‑entropy to distinguish correct vs incorrect predictions.[^28][^26]
- Crucially, ViLU is loss‑agnostic and can operate in post‑hoc mode where only embeddings are available, without access to internal logits or training data, making it practical for wrapped foundation models.[^27][^26]

**Mathematical formulation**

- Let \(f_v\) be the vision encoder and \(f_t\) the text encoder of a base VLM; given an image–text pair, ViLU computes \(z_v = f_v(x)\), \(z_{txt} = f_t(y)\) for the model’s output \(y\), and an image‑conditioned text representation \(z_{cond}\) via cross‑attention mechanisms.[^26][^28]
- Concatenating or fusing these into \(h = \phi(z_v, z_{txt}, z_{cond})\), ViLU trains a classifier \(u_\theta(h)\) with output \(p_\theta(c=1 \mid h)\) indicating correctness, using a weighted binary cross‑entropy loss to handle imbalance between correct and incorrect samples.[^28][^26]
- The output \(u_\theta(h)\) can be interpreted as an *epistemic failure probability* conditioned on the embedding geometry, distinct from softmax confidence on labels.[^27][^26]

**Splice into Act‑vs‑Think loop**

- For each Cosmos agent’s answer about release safety, capture the vision and text embeddings (either via Cosmos hooks or a parallel CLIP‑like encoder) and feed them to a ViLU‑style uncertainty head trained on labeled safe/failure decisions for handoff episodes.
- Use the predicted failure probability \(p_\theta(c=0 \mid h)\) as:
  - A scalar *epistemic budget consumption rate* (higher failure probability consumes more budget and biases toward THINK).
  - A per‑agent reliability weight in ensemble aggregation (agents with high predicted failure probability are down‑weighted when aggregating ACT/THINK votes).
- Expose to Cosmos itself a brief diagnostic ("post‑hoc failure probability: 0.74; treat this answer as highly uncertain") so that chain‑of‑thought reasoning can explicitly verbalize its epistemic state if desired.

***

### 4.2 VL‑Uncertainty: Perturbation‑Based Hallucination Detection

**Key technique**

- **VL‑Uncertainty** detects hallucinations in large VLMs by measuring uncertainty as prediction variance across semantically equivalent but perturbed prompts (visual and textual).[^30][^31][^32]
- For a given image and base prompt, the method generates multiple perturbed prompts (e.g., rephrasings, masked regions) and clusters the resulting answers by semantic similarity; the entropy of the cluster distribution serves as an intrinsic uncertainty measure.[^32][^30]
- High entropy correlates strongly with hallucinations across 10 LVLMs and four benchmarks, outperforming baseline methods that rely on pseudo‑labels or external grounding.[^32]

**Mathematical view**

- Let \(P = \{p_i\}_{i=1}^N\) be semantically equivalent prompts and \(A = \{a_i\}_{i=1}^N\) the corresponding model answers; cluster \(A\) into \(K\) semantic clusters with probabilities \(q_k\).[^32]
- Define uncertainty as \(U = H(q) = -\sum_k q_k \log q_k\); high \(U\) indicates unstable semantics given stable intent, a hallmark of epistemic uncertainty.[^32]

**Splice into Act‑vs‑Think loop**

- For each frame (or short clip), sample a small set of paraphrased questions to each Cosmos agent (e.g., 3–4 variants asking whether release is safe). Use fast sampling or constrained decoding to limit overhead.
- Cluster the resulting answers via an embedding space (e.g., sentence embeddings) and compute entropy; if entropy exceeds a calibrated threshold but softmax confidence (or logprob of the top answer) is high, flag this as a *consensus hallucination risk*.
- This flag should: (i) push the POMDP to THINK, (ii) increment an adversarial loss for that agent in offline training (rewarding consistency under paraphrases when correct and diversity when incorrect), and (iii) feed into ensemble aggregation as an epistemic penalty.

***

### 4.3 Dropout Decoding and Epistemic Decomposition of Visual Tokens

**Key technique**

- **Dropout Decoding** introduces an inference‑time procedure that quantifies visual‑token uncertainty by projecting them into text space and decomposing uncertainty into aleatoric and epistemic components, then performing uncertainty‑guided token dropout to reduce hallucinations.[^33][^34]
- The method treats uncertain visual tokens analogously to dropout: it constructs multiple masked decoding contexts where high‑uncertainty tokens are stochastically dropped and aggregates predictions to robustly mitigate misinterpretations.[^34][^33]

**Mathematical aspects**

- Visual tokens \(v_j\) from the vision encoder are projected into the text space and their contribution to the output distribution is analyzed; uncertainty for each token is quantified via decomposition into aleatoric (data ambiguity) and epistemic (model uncertainty) parts.[^33]
- Epistemic uncertainty is emphasized as more relevant to perception errors; high‑epistemic tokens are prime candidates for dropout, and the ensemble of masked decodings yields calibrated confidence estimates.[^33]

**Splice into Act‑vs‑Think loop**

- For each Cosmos frame‑level query, approximate Dropout Decoding by:
  - Identifying high‑sensitivity visual tokens (e.g., via gradient norms or small adversarial perturbations) and marking them as potentially epistemic.[^33]
  - Running a small ensemble of decodings where subsets of such tokens are masked, and measuring variance or entropy over safety decisions.
- If ACT decisions depend strongly on a small set of highly epistemic visual tokens, treat this as a brittle regime; in the ensemble, these decisions should trigger THINK or prompt re‑observation until those regions become less ambiguous (e.g., occlusion clears, hand–object relation becomes clearer).

***

### 4.4 Epistemic Uncertainty of Visual Tokens and Adversarial Probes

**Key technique**

- **On Epistemic Uncertainty of Visual Tokens for Object Hallucinations** shows that visual tokens with high epistemic uncertainty in the vision encoder correlate strongly with hallucinated objects and that such tokens can be identified via representation deviations under small adversarial perturbations.[^35][^36][^37]
- The authors prove that, under small perturbations and smoothness assumptions, perturbed hidden states can be approximated locally by a Gaussian, and that large deviations in early encoder layers signal epistemic uncertainty.[^37]
- A practical method is proposed to efficiently approximate uncertain tokens and then mask them in intermediate self‑attention, significantly reducing hallucinations while being compatible with other mitigation techniques.[^36][^37]

**Splice into Act‑vs‑Think loop**

- For each handoff frame, perform a *lightweight adversarial probe* on the image (e.g., small bounded perturbations in regions around the hand/object), recompute early‑layer visual tokens, and measure representation deviation norms.
- Use the aggregate deviation as a *physics hallucination prior*: if small perturbations drastically change the token representations in regions critical for contact/support reasoning, the epistemic uncertainty is high and the system should be unwilling to ACT on that frame.
- Summarize this as a scalar "visual epistemic instability" and optionally as a text snippet to Cosmos ("small perturbations to hand region produce large latent shifts; visual evidence for support is unreliable").

***

### 4.5 Benchmarks and Taxonomies of Multimodal Epistemic vs Aleatoric Uncertainty

**Key works**

- An ICLR 2025 work proposes a benchmark and metric for multimodal epistemic and aleatoric uncertainty, offering a taxonomy of uncertainty types specific to vision–language systems and fine‑grained categories within epistemic and aleatoric uncertainty.[^38]
- A TMLR submission studies whether VLMs are robust to classic uncertainty challenges (anomaly detection, ambiguous classification), showing that modern VLMs can abstain effectively when asked to output "Unknown" but still tend to hallucinate confident responses in domain‑specific tasks without specialized knowledge.[^39]
- A "tiny" ICLR paper shows that VLMs can implicitly quantify aleatoric uncertainty by being prompted to output "Unknown" on ambiguous inputs, without additional training; this underscores the gap between aleatoric and epistemic uncertainty in current models.[^40]

**Implications for your ensemble**

- These works support a **two‑axis epistemic budget**: one axis measuring aleatoric ambiguity (e.g., motion blur, occlusion, ambiguous camera angle) and another measuring epistemic uncertainty (e.g., OOD dynamics, unmodeled contact patterns).
- The budget can be explicitly split: THINK actions focused on reducing aleatoric uncertainty (waiting for better views) vs THINK actions focused on epistemic uncertainty (probing with adversarial perturbations, cross‑agent disagreement tests, or consulting ArchiveKV for similar past episodes).

***

### 4.6 Toward a Mathematical Notion of "Hallucinating Physics"

Combining the above, a practical and semi‑formal criterion that a VLM is "hallucinating physics" can be:

1. **Invariant‑violation criterion**
   - Define a constraint set \(C\) representing approximate physical laws relevant to handoff (e.g., non‑penetration, bounded accelerations) instantiated via JEPA/GeoWorld latent rollouts and SAM‑derived kinematics (Sections 2–3).[^12][^13][^17][^7][^6]
   - If the VLM’s explanation asserts safety while the estimated violation measure \(v(s_{t:t+H})\) (e.g., penetration depth, energy non‑conservation) exceeds a calibrated bound in repeated simulations, label the explanation as violating physics with high probability.[^13][^12]

2. **Epistemic‑instability criterion**
   - Use ViLU’s estimated failure probability \(p_\theta(c=0 \mid h)\), VL‑Uncertainty’s entropy, and adversarial token instability (Section 4.4) as orthogonal estimates of epistemic uncertainty.[^36][^37][^26][^27][^32]
   - If these metrics are all high while the VLM’s own verbal confidence or softmax margin is high, there is strong evidence of *overconfident epistemic error*, a hallmark of hallucination.

3. **Statistical test**
   - Over a dataset of handoff episodes with ground‑truth safety labels, compute the joint distribution of (constraint violations, epistemic metrics, verbal confidence). If ACT decisions in regions of high violation + high epistemic uncertainty + high verbal confidence have significantly elevated failure rates, then one can statistically reject the null that "high confidence implies physical correctness" for that VLM under your deployment distribution.

In your system, these composite criteria become guardrails: whenever they fire, the ensemble is *prohibited* from ACT and may be forced to either THINK further or escalate to human supervision.

***

## 5. Multi‑Agent Asymmetry and Avoiding Consensus Collapse

### 5.1 Failure Modes of Naïve Multi‑Agent Ensembles

**Key observations from recent work**

- Analyses of "consensus hallucination" show that when multiple correlated LLMs are ensembled with majority voting, they can agree on the same wrong answer due to shared priors, similar decoding strategies, and correlated errors, so majority voting can amplify rather than correct hallucinations.[^41]
- Empirical audits find that the performance gain from ensembling scales like \(p^n\) only under independence; correlations effectively reduce ensemble size and can leave failure probability almost unchanged even with many agents.[^41]
- A 2025 preprint on LLM interactions observes that two independent LLMs in unsupervised multi‑turn conversation often converge to repetitive, low‑diversity loops within tens of turns, even without shared memory or explicit coordination, highlighting an intrinsic tendency toward *herding* in long‑horizon interactions.[^42]
- Multi‑Agent Debate (MAD) studies find that naive debate protocols do not reliably outperform self‑consistency or simple ensembles; performance depends strongly on hyperparameters and how much agents are encouraged to agree vs disagree.[^43][^44][^45][^46]

These findings argue against a symmetric Cosmos ensemble with identical prompts, memories, and decoding, as such a design is prone to consensus hallucination and collapse.

***

### 5.2 Consensus–Diversity Trade‑off and Complementary Agents

**Key works**

- **Unraveling the Consensus–Diversity Tradeoff in Adaptive Multi‑Agent LLM Systems** proposes a framework for measuring behavioral alignment and diversity, showing that moderate, role‑driven deviations from uniformity improve performance and robustness on complex problems.[^47]
- The authors highlight "transient diversity"—delayed convergence after an initial phase of diverse hypotheses—as particularly beneficial.[^47]
- **Mixture of Complementary Agents (complementary‑MoA)** frames agent selection as optimizing *complementarity*, not just accuracy or diversity in isolation, and introduces greedy algorithms that pick proposers whose joint performance with a summarizer is maximized.[^48]

**Implications**

- Diversity should be *structured* (role‑ or prior‑driven), not random; agents should be biased to cover different epistemic failure modes (e.g., over‑ vs under‑cautious release decisions), and convergence should be delayed until enough evidence accumulates.[^48][^47]

***

### 5.3 Asymmetric Self‑Play, Information Asymmetry, and Diversity Preservation

**Key works**

- **Asymmetric self‑play** originates from Sukhbaatar et al. and has recently been extended to LLMs: different roles (e.g., task creator vs solver, attacker vs defender) with non‑identical objectives interact in a shared environment, producing curricula and robust policies.[^49]
- R‑Diverse shows that self‑play training of reasoning LLMs can suffer from a "diversity illusion" where superficially different prompts mask structurally similar reasoning; a memory‑augmented penalty that discourages repeated solution patterns can restore genuine diversity and improve generalization.[^50]
- **AsymPuzl** introduces a controlled two‑agent environment with strict information asymmetry, showing that communication patterns and coordination quality depend strongly on role‑specific views and feedback, not just model capacity.[^51][^52]
- Multi‑agent sentiment‑analysis architectures have used **asymmetric LoRA initialization** (e.g., perception vs reasoning agent with different sparse masks) to improve robustness and handle nuanced phenomena like sarcasm.[^53]

**Implications for your ensemble**

- Asymmetry in *information sets*, *parametrization*, and *training objectives* is key to preventing collapse into a single effective agent.
- Memory mechanisms (like your LiveKV/ArchiveKV) should be used to *penalize* repeated reasoning trajectories across agents and time, not just to share evidence.[^50]

***

### 5.4 Concrete Asymmetry Mechanisms for the Adversarial Blind Epistemic Ensemble

The following mechanisms map the above literature to Cosmos‑based agents in your Cookoff system.

#### (A) Parametric Bias via LoRA Heads and Training Data

- Attach distinct low‑rank adaptation (LoRA) modules or lightweight finetuning heads to each Cosmos agent, initialized and trained with biased datasets or objectives:
  - **Agent R (Risk‑Averse)**: overweights near‑miss failure episodes where early release caused problems; trained to maximize safety recall at the expense of false negatives.
  - **Agent E (Efficiency‑Seeking)**: trained to minimize average handoff time under a safety constraint, thus more willing to ACT early when evidence seems strong.
  - **Agent P (Physics‑Skeptic)**: co‑trained with JEPA/GeoWorld energy cues and SAM‑derived constraints, penalized heavily for agreeing with ACT decisions when physical plausibility scores are low.
  - **Agent A (Adversarial)**: trained to search for counter‑examples and alternative interpretations that could make ACT unsafe; exposed to perturbations and OOD scenarios.
- This mirrors complementary‑MoA’s idea of agents with complementary strengths and error profiles.[^48]

#### (B) Temporal and Memory Masking

- **Temporal masking**: at frame \(t\), give each agent a different temporal slice of the history:
  - Agent R sees frames \(0\) to \(t\) in full (long‑horizon cautious perspective).
  - Agent E sees only the recent \(t-k\) to \(t\) window, emphasizing immediate cues.
  - Agent P sees sub‑sampled frames and explicit world‑model rollouts (latent or decoded), but not raw video, forcing reliance on physics summaries.
  - Agent A sees counterfactual rollouts (what if release had occurred 3 frames ago?) and is asked explicitly whether any plausible failure trajectory still exists.
- **LiveKV masking**: restrict LiveKV so that not all intermediate rationales are visible to all agents at all times; for example, allow only numeric diagnostics (plausibility scores, constraint margins) but not other agents’ binary ACT/THINK decisions until after each agent has issued its own vote.
- This enforces information asymmetry along the lines of AsymPuzl and reduces premature herding.[^52][^51]

#### (C) Diversity‑Aware Aggregation and Penalties

- Maintain a memory of recent reasoning traces in ArchiveKV and measure similarity across agents’ chain‑of‑thought and final justifications.
- Inspired by R‑Diverse, introduce a diversity penalty when multiple agents repeatedly follow nearly identical reasoning paths and ACT in high‑risk regions; this penalty can be used both in offline RLHF‑style training and online as an epistemic cost that pushes the POMDP to THINK.[^50]
- Aggregate decisions with weights that depend on both ViLU‑style reliability and diversity score: e.g., an ACT decision supported only by agents with highly correlated rationales is discounted relative to one where independent rationales converge.

#### (D) Adversarial Blindness and Cross‑Checking

- **Adversarial blindness**: when evaluating a given frame, mask different subsets of diagnostic signals for different agents (e.g., hide world‑model energy from one, hide SAM contact features from another) and observe whether ACT decisions remain consistent.
- If an agent ACTs when blinded to a diagnostic that strongly argues against safety (according to other agents), mark this as a potential over‑reliance on a narrow cue and down‑weight that agent.
- Cross‑checking between physics‑aware and physics‑agnostic agents can reveal instances where language‑level confidence is not backed by physical plausibility, further supporting physics hallucination detection.

***

## 6. Integrating Everything into an Act‑vs‑Think Stopping‑Time POMDP

### 6.1 POMDP Augmentation

Define the POMDP state as a factored tuple \(s_t = (x_t^{video}, x_t^{wm}, x_t^{sam}, x_t^{epi})\), where:

- \(x_t^{video}\): raw or tokenized video observation (already consumed by Cosmos).[^2]
- \(x_t^{wm}\): world‑model latents and energies from VJEPA/BJEPA/GeoWorld: \(z_t\), \(E_t^{dyn}\), \(E_t^{phys}\), geodesic release energies.[^5][^7][^6]
- \(x_t^{sam}\): SAM 2–derived geometric/contact features and tracking innovations (including Kalman residuals).[^24][^16][^25][^14][^17]
- \(x_t^{epi}\): epistemic/aleatoric uncertainty features from ViLU, VL‑Uncertainty, Dropout Decoding, adversarial token instability, and ensemble‑level diversity metrics.[^37][^26][^27][^36][^47][^48][^50][^33][^32]

Actions are \(a_t \in \{\text{THINK}, \text{ACT}\}\), with rewards:

- \(R(\text{ACT}, s_t) = R_{success}\) if release is actually safe, \(R_{failure} \ll 0\) otherwise.
- \(R(\text{THINK}, s_t) = -c\), a per‑frame cost for waiting.

Transition dynamics encode how additional frames update \(x_t^{wm}, x_t^{sam}, x_t^{epi}\); these can be learned via RL or approximated with heuristic thresholds.

### 6.2 Stopping Rule and Epistemic Budget

A practical implementation is a *composite threshold rule* that approximates an optimal stopping time:

- Define scalar scores:
  - Physical plausibility \(\pi_t\) from world models.
  - Contact safety margin \(\sigma_t\) from SAM 2.
  - Epistemic failure probability \(\rho_t\) from ViLU and related methods.
  - Ensemble diversity score \(\delta_t\) from multi‑agent asymmetry mechanisms.
- Define an ACT‑admissible region \(\mathcal{A}\) in this 4D space, e.g.:
  \[
    \mathcal{A} = \{ (\pi, \sigma, \rho, \delta) : \pi > \tau_\pi,\ \sigma > \tau_\sigma,\ \rho < \tau_\rho,\ \delta > \tau_\delta \},
  \]
  where \(\delta\) is *higher* when multiple agents agree from diverse rationales.[^21][^26][^47][^48][^32][^7]
- The epistemic budget can be formalized as a constraint on the expected cumulative epistemic cost \(\sum_t e_t\), where \(e_t\) is a function of \(\rho_t\), VL‑Uncertainty entropy, and adversarial token instability; THINK actions are allowed until this budget is exhausted or \(s_t\) enters \(\mathcal{A}\), whichever comes first.

### 6.3 Cosmos‑Compatible Interface

At implementation time, each agent’s Cosmos prompt at frame \(t\) can take the form:

> System: You are Agent P, specialized in physics‑aware safety reasoning for human–robot object handoff. You see video frames up to time t. You must decide whether it is safe for the robot to release the object now or whether it should wait and observe more.
>
> Additional diagnostics:
> - World‑model dynamics surprise: 0.12; physics violation energy: 0.76; low plausibility.
> - Segmentation constraints: object–gripper IoU 0.69 decreasing; object–hand IoU 0.18 increasing; vertical gap 2.7 cm; predicted post‑release drop 4.5 cm.
> - Post‑hoc failure probability (ViLU): 0.63.
> - Paraphrase entropy (VL‑Uncertainty): high.
> - Ensemble diversity: medium.
>
> You must output either THINK (if more observation is needed) or ACT (if release is safe), plus a concise justification referencing these diagnostics.

Different agents receive different subsets of diagnostics and temporal context per Section 5, implementing adversarial blindness and asymmetry.

A thin orchestration layer then:

- Converts diagnostics into textual snippets.
- Calls the four Cosmos‑Reason2 agents in parallel.
- Aggregates ACT/THINK votes with diversity‑ and reliability‑aware weighting.
- Updates the epistemic budget and decides whether to proceed or observe further.

***

## 7. Summary of Key Integration Opportunities

The table below summarizes the main techniques and their primary roles in the ensemble.

| Vector | Technique | Role in System |
|-------|-----------|----------------|
| Latent dynamics | VJEPA/BJEPA, V‑JEPA, GeoWorld[^6][^5][^7] | Provide latent physical plausibility and release‑energy scores; enable constraint injection and geodesic planning. |
| Segmentation | SAM 2, SAM2Grasp, SAM2Plus[^14][^16][^20][^21][^24] | Convert masks into kinematic/contact constraints and tracking innovation signals; drive counterfactual release risk. |
| Epistemic UQ | ViLU, VL‑Uncertainty, Dropout Decoding, visual‑token epistemic analysis[^26][^32][^33][^36][^37] | Estimate true epistemic failure probability, paraphrase instability, and token‑level visual uncertainty; detect physics hallucinations. |
| Multi‑agent asymmetry | Consensus‑diversity framework, complementary‑MoA, asymmetric self‑play, R‑Diverse, AsymPuzl[^47][^48][^49][^50][^51] | Design parametric and informational asymmetry across agents; penalize herding; weight agents by diversity and reliability. |

Together, these components instantiate an Adversarial Blind Epistemic Ensemble that couples physically grounded world models, segmentation‑based constraints, principled uncertainty quantification, and asymmetric multi‑agent design into a coherent Act‑vs‑Think stopping‑time mechanism for robotic handoff safety.

---

## References

1. [Maximize Robotics Performance by Post-Training NVIDIA Cosmos Reason](https://developer.nvidia.com/blog/maximize-robotics-performance-by-post-training-nvidia-cosmos-reason/) - First unveiled at NVIDIA GTC 2025, NVIDIA Cosmos Reason is an open and fully customizable reasoning ...

2. [NVIDIA Cosmos Reason 2 Brings Advanced Reasoning To Physical AI](https://huggingface.co/blog/nvidia/nvidia-cosmos-reason-2-brings-advanced-reasoning) - A Blog post by NVIDIA on Hugging Face

3. [NVIDIA Cosmos Cookoff - Luma](https://luma.com/nvidia-cosmos-cookoff) - Host: NVIDIA Sponsors: Nebius and Milestone Systems Community: Discord Prizes:First Place: $3,000 an...

4. [Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/index.html)

5. [V-JEPA: The next step toward advanced machine intelligence](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) - We’re releasing the Video Joint Embedding Predictive Architecture (V-JEPA) model, a crucial step in ...

6. [VJEPA: Variational Joint Embedding Predictive Architectures as ...](https://arxiv.org/abs/2601.14354) - Joint Embedding Predictive Architectures (JEPA) offer a scalable paradigm for self-supervised learni...

7. [GeoWorld: Geometric World Models - arXiv](https://arxiv.org/html/2602.23058v1) - Energy-based predictive world models provide a powerful approach for multi-step visual planning by r...

8. [VL-JEPA: Joint Embedding Predictive Architecture for Vision-language](https://openreview.net/forum?id=tjimrqc2BU) - TL;DR: We introduce a vision-language model based on JEPA, that achieves competitive socres while be...

9. [VL-JEPA: Joint Embedding Predictive Architecture for Vision-language](https://arxiv.org/abs/2512.10942) - We introduce VL-JEPA, a vision-language model built on a Joint Embedding Predictive Architecture (JE...

10. [VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language. Vision Language Models (VLMs)](https://www.youtube.com/watch?v=D2il7SDF0Hc) - VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language

The podcast provides the techn...

11. [A benchmark for physical reasoning in general world models with ...](https://www.sciencedirect.com/science/article/abs/pii/S0957417425001708) - A physical reasoning benchmark covering four physical phenomena. The first physical reasoning benchm...

12. [How Far is Video Generation from World Model: A Physical Law ...](https://phyworld.github.io) - We conduct a systematic study to investigate whether video generation is able to learn physical laws...

13. [From Generative Engines to Actionable Simulators: The Imperative of Physical Grounding in World Models](https://web3.arxiv.org/pdf/2601.15533)

14. [Segment Anything 2 (SAM 2)](https://ai.meta.com/sam2/) - SAM 2 is a segmentation model that enables fast, precise selection of any object in any video or ima...

15. [SAM 2: Segment Anything in Images and Videos - arXiv](https://arxiv.org/html/2408.00714v1)

16. [SAM 2: Segment Anything in Images and Videos](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/) - We present Segment Anything Model 2 (SAM 2 ), a foundation model towards solving promptable visual s...

17. [Segment Anything Model 2 (SAM 2) & SA-V Dataset from Meta AI](https://encord.com/blog/segment-anything-model-2-sam-2/) - Meta AI has released Segment Anything Model 2 (SAM 2), a groundbreaking new foundation model designe...

18. [SAM 2: Meta's Next-Gen Model for Video and Image Segmentation](https://www.digitalocean.com/community/tutorials/sam-2-metas-next-gen-model-for-video-and-image-segmentation) - # Clone the repo !git clone https://github.com/facebookresearch/segment-anything-2.git # Move to the...

19. [Published as a conference paper at ICLR 2025](https://openreview.net/pdf/7c41968163abe4e3700e3e3a15174a9d679fcd52.pdf)

20. [Resolve Multi-modal Grasping via Prompt-conditioned ...](https://www.themoonlight.io/en/review/sam2grasp-resolve-multi-modal-grasping-via-prompt-conditioned-temporal-action-prediction) - The paper introduces SAM2Grasp, a novel framework designed to resolve the pervasive multi-modality p...

21. [SAM2Grasp: Resolve Multi-modal Grasping via Prompt-conditioned ...](https://arxiv.org/abs/2512.02609) - Our method leverages the frozen SAM2 model to use its powerful visual temporal tracking capability a...

22. [SAM2Grasp: Resolve Multi-modal Grasping via Prompt- ...](https://www.arxiv.org/pdf/2512.02609.pdf)

23. [[PDF] SAM2Grasp: Resolve Multi-modal Grasping via Prompt-conditioned ...](https://arxiv.org/pdf/2512.02609.pdf)

24. [Improvement of SAM2 Algorithm Based on Kalman ...](https://pubmed.ncbi.nlm.nih.gov/40648454/) - The Segment Anything Model 2 (SAM2) has achieved state-of-the-art performance in pixel-level object ...

25. [Improvement of SAM2 Algorithm Based on Kalman Filtering for Long ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC12252479/) - The Kalman filter models object motion using physical constraints ... Despite these advancements, SA...

26. [ViLU: Learning Vision-Language Uncertainties for Failure Prediction](https://arxiv.org/abs/2507.07620v2) - Reliable Uncertainty Quantification (UQ) and failure prediction remain open challenges for Vision-La...

27. [ViLU - ICCV 2025 Open Access Repository](https://openaccess.thecvf.com/content/ICCV2025/html/Lafon_ViLU_Learning_Vision-Language_Uncertainties_for_Failure_Prediction_ICCV_2025_paper.html)

28. [ViLU: Learning Vision-Language Uncertainties for Failure ...](https://arxiv.org/pdf/2507.07620.pdf)

29. [[PDF] Learning Vision-Language Uncertainties for Failure Prediction](https://openaccess.thecvf.com/content/ICCV2025/papers/Lafon_ViLU_Learning_Vision-Language_Uncertainties_for_Failure_Prediction_ICCV_2025_paper.pdf)

30. [VL-Uncertainty: Detecting Hallucination in Large Vision-Language Model via Uncertainty Estimation](https://bohrium.dp.tech/paper/arxiv/2411.11919)

31. [VL-Uncertainty Framework Overview - Emergent Mind](https://www.emergentmind.com/topics/vl-uncertainty-framework) - Novel architectures like ViLU integrate multi-part embeddings and post-hoc uncertainty ... Vision-La...

32. [VL-Uncertainty: Detecting Hallucination in Large Vision ...](https://arxiv.org/abs/2411.11919) - Given the higher information load processed by large vision-language models (LVLMs) compared to sing...

33. [From Uncertainty to Trust: Enhancing Reliability in Vision- ...](https://arxiv.org/abs/2412.06474) - Large vision-language models (LVLMs) excel at multimodal tasks but are prone to misinterpreting visu...

34. [Enhancing Vision-Language Model Reliability with Uncertainty ...](https://papers.cool/venue/LAflniLUwx@OpenReview) - Large vision-language models (LVLMs) excel at multimodal tasks but are prone to misinterpreting visu...

35. [On Epistemic Uncertainty of Visual Tokens for Object Hallucinations ...](https://huggingface.co/papers/2510.09008) - This paper investigates the problem of object hallucination—when large vision-language models (LVLMs...

36. [On Epistemic Uncertainty of Visual Tokens for Object Hallucinations ...](https://arxiv.org/abs/2510.09008) - Our statistical analysis found that there are positive correlations between visual tokens with high ...

37. [On Epistemic Uncertainty of Visual Tokens for Object Hallucinations ...](https://keenyjin.github.io/epistemic/) - Our statistical analysis found that there are positive correlations between visual tokens with high ...

38. [A Benchmark and Metric for Multimodal Epistemic and Aleatoric ...](https://iclr.cc/virtual/2025/poster/29051)

39. [Are vision language models robust to classic uncertainty challenges?](https://openreview.net/forum?id=4lCSYCNfmo) - Robustness against uncertain and ambiguous inputs is a critical challenge for deep learning models. ...

40. [[TINY] VISION LANGUAGE MODELS CAN IMPLICITLY](https://openreview.net/pdf?id=BkWVcXevTs)

41. [Consensus Hallucination: Why Five LLMs Agree on the Wrong Answer](https://notes.suhaib.in/docs/tech/news/consensus-hallucination-why-five-llms-agree-on-the-wrong-answer-and-why-ensembling-fails/) - When correlated LLMs converge on the same false output, majority voting backfires. Learn causes, mat...

42. [Multi-Agent AI Systems Collapse into Repetition](https://www.linkedin.com/posts/robrogowski_convergence-of-outputs-when-two-llm-interact-activity-7415428090937442305-qg5i) - Quotations 📚 “Conversations start coherent but often fall into repetition after a few turns.” 📚 “Onc...

43. [Should we be going MAD? A Look at Multi-Agent Debate ...](https://arxiv.org/html/2311.17371v3)

44. [Should we be going MAD? A Look at Multi-Agent Debate Strategies ...](https://proceedings.mlr.press/v235/smit24a.html) - Recent advancements in large language models (LLMs) underscore their potential for responding to inq...

45. [[2507.05981] Multi-Agent Debate Strategies to Enhance ... - arXiv](https://arxiv.org/abs/2507.05981) - Context: Large Language Model (LLM) agents are becoming widely used for various Requirements Enginee...

46. [Multi-Agent Debate Strategies to Enhance Requirements ...](https://arxiv.org/html/2507.05981v1) - MAD presents a promising approach for improving LLM accuracy in RE tasks. This study provides a foun...

47. [[PDF] Unraveling the Consensus-Diversity Tradeoff in Adaptive Multi ...](https://aclanthology.org/2025.emnlp-main.772.pdf) - 2024. Cooperate or collapse: Emergence of sustainable cooperation in a society of llm agents. Advanc...

48. [Mixture of Complementary Agents for Robust LLM Ensemble](https://openreview.net/forum?id=SbDf6E4kA2) - The authors claim that prior literature only considers the diversity and accuracy of individual LLM ...

49. [Asymmetric Self-Play for Language Models - Emergent Mind](https://www.emergentmind.com/topics/asymmetric-self-play-for-language-models) - Asymmetric self-play is a training paradigm that assigns distinct roles to agents for unsupervised c...

50. [R-Diverse: Mitigating Diversity Illusion in Self-Play LLM Training](https://arxiv.org/html/2602.13103v1) - Self-play has enabled strong performance gains by letting an agent improve through competition with ...

51. [AsymPuzl: An Asymmetric Puzzle for multi-agent cooperation](https://openreview.net/forum?id=SXcJh8hoLz) - Large Language Model (LLM) agents are increasingly studied in multi-turn, multi-agent scenarios, yet...

52. [[PDF] ASYMPUZL: AN ASYMMETRIC PUZZLE FOR MULTI- AGENT ...](https://openreview.net/pdf/b8bca2b178b7bc1d9b0d82800fe3598c28dbf54c.pdf)

53. [Emotion meets coordination: Designing multi-agent LLMs for fine ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC12885380/) - This study presents a modular multi-agent architecture for sentiment analysis, implemented with the ...

