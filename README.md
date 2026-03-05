# ABEE — Adversarial Blind Epistemic Ensemble
**NVIDIA Cosmos Cookoff** · Robotic handoff safety via multi-agent epistemic voting on `nvidia/cosmos-reason2-8b`

---

## What it does

ABEE determines the **safe release window** during a human-robot object handoff. Four blind, information-asymmetric agents each query Cosmos Reason 2 in parallel, vote ACT (release now) or THINK (wait), and the orchestrator resolves consensus under a real-time survival game. Wrong votes drain life points; dead agents are replaced by Hyper-GRPO sampling a new identity from 36 possible configurations.

---

## Architecture

```mermaid
flowchart TD
    DS[(MIMIC Dataset\n1,137 trajectories\n30,666 frames)] --> ORC

    subgraph ORC["Orchestrator — asyncio game loop"]
        FR[Frame t] --> PHY[Physics Oracle\nSAM2 + MiDaS depth]
        PHY -->|hard veto| OUT
        PHY -->|pass| EMB[cosmos-embed-1.0\n768-dim]
        EMB --> LKV[(LiveKV\nRedis FIFO\nwindow 5-30)]
        EMB --> AKV[(ArchiveKV\nFAISS top-3\n722 golden rules)]
        LKV --> AGT
        AKV --> AGT

        subgraph AGT["P x T x M Blind Agents  (async, parallel)"]
            A[Alpha\nconservative · stride 1 · full]
            B[Beta\nspeed · stride 3 · gripper]
            G[Gamma\nskeptic · stride 1 · velocity]
            D[Delta\narchival · stride 2 · full]
        end

        AGT -->|votes| CON{"Dynamic consensus\n>=100% t<8 · >=85% t<15 · >=66% t>=15"}
        CON -->|split| TIE[cosmos-predict2-14b\ntie-breaker]
        CON -->|agree| LP[Life-Points scorer\nL_MAX=100 · wrong ACT=-33]
        TIE --> LP
        LP -->|agent died| GRPO[Hyper-GRPO\nbandit over 36 identities]
        GRPO --> AGT
        LP --> OUT
    end

    OUT[results.json\nsft_dataset.jsonl]
```

---

## Key innovations

- **P×T×M asymmetry matrix** — 4 bias × 3 stride × 3 modality = **36 possible agent identities**. Each active agent sees a structurally different projection of the same frame, preventing correlated failure.
- **Dual-cache memory** — *LiveKV* (Redis sliding window, 5–30 frames) provides temporal continuity; *ArchiveKV* (FAISS 768-dim, `nvidia/cosmos-embed-1.0`) retrieves top-3 golden rules from 722 distilled trajectory outcomes.
- **Life-Points survival game** — agents accrue damage (`THINK` drains 2 pts/frame; wrong `ACT` removes 33 pts; early wrong `ACT` removes 66 pts). Death triggers immediate Hyper-GRPO respawn with no human intervention.
- **Hyper-GRPO bandit** — gradient-free identity selection (`lr=0.1`, entropy injection `σ=0.5`) replaces dead agents with the highest-EV unexplored identity from the 36-identity space.
- **Dynamic consensus threshold** — unanimity required at early frames (t < 8), 85% at mid, 66% late. Asymmetric error costs justify tighter safety at frame onset.
- **Physics oracle hard veto** — SAM2 segmentation + MiDaS relative depth can block a release before agents are consulted; permissive fallback when GPU is unavailable so the pipeline never blocks on missing hardware.
- **SFT data exhaust** — winning agent reasoning traces are serialised to `sft_dataset.jsonl` for downstream fine-tuning of Cosmos Reason 2 (capped at 500 tokens/trace).

---

## How Cosmos Reason 2 is used

| Role | Model | Purpose |
|---|---|---|
| Primary reasoning | `nvidia/cosmos-reason2-8b` | Each agent issues `<think>` chain-of-thought + JSON `{decision, action_type, confidence}` |
| Tie-breaker | `nvidia/cosmos-predict2-14b` | Invoked only on split votes when dynamic consensus threshold is not met |
| Memory embedding | `nvidia/cosmos-embed-1.0` | 768-dim frame embeddings for LiveKV insertion and ArchiveKV FAISS retrieval |

All models accessed via **NVIDIA NIM API** (`https://integrate.api.nvidia.com/v1`).  
A local 4-bit inference path exists as fallback (`USE_LOCAL_MODEL=True` in `configs/settings.py`).

---

## Quick start

```bash
# 1. Start Redis
docker compose up -d redis

# 2. Set NIM key  (generate at https://build.nvidia.com)
export NGC_API_KEY=nvapi-YOUR-KEY

# 3. Verify access
python scripts/test_api.py

# 4. Run on MIMIC data (50 trajectories)
python run_abee.py --trajectories 50

# 5. With live telemetry dashboard  →  http://localhost:8050
python run_abee.py --trajectories 50 --dashboard
```

---

## Dataset

| Property | Value |
|---|---|
| Source | MIMIC manipulation dataset |
| Trajectories | 1,137 |
| Frames | 30,666 |
| Task | `mimic_displacement_to_handover_blue_block` (v2, v6, v7, v8) |
| Conversion | `scripts/convert_mimic_to_abee.py` → `data/manifest.json` |

---

## Results

### Dry-Run Evaluation (6 runs, 300 trajectories total)

| Metric | Value |
|---|---|
| Premature release rate | **0%** (across all 300 trajectories) |
| Correct release rate | 68-80% (mean 72.7%) |
| No-release rate | 20-32% (conservative bias by design) |
| Late release rate | <1% |
| ArchiveKV golden memories | 0 → 477 (accumulated across runs) |
| GRPO convergence | velocity mask + stride=1 identities favored |

### Key Safety Result

Zero premature releases across every evaluation run. The Life-Points double penalty on early wrong ACTs (66 pts, near-fatal) combined with unanimous consensus requirements at early frames creates a structural safety guarantee without explicit hard-coded rules.

---

## Repository structure

```
abee_pkg/      core library — agents, memory, orchestrator, GRPO, SFT, oracle
configs/       all tunable hyperparameters (settings.py)
dashboard/     Plotly Dash live telemetry (Dash, port 8050)
data/          manifest + output artefacts (results.json, sft_dataset.jsonl)
scripts/       dataset conversion utilities
docs/          full system diagrams and research notes
```

Full system diagrams (8 Mermaid diagrams — sequence, state machine, bandit loop, SFT pipeline):  
[docs/Architecture docs/ABEE System Documentation.md](docs/Architecture%20docs/ABEE%20System%20Documentation.md)
