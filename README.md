# ABEE — Adversarial Blind Epistemic Ensemble
### NVIDIA Cosmos Cookoff Submission

**Goal:** Predict the safe human-robot object handoff release window using 3 blind, information-asymmetric VLM agents backed by Cosmos-Reason2-8B.

---

## Quick Start

```bash
# 1. Start Redis
docker compose up -d redis

# 2. Set your NIM API key (from build.nvidia.com)
export NGC_API_KEY=nvapi-YOUR-KEY

# 3. Test API access
python test_api.py

# 4. Dry run (no API calls, synthetic data)
python run_abee.py --dry-run --trajectories 10

# 5. Real run with Cosmos-Reason2-8B
python run_abee.py --trajectories 50

# 6. With live telemetry dashboard (http://localhost:8050)
python run_abee.py --trajectories 50 --dashboard
```

## Architecture

```
3 Blind Agents (Cosmos-Reason2-8B via NIM)
  Agent Alpha  — hyper-conservative, full embedding, stride=1
  Agent Beta   — speed-optimized, gripper subspace, stride=3
  Agent Gamma  — kinematic skeptic, velocity subspace, stride=1

Each agent: THINK (defer) | ACT (commit release)
Consensus: ≥2 agents ACT → release committed

Memory:
  LiveKV  — Redis FIFO sliding window (temporal continuity)
  ArchiveKV — FAISS RAG (golden memory retrieval)

Tie-breaker: Cosmos-Predict2.5 invoked on split votes

Output: results.json + sft_dataset.jsonl (SFT training data)
```

## Directory Structure

```
cosmos-cookoff/
├── run_abee.py          # CLI entry point
├── test_api.py          # NIM API key tester
├── docker-compose.yml   # Redis service
├── configs/
│   └── settings.py      # All tunable parameters
├── abee_pkg/
│   ├── agents.py        # NIM async dispatcher
│   ├── memory.py        # LiveKV + ArchiveKV
│   ├── models.py        # Pydantic schemas
│   ├── orchestrator.py  # Main asyncio game loop
│   ├── scorer.py        # O(1) kinematic evaluator
│   ├── sft.py           # SFT dataset serializer
│   └── data_loader.py   # DROID/YCB manifest loader
├── dashboard/
│   └── app.py           # Plotly Dash UMAP telemetry
├── data/                # Output artifacts
│   ├── manifest.json    # (place real dataset here)
│   ├── sft_dataset.jsonl
│   └── results.json
└── docs/                # Architecture docs (place here)
```

## NIM API Key

Generate at **https://build.nvidia.com** → any model → "Get API Key"

Set via env: `export NGC_API_KEY=nvapi-...`
