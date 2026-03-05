# ABEE Training Log

Tracks all training runs — formal (cloud SFT/GRPO) and informal (local dry-runs, experiments).

## Run History

### Dry-Runs (Local, Synthetic)

| Run # | Date | Trajectories | Correct | Premature | Late | No-Release | Deaths | ArchiveKV | Notes |
|-------|------|-------------|---------|-----------|------|-----------|--------|-----------|-------|
| 1 | 2026-03-05 | 50 | 80% (40) | 0% | 0% | 20% (10) | 69 | 0 | First run with full P0 features. Pre-ArchiveKV fix. |
| 2 | 2026-03-05 | 50 | 72% (36) | 0% | 0% | 28% (14) | 43 | 0 | Second baseline. ArchiveKV not loading from disk. |
| 3 | 2026-03-05 | 50 | 70% (35) | 0% | 0% | 30% (15) | 49 | 116 | Post-ArchiveKV write fix, but no load-on-startup. |
| 4 | 2026-03-05 | 50 | 74% (37) | 0% | 0% | 26% (13) | 61 | 241 | Post auto-load fix. ArchiveKV accumulating from run 3. |
| 5 | 2026-03-05 | 50 | 68% (34) | 0% | 0% | 32% (16) | 68 | 352 | Consecutive. GRPO top: identity[19] stride=1 mask=gripper. |
| 6 | 2026-03-05 | 50 | 72% (36) | 0% | 4% (2) | 24% (12) | 49 | 477 | First late releases observed. Velocity mask converging. |

### Real Inference Runs (Cosmos-Reason2-8B, Local 4-bit)

| Run # | Date | Trajectories | Correct | Premature | Late | No-Release | Model | Notes |
|-------|------|-------------|---------|-----------|------|-----------|-------|-------|
| — | — | — | — | — | — | — | — | No real inference runs yet |

### Cloud Training Runs (SFT / GRPO)

| Run # | Date | Type | Platform | GPUs | Epochs | Dataset Size | Final Accuracy | ACT Precision | Cost | Checkpoint |
|-------|------|------|----------|------|--------|-------------|----------------|---------------|------|-----------|
| — | — | — | — | — | — | — | — | — | — | — |

## Observations

### Dry-Run Trends
- **0% premature rate across ALL runs** — Life-Points + dynamic consensus is rock solid
- **20-32% no-release rate** — agents are conservative; may improve with real inference + ArchiveKV
- **ArchiveKV accumulation works** — 0 → 116 → 241 → 352 → 477 memories across consecutive runs
- **GRPO convergence**: velocity mask and stride=1 identities are being favored
- **Deaths decreasing**: 69 → 43 → 49 → 61 → 68 → 49 (GRPO learning better identities)
- **Accuracy stable ~70-74%** — expected floor for synthetic random decisions (not informed by archive)

### Key Insight
Dry-run accuracy won't improve from ArchiveKV because agents make random synthetic decisions that don't read the archive. Real inference is needed to see the benefit.
