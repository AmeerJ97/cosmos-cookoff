"""
ABEE Main Orchestrator — asyncio game loop.
Processes trajectories frame-by-frame, dispatches blind agents,
evaluates kinematic safety, manages dual-cache memory.

Implements:
- Life-Points (L_i) survival game
- Dynamic window expansion (W_i)
- Hyper-GRPO mutation on agent death
- Dynamic consensus threshold by frame index
- Physics oracle hard veto
"""
from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import aiohttp

from .models import AgentState, AgentResponse, FrameData, ArchiveMemory, SFTRecord
from .agents import dispatch_all_agents
from .local_inference import run_all_agents_local, load_model
from .oracle import PhysicsOracle
from .memory import DualCache
from .scorer import evaluate_frame, FrameVerdict, TrajectoryResult, compute_consensus_threshold
from .grpo import HyperGRPOManager
from .sft import SFTSerializer
from configs.settings import (
    DEFAULT_AGENTS, NGC_API_KEY, WINDOW_MIN, WINDOW_MAX,
    PREDICT_ENABLED, CONSENSUS_THRESHOLD, USE_LOCAL_MODEL,
    L_MAX,
)

log = logging.getLogger("abee.orchestrator")


# ── Telemetry callback type ──────────────────────────────────────────────────
TelemetryCallback = Callable[[str, int, FrameVerdict, list[float]], None]


# ── Embedder shim ────────────────────────────────────────────────────────────

async def embed_frame(
    session: aiohttp.ClientSession,
    frame: FrameData,
) -> list[float]:
    """Get 768-dim embedding for a frame via NIM embed API, or return zeros."""
    from configs.settings import NIM_BASE_URL, NIM_EMBED_MODEL
    if not frame.image_b64:
        return [0.0] * 768

    try:
        payload = {
            "model": NIM_EMBED_MODEL,
            "input": [{"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame.image_b64}"}}],
        }
        headers = {"Authorization": f"Bearer {NGC_API_KEY}", "Content-Type": "application/json"}
        async with session.post(
            f"{NIM_BASE_URL}/embeddings",
            json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["data"][0]["embedding"]
    except Exception as e:
        log.warning("Embedding failed: %s — using zeros", e)
    return [0.0] * 768


def frame_summary(frame_idx: int, embedding: list[float]) -> str:
    """Generate a compact text summary of a frame for LiveKV."""
    if not embedding or all(v == 0.0 for v in embedding):
        return f"Frame {frame_idx}: [no visual data]"
    import numpy as np
    arr = np.array(embedding[:64])
    return (
        f"Frame {frame_idx}: emb_mean={arr.mean():.4f} "
        f"emb_std={arr.std():.4f} emb_l2={float(np.linalg.norm(arr)):.4f}"
    )


# ── Conditional Predict2.5 tie-breaker ───────────────────────────────────────

async def invoke_predict_tiebreaker(
    session: aiohttp.ClientSession,
    frame: FrameData,
    responses: list[AgentResponse],
) -> str | None:
    """
    When agents disagree, optionally call Cosmos-Predict2.5 to assess stability.
    Returns "ACT" | "THINK" | None (if disabled or failed).
    """
    if not PREDICT_ENABLED:
        return None

    from configs.settings import NIM_BASE_URL, NIM_PREDICT_MODEL
    log.info("Invoking Predict2.5 tie-breaker at frame %d", frame.frame_idx)

    decisions = {r.agent_name: (r.decision.decision if r.decision else "FAIL") for r in responses}
    decision_summary = ", ".join(f"{k}={v}" for k, v in decisions.items())

    payload = {
        "model": NIM_PREDICT_MODEL,
        "messages": [{
            "role": "user",
            "content": (
                f"Agent disagreement at frame {frame.frame_idx}. "
                f"Decisions: {decision_summary}. "
                "Based on your world model prediction of the next frame, "
                "is this handoff currently SAFE to release? "
                "Reply with exactly: ACT or THINK."
            ),
        }],
        "temperature": 0.0,
        "max_tokens": 10,
    }
    headers = {"Authorization": f"Bearer {NGC_API_KEY}", "Content-Type": "application/json"}

    try:
        async with session.post(
            f"{NIM_BASE_URL}/chat/completions",
            json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=45),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                text = data["choices"][0]["message"]["content"].strip().upper()
                if "ACT" in text:
                    return "ACT"
                return "THINK"
    except Exception as e:
        log.warning("Predict2.5 tiebreaker failed: %s", e)
    return None


# ── Per-Agent Oracle Block Filter ────────────────────────────────────────────

def _filter_oracle_block(oracle_block: str, modality_mask: str) -> str:
    """Filter oracle output by agent modality to prevent correlated contamination.

    - 'gripper' agents see only contact/grip fields
    - 'velocity' agents see only motion/velocity fields
    - 'full' agents see everything
    """
    if modality_mask == "full" or not oracle_block:
        return oracle_block

    lines = oracle_block.split("\n")
    filtered = []
    for line in lines:
        lower = line.lower()
        if lower.startswith("[oracle]") or lower.startswith("[/oracle]"):
            filtered.append(line)
            continue
        if modality_mask == "gripper":
            # Only keep contact/grip/occlusion fields
            if any(k in lower for k in ["contact", "grip", "occlusion", "no_image"]):
                filtered.append(line)
        elif modality_mask == "velocity":
            # Only keep velocity/motion/physics score fields
            if any(k in lower for k in ["velocity", "physics_score", "depth", "no_image"]):
                filtered.append(line)

    return "\n".join(filtered)


# ── Main Orchestrator ────────────────────────────────────────────────────────

class Orchestrator:
    def __init__(
        self,
        sft: SFTSerializer | None = None,
        telemetry_cb: TelemetryCallback | None = None,
    ):
        self.grpo = HyperGRPOManager()
        self.agents: list[AgentState] = self.grpo.create_initial_ensemble(
            n_agents=len(DEFAULT_AGENTS)
        )
        self.cache = DualCache()
        self.oracle = PhysicsOracle()
        self.sft = sft
        self.telemetry_cb = telemetry_cb
        self.results: list[TrajectoryResult] = []

    def _get_living_agents(self) -> list[AgentState]:
        """Return only agents that are still alive."""
        return [a for a in self.agents if a.alive]

    def _handle_deaths(self) -> int:
        """
        Check for dead agents, update GRPO policy, and respawn replacements.
        Returns number of agents respawned.
        """
        respawns = 0
        for i, agent in enumerate(self.agents):
            if agent.is_dead and not hasattr(agent, '_death_processed'):
                # Record death in GRPO
                self.grpo.update_policy(agent.identity_idx, agent.accumulated_reward)
                log.info(
                    "Agent %s DIED (L=%.1f, reward=%.1f, survived=%d frames)",
                    agent.name, agent.life_points,
                    agent.accumulated_reward, agent.frames_survived,
                )

                # Spawn replacement
                new_agent = self.grpo.spawn_agent(agent.agent_idx)
                self.agents[i] = new_agent
                respawns += 1

        return respawns

    def _reset_agents_for_trajectory(self):
        """Reset per-trajectory agent state while preserving identity and GRPO history."""
        for agent in self.agents:
            agent.life_points = L_MAX
            agent.alive = True
            agent.window_size = WINDOW_MIN
            agent.spawn_correct_acts = 0
            agent.spawn_wrong_acts = 0
            agent.spawn_thinks = 0
            agent.frames_survived = 0
            agent.accumulated_reward = 0.0

    async def run_trajectory(
        self,
        session: aiohttp.ClientSession,
        trajectory,
        frames: list[FrameData],
    ) -> TrajectoryResult:
        """Process a single trajectory frame-by-frame with full survival game."""
        log.info(
            "Trajectory %s: %d frames, t_release=%d, T_safe=[%d,%d]",
            trajectory.trajectory_id, len(frames), trajectory.t_release,
            trajectory.t_safe_start, trajectory.t_safe_end,
        )

        # Reset per-trajectory state
        self.cache.clear_trajectory(trajectory.trajectory_id)
        self.oracle.reset()
        self._reset_agents_for_trajectory()

        frame_verdicts: list[FrameVerdict] = []
        release_frame: int | None = None
        correct_release = premature_release = late_release = False
        total_deaths = 0
        total_respawns = 0

        for frame in frames:
            frame_idx = frame.frame_idx

            # ── Check if all agents are dead ─────────────────────────────
            living = self._get_living_agents()
            if not living:
                log.warning("  [t=%d] ALL AGENTS DEAD — respawning ensemble", frame_idx)
                for i in range(len(self.agents)):
                    self.agents[i] = self.grpo.spawn_agent(i)
                    total_respawns += 1
                living = self._get_living_agents()

            # ── Step 1: Embed frame ──────────────────────────────────────
            embedding = await embed_frame(session, frame)
            frame.embedding = embedding

            # ── Step 2: Store in LiveKV ──────────────────────────────────
            frame.summary = frame.summary or frame_summary(frame_idx, embedding)
            self.cache.store_frame(frame)

            # ── Step 3: Retrieve from ArchiveKV ──────────────────────────
            archive_hits = self.cache.retrieve_archive(embedding)

            # ── Step 3b: Run physics oracle ──────────────────────────────
            img_rgb = None
            if frame.image_b64:
                import base64, numpy as np
                from PIL import Image
                import io
                img_rgb = np.array(Image.open(io.BytesIO(base64.b64decode(frame.image_b64))).convert("RGB"))
            oracle_report, oracle_block = self.oracle.run(img_rgb, frame_idx)

            # Hard veto — skip VLM entirely if physics says no
            if oracle_report.should_veto:
                log.info("  [t=%d] ORACLE VETO (physics=%.2f)", frame_idx, oracle_report.physics_score)
                from .models import EpistemicDecision
                responses = []
                for agent in living:
                    dec = EpistemicDecision(decision="THINK", action_type="CONTINUE_HOLD", confidence=0.0)
                    responses.append(AgentResponse(
                        agent_idx=agent.agent_idx, agent_name=agent.name,
                        frame_idx=frame_idx, decision=dec,
                        think_trace=f"[ORACLE VETO] physics_score={oracle_report.physics_score:.2f}",
                    ))
                verdict = evaluate_frame(frame_idx, responses, trajectory, self.agents)
                frame_verdicts.append(verdict)

                # Handle deaths from THINK drain during veto
                deaths = self._handle_deaths()
                total_deaths += sum(1 for v in verdict.agent_verdicts if not v.alive)
                total_respawns += deaths

                if self.telemetry_cb:
                    self.telemetry_cb(trajectory.trajectory_id, frame_idx, verdict, embedding)
                continue

            # ── Step 4: Build per-agent isolated context ────────────────
            # Each agent gets its own LiveKV window, archive memories,
            # and oracle block — filtered by modality mask to prevent
            # correlated failures from shared input contamination.
            live_windows: dict[int, list[str]] = {}
            agent_archives: dict[int, list] = {}
            agent_oracle_blocks: dict[int, str] = {}
            for agent in living:
                live_windows[agent.agent_idx] = self.cache.get_live_window(
                    trajectory.trajectory_id, frame_idx, agent.window_size
                )
                # Per-agent archive retrieval (modality-masked embedding query)
                agent_archives[agent.agent_idx] = self.cache.retrieve_archive(
                    embedding, modality_mask=agent.modality_mask
                )
                # Per-agent oracle block (filter by modality)
                agent_oracle_blocks[agent.agent_idx] = _filter_oracle_block(
                    oracle_block, agent.modality_mask
                )

            # ── Step 5: Dispatch living agents ───────────────────────────
            if USE_LOCAL_MODEL:
                responses = run_all_agents_local(
                    living, frame, live_windows,
                    agent_archives, agent_oracle_blocks,
                )
            else:
                responses = await dispatch_all_agents(
                    session, living, frame, live_windows, agent_archives
                )

            # ── Step 6: Kinematic evaluation (with Life-Points) ──────────
            verdict = evaluate_frame(frame_idx, responses, trajectory, self.agents)
            frame_verdicts.append(verdict)

            # Log per-agent status
            for av in verdict.agent_verdicts:
                status = "ALIVE" if av.alive else "DEAD"
                log.info(
                    "    %s: %s (conf=%.2f) L=%.0f W=%d [%s] %s",
                    av.agent_name, av.decision, av.confidence,
                    av.life_points, av.window_size,
                    av.life_reason, status,
                )

            log.info(
                "  [t=%d] ACT=%d THINK=%d threshold=%d/%d safe=%s consensus=%s",
                frame_idx, verdict.act_count, verdict.think_count,
                verdict.consensus_threshold, verdict.n_alive,
                verdict.is_in_safe_window, verdict.consensus_act,
            )

            # ── Step 7: Handle agent deaths and respawns ─────────────────
            new_deaths = sum(1 for av in verdict.agent_verdicts if not av.alive)
            total_deaths += new_deaths
            respawns = self._handle_deaths()
            total_respawns += respawns

            # ── Step 8: Telemetry callback ───────────────────────────────
            if self.telemetry_cb:
                self.telemetry_cb(
                    trajectory.trajectory_id, frame_idx, verdict, embedding
                )

            # ── Step 9: Tie-breaker for split decisions ────────────────────
            # Only invoke if tiebreaker could push us over threshold
            if (verdict.act_count > 0
                    and verdict.act_count < verdict.n_alive
                    and not verdict.consensus_act
                    and verdict.act_count + 1 >= verdict.consensus_threshold):
                tb = await invoke_predict_tiebreaker(session, frame, responses)
                if tb == "ACT":
                    log.info("  Predict2.5 tiebreaker -> ACT")
                    # Recompute consensus with tiebreaker counted as +1
                    effective_act = verdict.act_count + 1
                    if effective_act >= verdict.consensus_threshold:
                        verdict = FrameVerdict(
                            frame_idx=frame_idx,
                            consensus_act=True,
                            act_count=effective_act,
                            think_count=verdict.think_count,
                            is_in_safe_window=verdict.is_in_safe_window,
                            is_premature=verdict.is_premature,
                            is_late=verdict.is_late,
                            mean_confidence=verdict.mean_confidence,
                            agent_verdicts=verdict.agent_verdicts,
                            consensus_threshold=verdict.consensus_threshold,
                            n_alive=verdict.n_alive,
                        )

            # ── Step 10: Commit release if consensus ─────────────────────
            if verdict.consensus_act and release_frame is None:
                release_frame = frame_idx
                correct_release = verdict.is_in_safe_window
                premature_release = verdict.is_premature
                late_release = verdict.is_late
                log.info(
                    "  RELEASE COMMITTED at frame %d — %s",
                    frame_idx,
                    "CORRECT" if correct_release else ("PREMATURE" if premature_release else "LATE"),
                )

                # ── Step 11: Archive golden memories + SFT ───────────────
                if correct_release:
                    for resp in responses:
                        if resp.decision and resp.decision.decision == "ACT" and resp.think_trace:
                            golden_rule = self._distill_rule(resp, trajectory, frame_idx)
                            mem = ArchiveMemory(
                                trajectory_id=trajectory.trajectory_id,
                                frame_idx=frame_idx,
                                agent_name=resp.agent_name,
                                golden_rule=golden_rule,
                                embedding=embedding,
                            )
                            self.cache.add_golden_memory(mem)

                            if self.sft:
                                agent = self.agents[resp.agent_idx]
                                rec = SFTRecord(
                                    trajectory_id=trajectory.trajectory_id,
                                    frame_idx=frame_idx,
                                    agent_name=resp.agent_name,
                                    agent_bias=agent.prompt_bias[:80],
                                    temporal_stride=agent.temporal_stride,
                                    modality_mask=agent.modality_mask,
                                    decision=resp.decision.decision,
                                    confidence=resp.decision.confidence,
                                    think_trace=resp.think_trace,
                                    is_correct=True,
                                    ground_truth_t_release=trajectory.t_release,
                                    embedding_snippet=embedding[:16],
                                    golden_rule=golden_rule,
                                )
                                self.sft.write(rec)

                break  # Stop processing frames after release

        result = TrajectoryResult(
            trajectory_id=trajectory.trajectory_id,
            total_frames=len(frames),
            release_frame=release_frame,
            ground_truth_release=trajectory.t_release,
            correct_release=correct_release,
            premature_release=premature_release,
            late_release=late_release,
            no_release=(release_frame is None),
            frame_verdicts=frame_verdicts,
            agent_deaths=total_deaths,
            agent_respawns=total_respawns,
        )
        self.results.append(result)
        log.info("  Result: %s", result.summary)
        return result

    def _distill_rule(
        self, resp: AgentResponse, trajectory, frame_idx: int
    ) -> str:
        """Locally distill a think trace into a compressed golden rule."""
        trace = resp.think_trace
        lines = [l.strip() for l in trace.split("\n") if l.strip()]
        keywords = ["grip", "velocity", "stable", "transfer", "hand", "wrist",
                    "release", "force", "contact", "safe", "moment", "frame"]
        relevant = [l for l in lines if any(k in l.lower() for k in keywords)]
        summary_lines = relevant[:5] if relevant else lines[:3]
        summary = " ".join(summary_lines)[:450]
        return (
            f"[{resp.agent_name} | frame={frame_idx} | traj={trajectory.trajectory_id}] "
            f"{summary}"
        )

    async def run_dataset(
        self,
        session: aiohttp.ClientSession,
        trajectories_and_frames: list[tuple],
    ) -> list[TrajectoryResult]:
        """Process all trajectories sequentially."""
        for trajectory, frames in trajectories_and_frames:
            await self.run_trajectory(session, trajectory, frames)
        return self.results

    def print_summary(self):
        """Print aggregate statistics."""
        total = len(self.results)
        correct = sum(1 for r in self.results if r.correct_release)
        premature = sum(1 for r in self.results if r.premature_release)
        late = sum(1 for r in self.results if r.late_release)
        no_rel = sum(1 for r in self.results if r.no_release)
        deaths = sum(r.agent_deaths for r in self.results)
        respawns = sum(r.agent_respawns for r in self.results)

        print(f"\n{'='*60}")
        print(f"ABEE RESULTS — {total} trajectories")
        print(f"  Correct releases:   {correct:3d} ({100*correct/max(total,1):.1f}%)")
        print(f"  Premature releases: {premature:3d} ({100*premature/max(total,1):.1f}%)")
        print(f"  Late releases:      {late:3d} ({100*late/max(total,1):.1f}%)")
        print(f"  No release:         {no_rel:3d} ({100*no_rel/max(total,1):.1f}%)")
        print(f"  ArchiveKV memories: {self.cache.archive.size}")
        print(f"  Total agent deaths: {deaths}")
        print(f"  Total respawns:     {respawns}")
        print(f"\n  Hyper-GRPO Stats:")
        grpo_stats = self.grpo.stats
        print(f"    Mean reward:  {grpo_stats['reward_mean']:.2f}")
        print(f"    Reward std:   {grpo_stats['reward_std']:.2f}")
        print(f"    Top identities:")
        for ti in grpo_stats['top_identities']:
            print(f"      [{ti['identity_idx']:2d}] p={ti['probability']:.3f} "
                  f"stride={ti['temporal_stride']} mask={ti['modality_mask']}")
        print()
        for agent in self.agents:
            status = "ALIVE" if agent.alive else "DEAD"
            print(
                f"  Agent {agent.name}: "
                f"L={agent.life_points:.0f} W={agent.window_size} "
                f"correct={agent.correct_acts} wrong={agent.wrong_acts} "
                f"thinks={agent.total_thinks} acc={agent.accuracy:.2f} [{status}]"
            )
        print(f"{'='*60}\n")
