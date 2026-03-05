"""
ABEE Data Models — Pydantic schemas for agent decisions, state, and SFT records.
"""
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from dataclasses import dataclass, field as dc_field
import time


# ── Agent Decision (validated from NIM response) ─────────────────────────────

class EpistemicDecision(BaseModel):
    """Strict schema for what each blind agent must return."""
    decision: Literal["ACT", "THINK"] = Field(
        ..., description="ACT to commit release, THINK to observe and defer."
    )
    action_type: Literal["SAFE_RELEASE_NOW", "CONTINUE_HOLD"] = Field(
        ..., description="Physical execution command."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Internal confidence scalar."
    )

    @field_validator("action_type")
    @classmethod
    def validate_action_logic(cls, v, info):
        decision = info.data.get("decision")
        if decision == "ACT" and v != "SAFE_RELEASE_NOW":
            raise ValueError("ACT must pair with SAFE_RELEASE_NOW")
        if decision == "THINK" and v != "CONTINUE_HOLD":
            raise ValueError("THINK must pair with CONTINUE_HOLD")
        return v


# ── Agent Runtime State ──────────────────────────────────────────────────────

@dataclass
class AgentState:
    """Mutable per-agent state tracked by the orchestrator."""
    agent_idx: int
    name: str
    prompt_bias: str
    temporal_stride: int
    modality_mask: str
    window_size: int = 5
    total_acts: int = 0
    correct_acts: int = 0
    wrong_acts: int = 0
    total_thinks: int = 0
    # Life-Points system
    life_points: float = 100.0
    alive: bool = True
    # Hyper-GRPO identity tracking
    identity_idx: int = -1      # index into the 36-combo asymmetry matrix
    accumulated_reward: float = 0.0
    frames_survived: int = 0
    # Lifecycle counters (per-spawn, reset on mutation)
    spawn_correct_acts: int = 0
    spawn_wrong_acts: int = 0
    spawn_thinks: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct_acts / max(self.total_acts, 1)

    @property
    def is_dead(self) -> bool:
        return self.life_points <= 0 or not self.alive

    def kill(self):
        """Mark agent as dead."""
        self.alive = False

    def reset_life(self, l_max: float = 100.0, w_min: int = 5):
        """Full life restoration on correct ACT."""
        self.life_points = l_max
        self.window_size = w_min


# ── Trajectory / Frame Data ─────────────────────────────────────────────────

@dataclass
class TrajectoryMeta:
    """Metadata for a single trajectory from the dataset."""
    trajectory_id: str
    total_frames: int
    t_release: int               # ground truth release frame
    t_safe_start: int            # t_release - tau_early
    t_safe_end: int              # t_release + tau_late
    source: str = "droid"        # "droid" | "ycb"
    video_path: str = ""


@dataclass
class FrameData:
    """Single frame extracted from a trajectory."""
    trajectory_id: str
    frame_idx: int
    image_b64: str = ""          # base64-encoded JPEG
    embedding: list[float] = dc_field(default_factory=list)  # 768-dim
    summary: str = ""            # text summary stored in LiveKV


# ── Agent Response (parsed from NIM) ─────────────────────────────────────────

@dataclass
class AgentResponse:
    """Parsed response from a single agent for a single frame."""
    agent_idx: int
    agent_name: str
    frame_idx: int
    decision: EpistemicDecision | None  # None if parse failed
    think_trace: str = ""
    raw_output: str = ""
    latency_ms: float = 0.0
    parse_error: str = ""


# ── SFT Record ──────────────────────────────────────────────────────────────

class SFTRecord(BaseModel):
    """A single training example for the SFT dataset."""
    trajectory_id: str
    frame_idx: int
    agent_name: str
    agent_bias: str
    temporal_stride: int
    modality_mask: str
    decision: str           # "ACT" or "THINK"
    confidence: float
    think_trace: str
    is_correct: bool
    ground_truth_t_release: int
    embedding_snippet: list[float] = Field(default_factory=list)  # first 16 dims
    golden_rule: str = ""   # distilled memory (post-curation)
    timestamp: float = Field(default_factory=time.time)


# ── Archive Memory (stored in FAISS) ─────────────────────────────────────────

@dataclass
class ArchiveMemory:
    """A golden memory stored in the ArchiveKV FAISS index."""
    trajectory_id: str
    frame_idx: int
    agent_name: str
    golden_rule: str         # distilled <500 token physical principle
    embedding: list[float]   # 768-dim for FAISS indexing
