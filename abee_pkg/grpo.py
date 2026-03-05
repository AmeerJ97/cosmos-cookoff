"""
ABEE Hyper-GRPO — Discrete policy optimization over the Asymmetry Matrix.

When agents die (L_i <= 0), their identity's logit is updated based on
accumulated reward. New agents are spawned by sampling from the updated
distribution. Over time, the system converges on optimal trait combinations.
"""
from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass

from .models import AgentState
from configs.settings import (
    PROMPT_BIASES, TEMPORAL_STRIDES, MODALITY_MASKS,
    N_IDENTITIES, GRPO_LEARNING_RATE, GRPO_STAGNATION_THRESHOLD,
    GRPO_ENTROPY_SIGMA, L_MAX, WINDOW_MIN,
)

log = logging.getLogger("abee.grpo")

# Agent name pool for respawned agents
_NAME_POOL = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi",
    "Rho", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega",
]


def _decode_identity(idx: int) -> tuple[int, int, int]:
    """Decode a flat identity index into (prompt_idx, temporal_idx, modality_idx)."""
    n_t = len(TEMPORAL_STRIDES)
    n_m = len(MODALITY_MASKS)
    p = idx // (n_t * n_m)
    remainder = idx % (n_t * n_m)
    t = remainder // n_m
    m = remainder % n_m
    return p, t, m


def _encode_identity(p: int, t: int, m: int) -> int:
    """Encode (prompt_idx, temporal_idx, modality_idx) into flat index."""
    n_t = len(TEMPORAL_STRIDES)
    n_m = len(MODALITY_MASKS)
    return p * (n_t * n_m) + t * n_m + m


class HyperGRPOManager:
    """
    Manages the discrete categorical distribution over the 36-combo
    Asymmetry Matrix. Updates policy when agents die, samples new
    identities for respawned agents.
    """

    def __init__(self, learning_rate: float = GRPO_LEARNING_RATE):
        self.alpha = learning_rate
        self.logits = np.zeros(N_IDENTITIES, dtype=np.float64)
        self.reward_history: list[float] = []
        self._spawn_counter = 0
        self._total_deaths = 0
        self._total_spawns = 0

    @property
    def probabilities(self) -> np.ndarray:
        """Current sampling probabilities (softmax of logits)."""
        shifted = self.logits - self.logits.max()  # numerical stability
        exp = np.exp(shifted)
        return exp / exp.sum()

    def sample_identity(self) -> int:
        """Sample an identity index from the current policy distribution."""
        probs = self.probabilities
        return int(np.random.choice(N_IDENTITIES, p=probs))

    def update_policy(self, identity_idx: int, reward: float):
        """
        Update the logit for the deceased agent's identity based on
        its accumulated reward relative to the historical mean.
        """
        self.reward_history.append(reward)
        self._total_deaths += 1

        if len(self.reward_history) < 2:
            return

        mean_r = np.mean(self.reward_history)
        std_r = np.std(self.reward_history) + 1e-8
        advantage = (reward - mean_r) / std_r

        self.logits[identity_idx] += self.alpha * advantage

        p, t, m = _decode_identity(identity_idx)
        log.info(
            "GRPO update: identity=%d (P=%d T=%d M=%d) reward=%.1f "
            "advantage=%.3f logit=%.3f",
            identity_idx, p, t, m, reward, advantage,
            self.logits[identity_idx],
        )

        # Stagnation guard: if mean reward is terrible, inject entropy
        if mean_r < GRPO_STAGNATION_THRESHOLD and len(self.reward_history) > 10:
            self.inject_entropy()

    def inject_entropy(self, sigma: float = GRPO_ENTROPY_SIGMA):
        """Break out of local minima by adding noise to logits."""
        noise = np.random.normal(0, sigma, size=self.logits.shape)
        self.logits += noise
        log.warning("GRPO stagnation detected — injecting entropy (sigma=%.2f)", sigma)

    def spawn_agent(self, agent_idx: int) -> AgentState:
        """
        Sample a new identity and create a fresh AgentState.
        """
        identity_idx = self.sample_identity()
        p_idx, t_idx, m_idx = _decode_identity(identity_idx)

        self._spawn_counter += 1
        self._total_spawns += 1

        name_idx = self._spawn_counter % len(_NAME_POOL)
        name = f"{_NAME_POOL[name_idx]}-{self._spawn_counter}"

        agent = AgentState(
            agent_idx=agent_idx,
            name=name,
            prompt_bias=PROMPT_BIASES[p_idx],
            temporal_stride=TEMPORAL_STRIDES[t_idx],
            modality_mask=MODALITY_MASKS[m_idx],
            window_size=WINDOW_MIN,
            life_points=L_MAX,
            alive=True,
            identity_idx=identity_idx,
        )

        log.info(
            "Spawned %s [idx=%d] identity=%d (P=%d T=%d M=%d) — "
            "bias=%s stride=%d mask=%s",
            agent.name, agent_idx, identity_idx, p_idx, t_idx, m_idx,
            PROMPT_BIASES[p_idx][:40] + "...",
            TEMPORAL_STRIDES[t_idx],
            MODALITY_MASKS[m_idx],
        )
        return agent

    def create_initial_ensemble(self, n_agents: int = 4) -> list[AgentState]:
        """Create the initial agent ensemble from DEFAULT_AGENTS config."""
        from configs.settings import DEFAULT_AGENTS

        agents = []
        for i, identity in enumerate(DEFAULT_AGENTS[:n_agents]):
            # Find the closest matching identity index
            p_idx = _find_closest_prompt(identity.prompt_bias)
            t_idx = TEMPORAL_STRIDES.index(identity.temporal_stride) if identity.temporal_stride in TEMPORAL_STRIDES else 0
            m_idx = MODALITY_MASKS.index(identity.modality_mask) if identity.modality_mask in MODALITY_MASKS else 0
            idx = _encode_identity(p_idx, t_idx, m_idx)

            agent = AgentState(
                agent_idx=i,
                name=identity.name,
                prompt_bias=identity.prompt_bias,
                temporal_stride=identity.temporal_stride,
                modality_mask=identity.modality_mask,
                window_size=WINDOW_MIN,
                life_points=L_MAX,
                alive=True,
                identity_idx=idx,
            )
            agents.append(agent)

        return agents

    def get_top_identities(self, k: int = 5) -> list[dict]:
        """Return the top-k identity combinations by probability."""
        probs = self.probabilities
        top_indices = np.argsort(probs)[::-1][:k]
        results = []
        for idx in top_indices:
            p, t, m = _decode_identity(int(idx))
            results.append({
                "identity_idx": int(idx),
                "probability": float(probs[idx]),
                "prompt_idx": p,
                "temporal_stride": TEMPORAL_STRIDES[t],
                "modality_mask": MODALITY_MASKS[m],
                "logit": float(self.logits[idx]),
            })
        return results

    @property
    def stats(self) -> dict:
        return {
            "total_deaths": self._total_deaths,
            "total_spawns": self._total_spawns,
            "reward_mean": float(np.mean(self.reward_history)) if self.reward_history else 0.0,
            "reward_std": float(np.std(self.reward_history)) if self.reward_history else 0.0,
            "top_identities": self.get_top_identities(3),
        }


def _find_closest_prompt(bias: str) -> int:
    """Find the prompt bias pool index that best matches a given bias string."""
    for i, p in enumerate(PROMPT_BIASES):
        if p[:50] == bias[:50]:
            return i
    return 0
