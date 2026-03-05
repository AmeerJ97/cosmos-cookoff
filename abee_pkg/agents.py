"""
ABEE Agent Dispatcher — async NIM API client for blind epistemic agents.
Constructs asymmetric payloads per agent identity and parses responses.
"""
from __future__ import annotations
import asyncio
import json
import re
import time
import logging
from typing import Sequence

import aiohttp

from .models import (
    AgentState, AgentResponse, EpistemicDecision,
    FrameData, ArchiveMemory,
)
from configs.settings import (
    NIM_BASE_URL, NIM_ACTIVE_MODEL, NGC_API_KEY,
    NIM_TEMPERATURE, NIM_MAX_TOKENS, NIM_TIMEOUT, MAX_RETRIES,
)

log = logging.getLogger("abee.agents")


# ── Payload Construction ─────────────────────────────────────────────────────

def _build_system_prompt(agent: AgentState) -> str:
    """Build the information-asymmetric system prompt for this agent."""
    return (
        "You are a physical AI agent evaluating a human-robot object handoff.\n"
        f"[BIAS: {agent.prompt_bias}]\n"
        f"Your temporal perception stride is {agent.temporal_stride}x.\n"
        f"Your sensor modality focus: {agent.modality_mask}.\n\n"
        "RULES:\n"
        "1. Wrap ALL reasoning in <think>...</think> tags.\n"
        "2. After </think>, output ONLY a JSON object with these exact fields:\n"
        '   {"decision": "ACT"|"THINK", "action_type": "SAFE_RELEASE_NOW"|"CONTINUE_HOLD", "confidence": 0.0-1.0}\n'
        "3. ACT means you believe the handoff is safe NOW. THINK means continue observing.\n"
        "4. ACT must pair with SAFE_RELEASE_NOW. THINK must pair with CONTINUE_HOLD.\n"
    )


def _build_user_content(
    frame: FrameData,
    live_kv_window: list[str],
    archive_memories: list[ArchiveMemory],
    agent: AgentState,
) -> list[dict]:
    """Build the multimodal user message content."""
    # Archive KV section
    if archive_memories:
        archive_text = "--- ARCHIVE MEMORY (Top-K Golden Rules) ---\n"
        for i, mem in enumerate(archive_memories, 1):
            archive_text += f"[{i}] {mem.golden_rule}\n"
        archive_text += "-------------------------------------------\n\n"
    else:
        archive_text = "--- ARCHIVE MEMORY: [Empty — burn-in phase] ---\n\n"

    # LiveKV temporal window (apply stride)
    stride = agent.temporal_stride
    if stride > 1:
        windowed = live_kv_window[::stride]
    else:
        windowed = live_kv_window

    live_text = "--- LIVE TEMPORAL WINDOW ---\n"
    for entry in windowed:
        live_text += f"{entry}\n"
    live_text += "----------------------------\n\n"

    # Embedding summary (masked by modality)
    emb = frame.embedding
    if agent.modality_mask == "gripper" and len(emb) >= 384:
        emb_display = emb[:384]
        mask_note = "(Gripper-subspace: dims 0-383)"
    elif agent.modality_mask == "velocity" and len(emb) >= 768:
        emb_display = emb[384:768]
        mask_note = "(Velocity-subspace: dims 384-767)"
    else:
        emb_display = emb[:768] if emb else []
        mask_note = "(Full 768-dim vector)"

    # Truncate display to first 16 dims for prompt efficiency
    emb_snippet = [round(x, 4) for x in emb_display[:16]]
    sensor_text = (
        f"--- CURRENT SENSOR STATE {mask_note} ---\n"
        f"Frame: {frame.frame_idx} | Embedding (first 16): {emb_snippet}\n"
        f"Frame summary: {frame.summary}\n"
        "---\n\n"
        "DIRECTIVE: Evaluate handoff safety. Wrap reasoning in <think></think> tags, "
        "then output a valid JSON object."
    )

    content: list[dict] = [
        {"type": "text", "text": archive_text + live_text + sensor_text},
    ]

    # Include the actual image frame if available
    if frame.image_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame.image_b64}"},
        })

    return content


# ── Response Parsing ─────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_response(raw: str) -> tuple[str, EpistemicDecision | None, str]:
    """Extract think trace and validated decision from raw NIM output.

    Returns: (think_trace, decision_or_None, error_string)
    """
    # Extract think trace
    think_match = _THINK_RE.search(raw)
    think_trace = think_match.group(1).strip() if think_match else ""

    # Extract JSON after </think>
    # Look for JSON after the think block, or anywhere if no think block
    search_area = raw[think_match.end():] if think_match else raw
    json_match = _JSON_RE.search(search_area)
    if not json_match:
        # Try the whole string
        json_match = _JSON_RE.search(raw)

    if not json_match:
        return think_trace, None, "No JSON object found in response"

    try:
        data = json.loads(json_match.group())
        decision = EpistemicDecision(**data)
        return think_trace, decision, ""
    except (json.JSONDecodeError, ValueError) as e:
        return think_trace, None, str(e)


# ── Async Dispatcher ─────────────────────────────────────────────────────────

async def dispatch_agent(
    session: aiohttp.ClientSession,
    agent: AgentState,
    frame: FrameData,
    live_kv_window: list[str],
    archive_memories: list[ArchiveMemory],
) -> AgentResponse:
    """Send a single agent query to NIM and return parsed response."""
    t0 = time.monotonic()

    system_prompt = _build_system_prompt(agent)
    user_content = _build_user_content(frame, live_kv_window, archive_memories, agent)

    payload = {
        "model": NIM_ACTIVE_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": NIM_TEMPERATURE,
        "max_tokens": NIM_MAX_TOKENS,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {NGC_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            async with session.post(
                f"{NIM_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=NIM_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    if attempt < MAX_RETRIES:
                        log.warning(
                            "Agent %s attempt %d: HTTP %d — %s",
                            agent.name, attempt, resp.status, error_body[:200],
                        )
                        # Bump temperature on retry
                        payload["temperature"] = min(0.5, payload["temperature"] + 0.1)
                        await asyncio.sleep(1)
                        continue
                    return AgentResponse(
                        agent_idx=agent.agent_idx,
                        agent_name=agent.name,
                        frame_idx=frame.frame_idx,
                        decision=None,
                        parse_error=f"HTTP {resp.status}: {error_body[:200]}",
                        latency_ms=(time.monotonic() - t0) * 1000,
                    )

                data = await resp.json()
                raw_output = data["choices"][0]["message"]["content"]

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < MAX_RETRIES:
                log.warning("Agent %s attempt %d: %s", agent.name, attempt, e)
                await asyncio.sleep(1)
                continue
            return AgentResponse(
                agent_idx=agent.agent_idx,
                agent_name=agent.name,
                frame_idx=frame.frame_idx,
                decision=None,
                parse_error=str(e),
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        # Parse response
        think_trace, decision, error = _parse_response(raw_output)

        if decision is None and attempt < MAX_RETRIES:
            log.warning("Agent %s parse fail attempt %d: %s", agent.name, attempt, error)
            payload["temperature"] = min(0.5, payload["temperature"] + 0.1)
            continue

        return AgentResponse(
            agent_idx=agent.agent_idx,
            agent_name=agent.name,
            frame_idx=frame.frame_idx,
            decision=decision,
            think_trace=think_trace,
            raw_output=raw_output,
            latency_ms=(time.monotonic() - t0) * 1000,
            parse_error=error,
        )

    # Should not reach here, but just in case
    return AgentResponse(
        agent_idx=agent.agent_idx,
        agent_name=agent.name,
        frame_idx=frame.frame_idx,
        decision=None,
        parse_error="Exhausted retries",
        latency_ms=(time.monotonic() - t0) * 1000,
    )


async def dispatch_all_agents(
    session: aiohttp.ClientSession,
    agents: list[AgentState],
    frame: FrameData,
    live_kv_windows: dict[int, list[str]],
    archive_memories: list[ArchiveMemory],
) -> list[AgentResponse]:
    """Dispatch all agents concurrently for a single frame."""
    tasks = [
        dispatch_agent(
            session, agent, frame,
            live_kv_windows.get(agent.agent_idx, []),
            archive_memories,
        )
        for agent in agents
    ]
    return await asyncio.gather(*tasks)
