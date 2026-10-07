# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
"""Official BEHAVIOR-protocol server for the stage-1 MEM-Lite checkpoints.

This is the "full service" entry point the official v3.9.x websocket evaluator
connects to.  It wires three independently published pieces together:

1. the vendored official bridge (``scripts/behavior_bridge/
   serve_behavior_policy_mem.py``) for the wire protocol — 61-dim proprio
   slicing, three-camera CHW uint8, official 23-dim action vector, fire-and
   -forget reset, ``/healthz`` and the official bytes-key msgpack encoding;
2. the stage-1 serving runtime (``scripts/memlite_stage1_runtime.py``) for the
   causal B-memory state machine — initial memory, K=3 recurrence admission
   and the audit-free model projections, byte-identical to
   ``g05.data.memlite_stage1_labels``;
3. the stage-1 checkpoint loader (``scripts/serve_memlite_stage1.py``) rebuilt
   from the frozen high/low training recipes.

Loop semantics (one websocket connection = one episode):

* reset (``{"reset": true}``/``{"__reset__": ...}``) or a task_id change starts
  a fresh B-memory state; nothing carries across the boundary.
* Training anchors replan the high planner every 16-frame chunk, so every time
  the served action chunk empties we run the planner on the current
  observation, validate + commit the event (grammar + K=3 recurrence +
  planner_only terminal rejection are enforced by the policy and again by the
  runtime), then run the low FM controller under the committed bundle and
  serve its first ``action_steps`` steps.
* Any admission failure, garbage generation or non-finite action raises; the
  evaluator connection fails loudly instead of accepting a degraded action
  (strict mode like the upstream bridge's ``require_memlite=True``).
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _extra in (str(REPO), str(REPO / "src"), str(SCRIPTS_DIR)):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

import websockets  # noqa: E402

from g05.models.g05.inferencer import PolicyInferencer  # noqa: E402
from memlite_stage1_runtime import (  # noqa: E402
    Stage1PlannerSession,
    load_stage1_tasks,
)
from serve_memlite_stage1 import ORIGINAL_ROOT, load_stage1  # noqa: E402
from serve_policy_memlite_fm import project_fm_grippers  # noqa: E402

logger = logging.getLogger("serve_memlite_stage1_behavior")


def load_bridge(directory: Path):
    """Import the vendored official bridge without touching its filesystem."""
    module_path = directory / "serve_behavior_policy_mem.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"bridge module not found: {module_path}")
    spec = importlib.util.spec_from_file_location("stage1_behavior_bridge", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strip_target_free_forbidden(samples: list[dict], forbidden: frozenset) -> list[dict]:
    """Drop unaliased raw target names before the target-free planner prefix.

    Training-placeholders after ``<EOC>`` are kept under ``planner_*`` aliases
    by ``bind_planner_rendered_fields``; the raw names exist only to satisfy
    the schema validator and must never reach ``generate_high_level`` (its own
    ``_validate_target_free_high_prefix`` rejects them).
    """
    clean = []
    for sample in samples:
        item = dict(sample)
        for key in forbidden:
            item.pop(key, None)
        clean.append(item)
    return clean


class Stage1BehaviorPolicy:
    """One high planner + one low FM controller under the official wire."""

    def __init__(
        self,
        *,
        high_policy,
        high_processor,
        low_policy,
        low_processor,
        shape_meta: dict,
        device: str,
        action_steps: int = 16,
        max_new_tokens: int = 160,
        tasks=None,
    ):
        from g05.models.g05.g05_policy_memlite_planner_outcome import (
            G05PolicyMEMLitePlannerOutcome,
        )

        if not isinstance(high_policy, G05PolicyMEMLitePlannerOutcome):
            raise TypeError("official serving requires the planner-outcome high policy")
        if not high_policy.planner_only:
            raise ValueError("official stage-1 serving requires planner_only=True")
        self.high = PolicyInferencer(high_policy, high_processor, device=device)
        self.low = PolicyInferencer(low_policy, low_processor, device=device)
        self.high_forbidden = G05PolicyMEMLitePlannerOutcome._TARGET_FREE_HIGH_FORBIDDEN_FIELDS
        self.shape_meta = shape_meta
        self.device = device
        if int(action_steps) < 1:
            raise ValueError("action_steps must be >= 1")
        self.action_steps = int(action_steps)
        self.max_new_tokens = int(max_new_tokens)
        self.tasks = load_stage1_tasks() if tasks is None else tasks

        self.session: Stage1PlannerSession | None = None
        self.chunk: list[dict[str, np.ndarray]] = []
        self.served_action_count = 0
        self.planner_generation = 0

    # ── episode state ────────────────────────────────────────────────────

    def reset(self, reason: str = "reset") -> None:
        logger.info(
            "STAGE1_STATE_RESET reason=%s served_actions=%d planner_gens=%d",
            reason,
            self.served_action_count,
            self.planner_generation,
        )
        self.session = None
        self.chunk = []
        self.served_action_count = 0
        self.planner_generation = 0

    def _ensure_session(self, task_id: int) -> Stage1PlannerSession:
        if self.session is not None and self.session.task_index != int(task_id):
            self.reset(reason=f"task_switch:{self.session.task_index}->{task_id}")
        if self.session is None:
            self.session = Stage1PlannerSession.for_task_id(task_id, self.tasks)
            logger.info(
                "STAGE1_EPISODE_START task_id=%d task_name=%r",
                self.session.task_index,
                self.session.task_name,
            )
        return self.session

    def is_fresh_bundle_needed(self) -> bool:
        return not self.chunk

    # ── inference ────────────────────────────────────────────────────────

    def _plan_once(self, g05_obs: dict) -> dict:
        """Generate, validate and commit one planner event (sync, GPU)."""
        session = self.session
        assert session is not None
        raw = session.raw_observation(g05_obs, self.shape_meta, session.high_projection())
        _, batch = self.high._prepare_branch_batch([raw])
        if "samples" not in batch or "pixel_values" not in batch:
            raise RuntimeError("high processor output lacks samples/pixel_values")
        samples = _strip_target_free_forbidden(
            [dict(s) for s in batch["samples"]], self.high_forbidden
        )
        batch = self.high._branch_device(batch)
        tokens_before = None
        with torch.no_grad(), self.high._branch_context():
            result = self.high.policy.generate_high_level(
                samples=samples,
                pixel_values=batch["pixel_values"],
                memory_text=session.memory,
                max_new_tokens=self.max_new_tokens,
            )
        events = result.get("planner_events", [])
        if len(events) != 1:
            raise RuntimeError(f"planner must return exactly one event, got {len(events)}")
        event = dict(events[0])
        raw_texts = result.get("high_level_text", [])
        raw_text = str(raw_texts[0]) if raw_texts else ""
        generated_ids = result.get("high_level_generated_ids")
        if isinstance(generated_ids, torch.Tensor):
            tokens_before = int(generated_ids.shape[1])
        session.commit(event)
        self.planner_generation += 1
        logger.info(
            "STAGE1_PLANNER_EVENT gen=%d parent=%r bundle=%r decision=%s tokens=%s raw=%r",
            self.planner_generation,
            event["parent_goal"],
            event["active_skills_text"],
            event["decision"],
            tokens_before,
            raw_text[:300],
        )
        return event

    def _low_chunk(self, g05_obs: dict) -> None:
        """Run one low FM chunk and queue its first ``action_steps`` steps."""
        session = self.session
        assert session is not None
        raw = session.raw_observation(g05_obs, self.shape_meta, session.low_projection())
        results, timing = self.low.infer_with_timing([raw])
        if len(results) != 1:
            raise RuntimeError(f"low FM must return exactly one action, got {len(results)}")
        action = project_fm_grippers(results[0])
        diagnostics = action.pop("_normalization_diagnostics", {})
        steps: list[dict[str, np.ndarray]] = []
        per_key: dict[str, np.ndarray] = {}
        horizon = None
        for key, value in action.items():
            if key.startswith("_"):
                continue
            if isinstance(value, torch.Tensor):
                array = value.detach().cpu().numpy()
            else:
                array = np.asarray(value)
            if array.ndim == 3:  # [1, H, D] postprocess output
                array = array[0]
            if array.ndim != 2:
                raise ValueError(f"low action {key!r} must be [H,D], got {array.shape}")
            horizon = array.shape[0] if horizon is None else horizon
            if array.shape[0] != horizon:
                raise ValueError("low action groups have inconsistent horizons")
            per_key[key] = array
        if horizon is None or horizon < 1:
            raise ValueError("low FM returned an empty action horizon")
        for step in range(min(self.action_steps, horizon)):
            steps.append(
                {key: per_key[key][step].astype(np.float32, copy=True) for key in per_key}
            )
        self.chunk = steps
        logger.info(
            "STAGE1_LOW_CHUNK chunk=%d horizon=%d serve=%d timing=%s",
            session.chunk_count if self.session else -1,
            horizon,
            len(steps),
            {k: round(v, 1) for k, v in timing.items() if isinstance(v, float)},
        )
        if diagnostics:
            logger.info("STAGE1_LOW_DIAGNOSTICS %s", json.dumps(diagnostics)[:400])

    async def get_action(self, g05_obs: dict, task_id: int) -> dict[str, np.ndarray]:
        """Serve the next official 23-dim action for one evaluator request."""
        session = self._ensure_session(task_id)
        g05_obs = dict(g05_obs)
        # Training conditions on the canonical task name, not the release
        # instruction sentence; keep on-distribution.
        g05_obs["task"] = session.task_name
        if self.is_fresh_bundle_needed():
            await asyncio.to_thread(self._plan_once, g05_obs)
            await asyncio.to_thread(self._low_chunk, g05_obs)
        if not self.chunk:
            raise RuntimeError("low chunk queue is empty after recompute")
        action = self.chunk.pop(0)
        self.served_action_count += 1
        return action


async def behavior_stage1_handler(websocket, *, policy: Stage1BehaviorPolicy, bridge, task_instructions) -> None:
    """One official evaluator connection with a fresh episode state."""
    client = websocket.remote_address
    logger.info("official evaluator connected: %s", client)
    await websocket.send(
        bridge.official_packb(
            {
                "policy": "G0.5",
                "embodiment": "R1Pro",
                "action_dim": 23,
                "action_steps": policy.action_steps,
            }
        )
    )
    policy.reset(reason="connection_open")
    try:
        async for message in websocket:
            try:
                payload = bridge.unpackb(message)
                if isinstance(payload, dict) and (
                    payload.get("reset") or payload.get("__reset__")
                ):
                    # The official client sends reset without waiting for a reply.
                    policy.reset(reason="episode_reset")
                    continue
                g05_obs = bridge.behavior_obs_to_g05(payload, task_instructions)
                task_values = np.asarray(payload["task_id"]).reshape(-1)
                task_id = int(task_values[0])
                t0 = time.monotonic()
                action_dict = await policy.get_action(g05_obs, task_id)
                action_dict = bridge.fill_missing_behavior_action(action_dict, g05_obs["state"])
                vector = bridge.behavior_action_to_vector(action_dict)
                await websocket.send(bridge.official_packb({"action": vector}))
                logger.info(
                    "STAGE1_ACTION served=%d task=%d l2=%.4f dt=%.1fms",
                    policy.served_action_count,
                    task_id,
                    float(np.linalg.norm(vector)),
                    (time.monotonic() - t0) * 1000.0,
                )
            except Exception:
                logger.error(
                    "stage-1 BEHAVIOR request failed (strict: closing connection)\n%s",
                    traceback.format_exc(),
                )
                # Strict mode: never let the evaluator accept a degraded or
                # text-frame action; fail the episode loudly instead.
                raise
    except websockets.exceptions.ConnectionClosed:
        logger.info("official evaluator disconnected: %s", client)
    finally:
        policy.reset(reason="connection_close")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path(ORIGINAL_ROOT),
                        help="deploy root with high/ low/ manifests/ models/")
    parser.add_argument("--high_ckpt", type=Path, default=None)
    parser.add_argument("--low_ckpt", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10100,
                        help="official evaluator connects here (default 10100)")
    parser.add_argument("--action_steps", type=int, default=16,
                        help="steps served per low FM chunk; training anchors replan every 16 frames")
    parser.add_argument("--max_new_tokens", type=int, default=160,
                        help="planner generation budget; <HL_END> rejection applies above it")
    parser.add_argument("--bridge_dir", type=Path,
                        default=SCRIPTS_DIR / "behavior_bridge")
    parser.add_argument("--tasks_path", type=Path, default=None,
                        help="official meta/tasks.jsonl for wire-side task text "
                             "(policy conditions on canonical names from memlite_stage1_tasks.json); "
                             "defaults to the staged challenge-demos copy under --root")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    bridge = load_bridge(args.bridge_dir)
    tasks_path = args.tasks_path
    if tasks_path is None:
        candidates = [
            args.bridge_dir / "tasks.jsonl",  # vendored official copy (git)
            Path(ORIGINAL_ROOT) / (
                "datasets/2026-challenge-demos/datasets/fduTristin--2026-challenge-demos/"
                "snapshots/master/meta/tasks.jsonl"
            ),  # legacy staged location on the training cluster
        ]
        for candidate in candidates:
            if candidate.is_file():
                tasks_path = candidate
                break
        else:
            raise FileNotFoundError(
                "tasks.jsonl not found in " + "; ".join(str(c) for c in candidates)
            )
    task_instructions = bridge.load_task_instructions(tasks_path)
    logger.info("loaded %d official task instructions from %s", len(task_instructions), tasks_path)

    logger.info("loading high planner + low FM policy from %s", args.root)
    high_policy, high_processor = load_stage1(args.root, "high", args.high_ckpt, args.device)
    low_policy, low_processor = load_stage1(args.root, "low", args.low_ckpt, args.device)
    shape_meta = json.loads(
        (args.root / "high" / "recipe.json").read_text()
    )["model"]["raw_shape"]
    policy = Stage1BehaviorPolicy(
        high_policy=high_policy,
        high_processor=high_processor,
        low_policy=low_policy,
        low_processor=low_processor,
        shape_meta=shape_meta,
        device=args.device,
        action_steps=args.action_steps,
        max_new_tokens=args.max_new_tokens,
    )

    import functools

    handler = functools.partial(
        behavior_stage1_handler,
        policy=policy,
        bridge=bridge,
        task_instructions=task_instructions,
    )
    logger.info("official stage-1 service listening on ws://%s:%d (healthz GET /healthz)",
                args.host, args.port)

    async def _run():
        async with websockets.serve(
            handler,
            args.host,
            args.port,
            max_size=None,
            ping_interval=None,
            process_request=bridge._health_check,
        ):
            await asyncio.Future()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
