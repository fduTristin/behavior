# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
"""Stage-1 planner B-memory runtime primitives for official BEHAVIOR serving.

Training-side ground truth lives in ``g05.data.memlite_stage1_labels``:
every planner anchor row builds its memory as

    canonical_json(dict(task_name=task_name, issued_command_history=history,
                        verified_world_facts=[]))

with ``append_b_memory_idempotent`` as the only allowed state transition.
This module mirrors exactly that logic for the serving runtime, so the
solver-side projection is byte-identical to training and covers all 100
released BEHAVIOR tasks (the legacy ``Task=<id>; Completed=none.`` strings
in ``scripts/behavior_bridge/serve_behavior_policy_mem.py`` belong to the
old 5-task v9 protocol and must NOT be used with stage-1 checkpoints).

The companion asset ``scripts/memlite_stage1_tasks.json`` pins the exact
canonical task names from the stage-1 v4 data-release manifest
(``datasets/memlite-stage1-20260930-v4/manifest.json``), checked 1:1
against the official ``meta/tasks.jsonl`` of the 2026 challenge demos.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from g05.utils.memlite_skill_protocol import (
    MEMLITE_B_MEMORY_HISTORY_LIMIT,
    MEMLITE_SKILL_SCHEMA_VERSION,
    append_b_memory_idempotent,
    canonical_json,
    validate_b_memory_text,
)

DEFAULT_TASKS_PATH = Path(__file__).resolve().parent / "memlite_stage1_tasks.json"

HIGH_ACTION_HORIZON = 32


def stop_high_placeholders(task_name: str) -> dict[str, Any]:
    """Validation-only target-region values for a target-free high prefix.

    Everything here sits strictly after ``<EOC>`` and is stripped from the
    planner prefix by the policy (see ``PolicyInferencer`` /
    ``G05PolicyMEMLitePlannerOutcome`` docs).  The empty terminal shape is the
    one legal placeholder that keeps ``_model_safe_v6_label`` satisfied
    without fabricating a skill bundle: an empty high projection must be
    terminal ``STOP`` per the published schema validator.
    """
    parent = f"Task goal: {task_name}"
    return {
        "parent_goal": parent,
        "target_parent_goal": parent,
        "active_skills_semantic_json": canonical_json([]),
        "active_skills_text": "Active skills: none.",
        "next_decision": "STOP",
        "task_complete": True,
        "outcome_target": "UNKNOWN",
        "outcome_supervision_mask": False,
        "parent_goal_supervision_mask": False,
        "low_action_supervision_mask": False,
    }


def load_stage1_tasks(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the 100 canonical stage-1 tasks ordered by official zero-based id."""
    tasks_path = Path(path) if path is not None else DEFAULT_TASKS_PATH
    with tasks_path.open("r", encoding="utf-8") as handle:
        tasks = json.load(handle)
    if [int(t["task_index"]) for t in tasks] != list(range(len(tasks))):
        raise ValueError(f"{tasks_path} must list tasks in task_index order")
    for task in tasks:
        # Refuse a drifted asset: each pinned memory must still validate.
        validate_b_memory_text(task["initial_memory"], task_name=task["task_name"])
    return tasks


def initial_b_memory(task_name: str) -> str:
    """Canonical planner memory for a fresh episode of ``task_name``.

    Byte-identical to the training-side projection in
    ``memlite_stage1_labels.projection`` when the anchor is the episode's
    first planner event (empty issued-command history).
    """
    memory = canonical_json(
        dict(task_name=task_name, issued_command_history=[], verified_world_facts=[])
    )
    return validate_b_memory_text(memory, task_name=task_name)


def advance_b_memory(memory_text: str, issued_intent_text: str, task_name: str) -> str:
    """Commit an already-issued bundle through the published K=3 transition.

    Semantics copied from training: the event appended is the *previous*
    bundle that was actually issued to the low controller; a same-bundle
    refresh keeps the ledger idempotent.
    """
    updated = append_b_memory_idempotent(
        memory_text, issued_intent_text, task_name=task_name
    )
    return validate_b_memory_text(updated, task_name=task_name)


def initial_planner_state(task_name: str) -> dict[str, str]:
    """Full initial serving-side sibling fields of a stage-1 planner projection.

    Mirrors ``memlite_stage1_labels.projection`` for the episode start:
    planner sentinels are the exact string ``"None"`` and there is no
    observable execution feedback yet.
    """
    return {
        "task_name": task_name,
        "previous_parent_goal": "None",
        "previous_intent": "None",
        "memory": initial_b_memory(task_name),
        "known_previous_outcome": "UNKNOWN",
        "execution_feedback": "none",
    }


class Stage1PlannerSession:
    """Per-connection planner episode state for official BEHAVIOR serving.

    Owns the causal B-memory state of one episode.  ``commit`` validates the
    planner event against the recurrence *before* the low controller is fed;
    a model that violates the K=3 memory grammar corners an exception instead
    of silently executing a corrupt bundle.

    Training anchors replan every 16-frame action chunk
    (``memlite_stage1_labels.anchor_records``), so the serving default is one
    planner event per executed low-level chunk.
    """

    def __init__(self, task_index: int, task_name: str):
        self.task_index = int(task_index)
        self.task_name = str(task_name)
        state = initial_planner_state(self.task_name)
        self.previous_parent_goal = state["previous_parent_goal"]
        self.previous_intent = state["previous_intent"]
        self.memory = state["memory"]
        self.known_previous_outcome = state["known_previous_outcome"]
        self.execution_feedback = state["execution_feedback"]
        # Committed bundle currently driving the low controller.
        self.current_parent_goal: str | None = None
        self.current_bundle_text: str | None = None
        self.current_bundle_semantic_json: str | None = None
        self.chunk_count = 0

    @classmethod
    def for_task_id(cls, task_id: int, tasks: list[dict[str, Any]] | None = None):
        table = load_stage1_tasks() if tasks is None else tasks
        for entry in table:
            if int(entry["task_index"]) == int(task_id):
                return cls(entry["task_index"], entry["task_name"])
        raise ValueError(f"Unknown official BEHAVIOR task_id: {task_id}")

    # ── projections ──────────────────────────────────────────────────────

    def high_projection(self) -> dict[str, Any]:
        """Target-free serving projection for the next planner event."""
        projection: dict[str, Any] = {
            "schema_version": MEMLITE_SKILL_SCHEMA_VERSION,
            "memlite_branch": "high",
            "task_name": self.task_name,
            "previous_parent_goal": self.previous_parent_goal,
            "previous_intent": self.previous_intent,
            "memory": self.memory,
            "known_previous_outcome": self.known_previous_outcome,
            "execution_feedback": self.execution_feedback,
            # The recurrence target is protocol-derived, never model-derived.
            "memory_update": advance_b_memory(
                self.memory, self.previous_intent, self.task_name
            ),
        }
        projection.update(stop_high_placeholders(self.task_name))
        return projection

    def low_projection(self) -> dict[str, Any]:
        """Conditioning projection for one low FM chunk under the live bundle."""
        if not self.current_bundle_text or not self.current_parent_goal:
            raise RuntimeError("low projection requires a committed planner event")
        projection: dict[str, Any] = {
            "schema_version": MEMLITE_SKILL_SCHEMA_VERSION,
            "memlite_branch": "low",
            "task_name": self.task_name,
            "parent_goal": self.current_parent_goal,
            "target_parent_goal": self.current_parent_goal,
            "previous_parent_goal": self.previous_parent_goal,
            "previous_intent": self.previous_intent,
            "memory": self.memory,
            "known_previous_outcome": self.known_previous_outcome,
            "execution_feedback": self.execution_feedback,
            "active_skills_semantic_json": self.current_bundle_semantic_json,
            "active_skills_text": self.current_bundle_text,
            "next_decision": "EXECUTE",
            "task_complete": False,
            "memory_update": self.memory,
            "outcome_target": "UNKNOWN",
            "outcome_supervision_mask": False,
            "parent_goal_supervision_mask": False,
            "low_action_supervision_mask": True,
        }
        return projection

    def raw_observation(self, g05_obs: dict[str, Any], shape_meta: dict[str, Any],
                        projection: dict[str, Any]) -> dict[str, Any]:
        """Bridge-converted obs -> the dataset-style raw record.

        Byte-layout mirrors ``memlite_stage1_dataset.Stage1Dataset.raw``:
        uint8 images [1,H,W,3], float32 state [|obs|,dim], a zero action with a
        fully padded horizon, plus the serving-side model projection.
        """
        import numpy as np
        import torch

        images = {}
        for meta in shape_meta["images"]:
            key = meta["key"]
            chw = np.asarray(g05_obs["images"][key])
            if chw.ndim != 3 or chw.shape[0] != 3:
                raise ValueError(f"{key} must be CHW uint8, got {chw.shape}")
            hwc = np.ascontiguousarray(chw.transpose(1, 2, 0))
            images[key] = torch.from_numpy(hwc).unsqueeze(0)
        state = {
            meta["key"]: torch.from_numpy(
                np.asarray(g05_obs["state"][meta["key"]], dtype=np.float32)
            ).unsqueeze(0)
            for meta in shape_meta["state"]
        }
        action = {
            meta["key"]: torch.zeros(
                HIGH_ACTION_HORIZON, int(meta["raw_shape"][-1]),
            )
            for meta in shape_meta["action"]
        }
        return {
            "idx": 0,
            "task": self.task_name,
            "embodiment": "galaxea_r1pro",
            "images": images,
            "state": state,
            "action": action,
            "action_is_pad": torch.ones(HIGH_ACTION_HORIZON, dtype=torch.bool),
            "state_is_pad": torch.zeros(1, dtype=torch.bool),
            "image_is_pad": torch.zeros(1, dtype=torch.bool),
            "frequency": 30,
            "model_projection": projection,
        }

    # ── state transition ─────────────────────────────────────────────────

    def commit(self, event: dict[str, Any]) -> None:
        """Commit one validated planner event through the published transition."""
        if event.get("decision") != "EXECUTE":
            # planner_only policies always route this through admission first;
            # repeat the guard so the runtime never fake-executes a terminal.
            raise ValueError(
                f"planner event decision {event.get('decision')!r} not executable; "
                "refusing to dispatch a bundle"
            )
        if event.get("task_complete_claimed"):
            raise ValueError("task completion is never an AR claim; refusing event")
        expected_memory = advance_b_memory(
            self.memory, self.previous_intent, self.task_name
        )
        if event.get("memory_update") != expected_memory:
            raise ValueError("planner event violates causal K=3 memory recurrence")
        self.previous_intent = event["active_skills_text"]
        self.previous_parent_goal = event["parent_goal"]
        self.memory = expected_memory
        self.current_parent_goal = event["parent_goal"]
        self.current_bundle_text = event["active_skills_text"]
        self.current_bundle_semantic_json = event["active_skills_semantic_json"]
        self.chunk_count += 1


__all__ = [
    "DEFAULT_TASKS_PATH",
    "HIGH_ACTION_HORIZON",
    "MEMLITE_B_MEMORY_HISTORY_LIMIT",
    "Stage1PlannerSession",
    "advance_b_memory",
    "initial_b_memory",
    "initial_planner_state",
    "load_stage1_tasks",
    "stop_high_placeholders",
]
