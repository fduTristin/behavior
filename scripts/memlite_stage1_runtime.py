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
    append_b_memory_idempotent,
    canonical_json,
    validate_b_memory_text,
)

DEFAULT_TASKS_PATH = Path(__file__).resolve().parent / "memlite_stage1_tasks.json"


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


__all__ = [
    "DEFAULT_TASKS_PATH",
    "MEMLITE_B_MEMORY_HISTORY_LIMIT",
    "advance_b_memory",
    "initial_b_memory",
    "initial_planner_state",
    "load_stage1_tasks",
]
