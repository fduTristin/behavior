"""CPU contract tests for the stage-1 serving runtime memory primitives."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from g05.data.memlite_stage1_labels import projection as training_projection
from g05.utils.memlite_skill_protocol import (
    MemLiteSkillProtocolError,
    MEMLITE_B_MEMORY_HISTORY_LIMIT,
    parse_b_memory_text,
)
from scripts.memlite_stage1_runtime import (
    advance_b_memory,
    initial_b_memory,
    initial_planner_state,
    load_stage1_tasks,
)


# Exact canonical bundle texts copied from the stage-1 v4 release
# (datasets/memlite-stage1-20260930-v4/episodes.jsonl segments).
_BUNDLES = [
    'Active skills: [verb="NAVIGATE"; target="radio_89"; source="NONE"; '
    'destination="NONE"; target_part="NONE"; arm="UNSPECIFIED"].',
    'Active skills: [verb="GRASP"; target="radio_89"; source="coffee_table_koagbh_0"; '
    'destination="NONE"; target_part="NONE"; arm="UNSPECIFIED"].',
    'Active skills: [verb="PRESS"; target="radio_89"; source="NONE"; '
    'destination="NONE"; target_part="NONE"; arm="UNSPECIFIED"].',
    'Active skills: [verb="PLACE_ON"; target="radio_89"; source="NONE"; '
    'destination="coffee_table_koagbh_0"; target_part="NONE"; arm="UNSPECIFIED"].',
    'Active skills: [verb="NAVIGATE"; target="coffee_table_koagbh_0"; source="NONE"; '
    'destination="NONE"; target_part="NONE"; arm="UNSPECIFIED"].',
]


class TestStage1RuntimeMemory(unittest.TestCase):
    def test_tasks_asset_covers_all_100_official_ids(self):
        tasks = load_stage1_tasks()
        self.assertEqual(len(tasks), 100)
        self.assertEqual([t["task_index"] for t in tasks], list(range(100)))
        for task in tasks:
            self.assertTrue(task["task_name"])
            self.assertTrue(task["official_task"])
            # Pinned memory in the asset must match the generative formula.
            self.assertEqual(task["initial_memory"], initial_b_memory(task["task_name"]))

    def test_initial_memory_byte_identical_to_training_projection(self):
        tasks = load_stage1_tasks()
        task = tasks[0]
        # Training-side episode-start anchor: first segment, no prior bundle.
        semester = {
            "parent": f"Parent command: description=\"pick up from\"; targets=[\"radio_89\"]; "
                      f"sources=[\"coffee_table_koagbh_0\"]; destinations=[]; "
                      f"target_parts=[]; arms=[]",
            "semantic": "[]",
            "text": _BUNDLES[0],
            "parent_supervised": False,
        }
        train = training_projection(
            semester,
            branch="high",
            task_name=task["task_name"],
            previous_intent="None",
            previous_parent="None",
            history=[],
        )
        runtime = initial_planner_state(task["task_name"])
        self.assertEqual(runtime["memory"], train["memory"])
        self.assertEqual(runtime["previous_parent_goal"], train["previous_parent_goal"])
        self.assertEqual(runtime["previous_intent"], train["previous_intent"])
        self.assertEqual(runtime["known_previous_outcome"], train["known_previous_outcome"])
        self.assertEqual(runtime["execution_feedback"], train["execution_feedback"])

    def test_advance_is_idempotent_for_same_bundle(self):
        task = load_stage1_tasks()[0]
        memory = initial_b_memory(task["task_name"])
        bundle = _BUNDLES[0]
        once = advance_b_memory(memory, bundle, task["task_name"])
        twice = advance_b_memory(once, bundle, task["task_name"])
        self.assertEqual(once, twice)
        parsed = parse_b_memory_text(once, task_name=task["task_name"])
        self.assertEqual(len(parsed["issued_command_history"]), 1)

    def test_history_is_capped_at_k3(self):
        task = load_stage1_tasks()[0]
        memory = initial_b_memory(task["task_name"])
        for bundle in _BUNDLES:
            memory = advance_b_memory(memory, bundle, task["task_name"])
        parsed = parse_b_memory_text(memory, task_name=task["task_name"])
        self.assertEqual(
            len(parsed["issued_command_history"]), MEMLITE_B_MEMORY_HISTORY_LIMIT
        )

    def test_legacy_a_memory_format_is_rejected(self):
        with self.assertRaises(MemLiteSkillProtocolError):
            parse_b_memory_text("Task=0; Completed=none.")

    def test_initial_none_sentinel_is_a_noop_commit(self):
        # Matches the training-side guard in memlite_stage1_labels.anchor_records:
        # the "None" initial sentinel is never appended to the issued history.
        task = load_stage1_tasks()[0]
        memory = initial_b_memory(task["task_name"])
        self.assertEqual(advance_b_memory(memory, "None", task["task_name"]), memory)


if __name__ == "__main__":
    unittest.main()
