"""CPU contract tests for the stage-1 serving runtime memory primitives."""

from __future__ import annotations

import json
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
    Stage1PlannerSession,
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


PARENT_FIXTURE = (
    'Parent command: description="pick up from"; targets=["radio_89"]; '
    'sources=["coffee_table_koagbh_0"]; destinations=[]; target_parts=[]; arms=[]'
)


class TestStage1PlannerSession(unittest.TestCase):
    """Contract tests for the serving-side planner episode state."""

    def _event(self, bundle: str, parent: str = PARENT_FIXTURE) -> dict:
        from g05.utils.memlite_skill_protocol import canonical_json, parse_active_skills_text

        semantic = parse_active_skills_text(bundle, allow_empty=False)
        return {
            "decision": "EXECUTE",
            "parent_goal": parent,
            "active_skills_text": bundle,
            "active_skills_semantic_json": canonical_json(semantic),
            "memory_update": None,  # filled below before commit
            "task_complete_claimed": False,
        }

    def test_high_projection_passes_model_safe_v6_label(self):
        session = Stage1PlannerSession(0, "turning on radio")
        projection = session.high_projection()
        from g05.data_processor.processor.memlite_v6_projection import _model_safe_v6_label

        label = _model_safe_v6_label(projection)
        self.assertEqual(label["memlite_branch"], "high")
        self.assertTrue(label["task_complete"])  # terminal placeholder shape

    def test_commit_advances_causal_state(self):
        session = Stage1PlannerSession(0, "turning on radio")
        first = self._event(_BUNDLES[0])
        first["memory_update"] = advance_b_memory(session.memory, session.previous_intent, session.task_name)
        session.commit(first)  # previous_intent "None" is a no-op commit
        self.assertEqual(session.current_bundle_text, _BUNDLES[0])
        self.assertEqual(session.memory, initial_b_memory(session.task_name))

        low = session.low_projection()
        from g05.data_processor.processor.memlite_v6_projection import _model_safe_v6_label

        label = _model_safe_v6_label(low)
        self.assertEqual(label["memlite_branch"], "low")
        self.assertEqual(label["next_decision"], "EXECUTE")
        self.assertTrue(label["low_action_supervision_mask"])

        second = self._event(_BUNDLES[1])
        second["memory_update"] = advance_b_memory(session.memory, session.previous_intent, session.task_name)
        session.commit(second)
        parsed = parse_b_memory_text(session.memory, task_name=session.task_name)
        self.assertEqual(parsed["issued_command_history"], [_BUNDLES[0]])

    def test_commit_rejects_memory_recurrence_violation(self):
        session = Stage1PlannerSession(0, "turning on radio")
        event = self._event(_BUNDLES[0])
        event["memory_update"] = initial_b_memory("picking up trash")
        with self.assertRaises(ValueError):
            session.commit(event)

    def test_commit_rejects_terminal_claim(self):
        session = Stage1PlannerSession(0, "turning on radio")
        event = self._event(_BUNDLES[0])
        event["memory_update"] = advance_b_memory(session.memory, session.previous_intent, session.task_name)
        event["task_complete_claimed"] = True
        with self.assertRaises(ValueError):
            session.commit(event)

    def test_raw_observation_matches_dataset_layout(self):
        import numpy as np
        import torch

        shape_meta = {
            "images": [
                {"key": "head_rgb", "raw_shape": (3, 4, 4)},
                {"key": "left_wrist_rgb", "raw_shape": (3, 4, 4)},
            ],
            "state": [{"key": "left_arm", "start_index": 3, "raw_shape": (7,)}],
            # recipe.json serializes state/action raw_shape as a scalar.
            "action": [{"key": "lower_body", "start_index": 0, "raw_shape": 7}],
        }
        g05_obs = {
            "images": {
                "head_rgb": np.full((3, 4, 4), 255, dtype=np.uint8),
                "left_wrist_rgb": np.zeros((3, 4, 4), dtype=np.uint8),
            },
            "state": {"left_arm": np.arange(7, dtype=np.float32)},
        }
        session = Stage1PlannerSession(0, "turning on radio")
        raw = session.raw_observation(g05_obs, shape_meta, session.high_projection())
        self.assertEqual(raw["images"]["head_rgb"].shape, (1, 3, 4, 4))
        self.assertEqual(raw["images"]["head_rgb"].dtype, torch.uint8)
        self.assertTrue((raw["images"]["head_rgb"] == 255).all())
        self.assertEqual(raw["state"]["left_arm"].shape, (1, 7))
        self.assertEqual(raw["action"]["lower_body"].shape, (32, 7))
        self.assertTrue(raw["action_is_pad"].all())
        self.assertEqual(raw["task"], "turning on radio")
        self.assertEqual(raw["embodiment"], "galaxea_r1pro")


PARENT_FIXTURE = (
    'Parent command: description="pick up from"; targets=["radio_89"]; '
    'sources=["coffee_table_koagbh_0"]; destinations=[]; target_parts=[]; arms=[]'
)


class TestOfficialTaskTableAlignment(unittest.TestCase):
    """The vendored official instruction table must agree with the runtime asset."""

    def test_vendored_tasks_jsonl_matches_stage1_asset(self):
        table_path = (
            Path(__file__).resolve().parent.parent
            / "scripts/behavior_bridge/tasks.jsonl"
        )
        rows = [json.loads(line) for line in table_path.open() if line.strip()]
        tasks = load_stage1_tasks()
        self.assertEqual(len(rows), 100)
        self.assertEqual([r["task_index"] for r in rows], list(range(100)))
        for row, entry in zip(rows, tasks):
            self.assertEqual(row["task_index"], entry["task_index"])
            self.assertEqual(row["task_name"].replace("_", " "), entry["task_name"])
            self.assertTrue(row["task"])

    def test_official_84_is_tidying_bathroom(self):
        """Pinned probe for the id-to-name mapping the team once suspected."""
        tasks = load_stage1_tasks()
        self.assertEqual(tasks[84]["task_name"], "tidying bathroom")


if __name__ == "__main__":
    unittest.main()
