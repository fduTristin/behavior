"""CPU regressions for the local stage-1 closed-loop serving adaptation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from g05.models.g05.g05_policy_memlite_planner_outcome import (
    G05PolicyMEMLitePlannerOutcome,
)
from g05.utils.memlite_planner_format import planner_only_format_constants
from scripts.serve_memlite_stage1_behavior import (
    _planner_token_budget,
    _target_free_planner_sample,
)
from scripts.smoke_official_client import synthetic_observation


def _training_shaped_sample() -> dict:
    return {
        "template": (
            "<task_name_text_!><previous_parent_goal_text_!><previous_intent_text_!>"
            "<memory_text_!><planner_known_previous_outcome_text_!>"
            "<execution_feedback_text_!><planner_prompt_text_!><EOC>"
            "<planner_outcome_target_text>|<planner_next_decision_text>|"
            "<current_parent_goal_text>|<planner_active_skills_semantic_json_text>|"
            "<planner_memory_update_text>|<planner_task_complete_text>|<HL_END>"
        ),
        "command": "turning on radio",
        "task_name": "turning on radio",
        "previous_parent_goal": "None",
        "previous_intent": "None",
        "memory": "{}",
        "known_previous_outcome": "UNKNOWN",
        "execution_feedback": "none",
        "planner_prompt": "plan",
        "proprio": {"value": None},
        "embodiment": "galaxea_r1pro",
        "schema_version": 6,
        "memlite_schema_version": 6,
        "memlite_branch": "high",
        "planner_known_previous_outcome": "Known previous outcome: UNKNOWN",
        "planner_outcome_target": "Previous outcome: UNKNOWN",
        "planner_next_decision": "Decision: STOP",
        "planner_active_skills_semantic_json": "Active skills: []",
        "planner_memory_update": "Memory update: {}",
        "planner_task_complete": "Task complete: true",
        "parent_goal": "Task goal: turning on radio",
        "target_parent_goal": "Task goal: turning on radio",
        "current_parent_goal": "Parent goal: Task goal: turning on radio",
        "current_parent_goal_value": "Task goal: turning on radio",
        "active_skills_semantic_json": "[]",
        "outcome_target": "UNKNOWN",
        "next_decision": "STOP",
        "memory_update": "{}",
        "task_complete": True,
    }


def test_target_free_sample_restores_prefix_contract_and_drops_targets():
    forbidden = G05PolicyMEMLitePlannerOutcome._TARGET_FREE_HIGH_FORBIDDEN_FIELDS
    sample = _target_free_planner_sample(_training_shaped_sample(), forbidden)
    assert sample["template"].endswith("<EOC>")
    assert "<known_previous_outcome_text_!>" in sample["template"]
    assert "<planner_known_previous_outcome_text_!>" not in sample["template"]
    assert sample["known_previous_outcome"] == "Known previous outcome: UNKNOWN"
    assert not (forbidden & set(sample))
    assert not any(key.startswith("planner_") and key != "planner_prompt" for key in sample)
    assert "current_parent_goal_value" not in sample


def test_target_free_sample_requires_an_eoc_boundary():
    sample = _training_shaped_sample()
    sample["template"] = "no boundary"
    with pytest.raises(ValueError, match="EOC"):
        _target_free_planner_sample(sample, frozenset())


def test_planner_budget_defaults_to_reviewed_config_and_rejects_drift():
    assert _planner_token_budget(1024, None) == 1024
    assert _planner_token_budget(1024, 1024) == 1024
    with pytest.raises(ValueError, match="configured budget"):
        _planner_token_budget(1024, 160)


def test_planner_format_receipt_and_tokenizer_agree():
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: 42)
    processor = SimpleNamespace(hl_end_token_id=42, tokenizer=tokenizer)
    with planner_only_format_constants(
        object(), processor, batch_size=1, hl_end_id=42, enabled=True
    ) as receipt:
        assert receipt == {
            "planner_only": True,
            "schema_version": 6,
            "batch_size": 1,
            "hl_end_id": 42,
            "hl_end_text": "<HL_END>",
        }


def test_planner_format_rejects_mismatched_stop_id():
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: 41)
    processor = SimpleNamespace(hl_end_token_id=42, tokenizer=tokenizer)
    with pytest.raises(ValueError, match="tokenizer maps"):
        with planner_only_format_constants(
            object(), processor, batch_size=1, hl_end_id=42, enabled=True
        ):
            pass


def test_smoke_observation_matches_official_wire_shapes():
    observation = synthetic_observation(84)
    assert observation["robot_r1::proprio"].shape == (61,)
    assert observation["task_id"].tolist() == [84]
    assert observation["robot_r1::robot_r1:zed_link:Camera:0::rgb"].shape == (
        720,
        720,
        4,
    )
    for key, value in observation.items():
        if key.endswith("::rgb"):
            assert value.dtype == np.uint8
