"""CPU-only tests for the G0.5 <-> BEHAVIOR protocol adapter."""

from __future__ import annotations

import asyncio
import functools
import importlib
import json
import msgpack
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import websockets

import serve_behavior_policy_mem as adapter


def official_pack_data(value: Any) -> Any:
    """Exact ndarray wire format used by OmniGibson v3.9.x."""
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def official_unpack_data(value: dict[Any, Any]) -> Any:
    """Exact ndarray decoder used by OmniGibson v3.9.x."""
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


official_packb = functools.partial(msgpack.packb, default=official_pack_data)
official_unpackb = functools.partial(msgpack.unpackb, object_hook=official_unpack_data)


def make_observation(task_id: int = 0) -> dict[str, Any]:
    proprio = np.arange(61, dtype=np.float32)
    return {
        "robot_r1::proprio": proprio,
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.zeros(
            (12, 16, 4), dtype=np.uint8
        ),
        "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": np.zeros(
            (10, 14, 3), dtype=np.uint8
        ),
        "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": np.zeros(
            (3, 8, 6), dtype=np.uint8
        ),
        "task_id": np.asarray([task_id], dtype=np.int64),
    }


def test_mapping() -> None:
    tasks = {0: "Turn on the radio."}
    converted = adapter.behavior_obs_to_g05(make_observation(), tasks)
    assert converted["task"] == tasks[0]
    assert converted["embodiment_type"] == "galaxea_r1pro"
    assert converted["frequency"] == 30.0
    assert converted["images"]["head_rgb"].shape == (3, 12, 16)
    assert converted["images"]["left_wrist_rgb"].shape == (3, 10, 14)
    assert converted["images"]["right_wrist_rgb"].shape == (3, 8, 6)

    proprio = np.arange(61, dtype=np.float32)
    expected = {
        "base_qvel": proprio[0:3],
        "left_arm": proprio[3:10],
        "left_gripper": proprio[24:26],
        "right_arm": proprio[28:35],
        "right_gripper": proprio[49:51],
        "trunk_qpos": proprio[53:57],
    }
    for key, value in expected.items():
        np.testing.assert_array_equal(converted["state"][key], value)

    broken = make_observation()
    broken["robot_r1::proprio"] = np.zeros(60, dtype=np.float32)
    try:
        adapter.behavior_obs_to_g05(broken, tasks)
    except ValueError as error:
        assert "61 values" in str(error)
    else:
        raise AssertionError("60D proprio was not rejected")


def test_action_mapping() -> None:
    split_action = {
        "base_qvel": np.asarray([1, 2, 3], dtype=np.float32),
        "trunk_qpos": np.asarray([4, 5, 6, 7], dtype=np.float32),
        "left_arm": np.arange(10, 17, dtype=np.float32),
        "left_gripper": np.asarray([17], dtype=np.float32),
        "right_arm": np.arange(18, 25, dtype=np.float32),
        "right_gripper": np.asarray([25], dtype=np.float32),
    }
    vector = adapter.behavior_action_to_vector(split_action)
    expected = np.concatenate(list(split_action.values()))
    np.testing.assert_array_equal(vector, expected)

    grouped_action = dict(split_action)
    grouped_action.pop("base_qvel")
    grouped_action.pop("trunk_qpos")
    # Current processor-internal order is trunk(4) + base(3).
    grouped_action["lower_body"] = np.asarray([4, 5, 6, 7, 1, 2, 3], dtype=np.float32)
    np.testing.assert_array_equal(
        adapter.behavior_action_to_vector(grouped_action), expected
    )


def test_missing_action_holds_state() -> None:
    state = {
        "base_qvel": np.asarray([0.2, 0.1, -0.2], dtype=np.float32),
        "trunk_qpos": np.asarray([1, 2, 3, 4], dtype=np.float32),
        "left_arm": np.arange(7, dtype=np.float32),
        "left_gripper": np.asarray([0.025, 0.025], dtype=np.float32),
        "right_arm": np.arange(10, 17, dtype=np.float32),
        "right_gripper": np.asarray([0.0, 0.0], dtype=np.float32),
    }
    completed = adapter.fill_missing_behavior_action({}, state)
    np.testing.assert_array_equal(completed["base_qvel"], np.zeros(3))
    np.testing.assert_array_equal(completed["trunk_qpos"], state["trunk_qpos"])
    np.testing.assert_array_equal(completed["left_arm"], state["left_arm"])
    # Float32 summation may leave a tiny (~1e-8) rounding residue.
    np.testing.assert_allclose(completed["left_gripper"], [0.0], atol=1e-7)
    np.testing.assert_allclose(completed["right_gripper"], [-1.0])
    assert adapter.behavior_action_to_vector(completed).shape == (23,)


def test_official_codec_compatibility() -> None:
    expected = np.arange(23, dtype=np.float32)
    decoded = official_unpackb(adapter.official_packb({"action": expected}))
    assert isinstance(decoded["action"], np.ndarray)
    np.testing.assert_array_equal(decoded["action"], expected)

    # The adapter decoder must also accept arrays produced by OmniGibson.
    inbound = adapter.unpackb(official_packb({"task_id": np.asarray([3], dtype=np.int64)}))
    assert isinstance(inbound["task_id"], np.ndarray)
    np.testing.assert_array_equal(inbound["task_id"], [3])


class FakeChunkedPolicyWrapper:
    instances: list["FakeChunkedPolicyWrapper"] = []

    def __init__(
        self,
        inferencer: Any,
        processor: Any,
        *,
        action_steps: int,
        strict_memlite: bool = False,
        memlite_replan_every_chunks: int = 0,
        memlite_high_level_max_new_tokens: int = 160,
        trace_hook=None,
    ):
        self.action_steps = action_steps
        self.strict_memlite = strict_memlite
        self.memlite_replan_every_chunks = memlite_replan_every_chunks
        self.memlite_high_level_max_new_tokens = memlite_high_level_max_new_tokens
        self.trace_hook = trace_hook
        self.reset_calls = 0
        self._last_memlite_updated = False
        self._last_high_level_reason = "start"
        self.request_count = 0
        self.chunk_index = 1
        self._chunk_step = 1
        self.served_action_count = 1
        self.action_execution_start_index = 0
        self.decoded_action_horizon = 32
        self.mem_state = type(
            "MemState",
            (),
            {
                "intent_text": "synthetic-intent",
                "memory_text": "",
                "memory_initialized": False,
                "task_id": None,
            },
        )()
        self.initializations: list[tuple[int, str, str]] = []
        self.__class__.instances.append(self)

    def reset(self) -> None:
        self.reset_calls += 1
        self.mem_state.intent_text = ""
        self.mem_state.memory_text = ""
        self.mem_state.memory_initialized = False
        self.mem_state.task_id = None

    def initialize_memlite_memory(
        self,
        *,
        task_id: int,
        canonical_memory: str,
        explicit_memory: str | None = None,
    ) -> bool:
        if self.mem_state.memory_initialized:
            if self.mem_state.task_id != task_id:
                raise ValueError("synthetic task identity changed without reset")
            return False
        chosen = (explicit_memory or "").strip() or canonical_memory
        if not chosen:
            raise ValueError("synthetic initial memory missing")
        self.mem_state.memory_text = chosen
        self.mem_state.memory_initialized = True
        self.mem_state.task_id = task_id
        self.initializations.append((task_id, chosen, "explicit" if explicit_memory else "canonical"))
        return True

    async def get_action(self, raw_obs: dict[str, Any]):
        self.request_count += 1
        return {
            "base_qvel": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            "trunk_qpos": raw_obs["state"]["trunk_qpos"],
            "left_arm": raw_obs["state"]["left_arm"],
            "right_arm": raw_obs["state"]["right_arm"],
            # Grippers intentionally absent: adapter must generate hold commands.
        }, None


async def test_server_protocol() -> None:
    mem_server = importlib.import_module("scripts.serve_policy_mem")
    original_wrapper = mem_server.ChunkedPolicyWrapper
    mem_server.ChunkedPolicyWrapper = FakeChunkedPolicyWrapper
    FakeChunkedPolicyWrapper.instances.clear()
    tasks = {index: f"Synthetic task {index}" for index in range(100)}

    handler = functools.partial(
        adapter.behavior_handler,
        inferencer=object(),
        processor=object(),
        task_instructions=tasks,
        action_steps=16,
        require_memlite=False,
        memlite_high_level_max_new_tokens=768,
    )
    try:
        async with websockets.serve(
            handler,
            "127.0.0.1",
            0,
            max_size=None,
            process_request=adapter._health_check,
        ) as server:
            port = server.sockets[0].getsockname()[1]

            def check_health() -> str:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/healthz", timeout=2
                ) as response:
                    assert response.status == 200
                    return response.read().decode().strip()

            assert await asyncio.to_thread(check_health) == "OK"
            async with websockets.connect(
                f"ws://127.0.0.1:{port}", max_size=None
            ) as client:
                metadata = official_unpackb(await client.recv())
                assert metadata["action_dim"] == 23
                assert metadata["embodiment"] == "R1Pro"

                # Official reset is fire-and-forget.
                await client.send(official_packb({"reset": True}))
                await client.send(official_packb(make_observation()))
                result = official_unpackb(await asyncio.wait_for(client.recv(), timeout=5))
                assert isinstance(result["action"], np.ndarray)
                assert result["action"].shape == (23,)
                assert result["action"].dtype == np.float32

                # A different official task ID on the same websocket is an
                # episode boundary. An explicit runtime memory must win over
                # its canonical prior and cannot inherit task 0 state.
                switched = make_observation(task_id=1)
                switched["memory"] = "Task=1; Completed=explicit."
                await client.send(official_packb(switched))
                switched_result = official_unpackb(
                    await asyncio.wait_for(client.recv(), timeout=5)
                )
                assert switched_result["action"].shape == (23,)

                # An unknown task is safely rejected without an explicit
                # memory, but an explicitly supplied memory may start and
                # continue one such episode without a later hidden lookup.
                explicit_unknown = make_observation(task_id=5)
                explicit_unknown["memory"] = "Task=5; Completed=explicit."
                await client.send(official_packb(explicit_unknown))
                assert (official_unpackb(await asyncio.wait_for(client.recv(), timeout=5))["action"].shape == (23,))
                await client.send(official_packb(make_observation(task_id=5)))
                assert (official_unpackb(await asyncio.wait_for(client.recv(), timeout=5))["action"].shape == (23,))
    finally:
        mem_server.ChunkedPolicyWrapper = original_wrapper

    # One reset at connection creation, one from the official fire-and-forget
    # episode reset, and one in the connection cleanup prove state isolation.
    assert len(FakeChunkedPolicyWrapper.instances) == 1
    assert FakeChunkedPolicyWrapper.instances[0].reset_calls >= 3
    wrapper = FakeChunkedPolicyWrapper.instances[0]
    assert wrapper.initializations == [
        (0, "Task=0; Completed=none.", "canonical"),
        (1, "Task=1; Completed=explicit.", "explicit"),
        (5, "Task=5; Completed=explicit.", "explicit"),
    ]
    assert wrapper.memlite_high_level_max_new_tokens == 768


def test_task_id_initial_memory_mapping_is_explicit_and_closed() -> None:
    assert adapter.MEMLITE_INITIAL_MEMORY_BY_TASK_ID == {
        task_id: f"Task={task_id}; Completed=none." for task_id in range(5)
    }
    for task_id in range(5):
        assert adapter.memlite_initial_memory_for_task_id(task_id) == (
            f"Task={task_id}; Completed=none."
        )
    try:
        adapter.memlite_initial_memory_for_task_id(5)
    except ValueError as error:
        assert "refusing to infer task semantics" in str(error)
    else:
        raise AssertionError("unknown task ID was silently mapped")


def test_trace_json_safely_records_23d_action() -> None:
    with tempfile.TemporaryDirectory() as directory:
        recorder = adapter.MemLiteTraceRecorder(directory)
        recorder.begin_episode("Synthetic task", task_id=3)
        action = np.arange(23, dtype=np.float32)
        recorder.record(
            "official_action",
            actual_action_23d=action,
            base3=action[:3],
            trunk4=action[3:7],
            finite=bool(np.isfinite(action).all()),
            l2_norm=float(np.linalg.norm(action)),
        )
        recorder.snapshot_rgb(
            {"images": {"cam": np.zeros((3, 4, 5), dtype=np.uint8)}},
            request_count=128,
            trigger="periodic_128_steps",
            label="step_00128",
        )
        trace_path = recorder.episode_dir / "policy_trace.jsonl"
        image_paths = list((recorder.episode_dir / "images").glob("*.jpg"))
        recorder.close()
        rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        action_row = next(row for row in rows if row["event"] == "official_action")
        assert action_row["actual_action_23d"] == list(range(23))
        assert action_row["base3"] == [0.0, 1.0, 2.0]
        assert action_row["trunk4"] == [3.0, 4.0, 5.0, 6.0]
        assert action_row["finite"] is True
        assert len(image_paths) == 1


def test_trace_saves_only_the_first_official_observation_and_redacts_sensitive_payloads() -> None:
    with tempfile.TemporaryDirectory() as directory:
        recorder = adapter.MemLiteTraceRecorder(directory)
        recorder.begin_episode("Synthetic task", task_id=3)
        first = official_packb({"task_id": np.asarray([3], dtype=np.int64), "frame": "first"})
        recorder.save_first_official_observation(first, {"task_id": np.asarray([3]), "frame": "first"})
        recorder.save_first_official_observation(b"second", {"frame": "second"})
        raw_path = recorder.episode_dir / "first_official_observation.msgpack"
        assert raw_path.read_bytes() == first
        assert not (recorder.episode_dir / "first_official_observation.decoded.json").exists()
        recorder.close()

        sensitive = adapter.MemLiteTraceRecorder(directory)
        sensitive.begin_episode("Sensitive task", task_id=4)
        sensitive.save_first_official_observation(
            b"must-not-be-written", {"token": "private", "frame": "first"}
        )
        decoded_path = sensitive.episode_dir / "first_official_observation.decoded.json"
        assert decoded_path.exists()
        saved = json.loads(decoded_path.read_text(encoding="utf-8"))
        assert saved["token"] == "<redacted>"
        assert not (sensitive.episode_dir / "first_official_observation.msgpack").exists()
        sensitive.close()


def main() -> None:
    test_mapping()
    test_action_mapping()
    test_missing_action_holds_state()
    test_official_codec_compatibility()
    test_task_id_initial_memory_mapping_is_explicit_and_closed()
    asyncio.run(test_server_protocol())
    test_trace_json_safely_records_23d_action()
    test_trace_saves_only_the_first_official_observation_and_redacts_sensitive_payloads()
    print("OBSERVATION_MAPPING=PASS")
    print("ACTION_MAPPING=PASS")
    print("OFFICIAL_MSGPACK_COMPATIBILITY=PASS")
    print("HEALTH_RESET_ACTION_PROTOCOL=PASS")
    print("MEMLITE_CONNECTION_AND_EPISODE_RESET=PASS")
    print("MEMLITE_TASK_ID_INITIAL_MEMORY_AND_SWITCH_ISOLATION=PASS")
    print("MEMLITE_TRACE_JSON_23D=PASS")
    print("MEMLITE_FIRST_OBSERVATION_REPLAY_TRACE=PASS")
    print("MEMLITE_BEHAVIOR_ADAPTER_CPU_TESTS=PASS")


if __name__ == "__main__":
    main()
