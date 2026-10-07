"""Serve MEM-Lite G0.5 through the official BEHAVIOR 2026 websocket protocol.

The MEM server and the official evaluator both use websocket + msgpack, but
their observation, reset, action, and ndarray encodings are different. This
adapter keeps the model's high-/low-level inference unchanged and translates
only the wire protocol at the server boundary. Each WebSocket connection owns
one MEM-Lite episode state and explicit reset messages clear it.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import websockets

from g05.utils.websocket import unpackb


LOGGER = logging.getLogger(__name__)


# Official RGBDFullResWrapper flattened observation suffixes. Depth is present
# in BEHAVIOR but G0.5 was trained with these three RGB streams only.
CAMERA_SUFFIXES = {
    "head_rgb": "robot_r1:zed_link:Camera:0::rgb",
    "left_wrist_rgb": "robot_r1:left_realsense_link:Camera:0::rgb",
    "right_wrist_rgb": "robot_r1:right_realsense_link:Camera:0::rgb",
}

# BEHAVIOR v3.9.x R1Pro proprioception layout (61 values). Only fields used by
# the G0.5 training processor are selected.
STATE_SLICES = {
    "base_qvel": slice(0, 3),
    "left_arm": slice(3, 10),
    "left_gripper": slice(24, 26),
    "right_arm": slice(28, 35),
    "right_gripper": slice(49, 51),
    "trunk_qpos": slice(53, 57),
}

# These are the exact canonical first-memory strings used by the trained
# high-level annotations for the five-task subset.  They are keyed solely by
# the official observation's task_id; no goal predicate, target state, or
# future trajectory enters the policy input.
MEMLITE_INITIAL_MEMORY_BY_TASK_ID = {
    task_id: f"Task={task_id}; Completed=none." for task_id in range(5)
}


def _official_pack_data(value: Any) -> Any:
    """Encode arrays exactly as the official OmniGibson client expects.

    OmniGibson's decoder checks *bytes* marker keys. G0.5's general-purpose
    websocket encoder uses string marker keys, which look similar but are not
    decoded into arrays by the official client.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {value.dtype}")
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


official_packb = functools.partial(msgpack.packb, default=_official_pack_data)


def load_task_instructions(tasks_path: str | Path) -> dict[int, str]:
    """Load the 100 released task instructions by official zero-based id."""
    instructions: dict[int, str] = {}
    with Path(tasks_path).open("r", encoding="utf-8") as task_file:
        for line in task_file:
            if not line.strip():
                continue
            task = json.loads(line)
            instructions[int(task["task_index"])] = str(task["task"])
    if len(instructions) != 100:
        raise ValueError(
            f"Expected 100 BEHAVIOR task instructions, found {len(instructions)}"
        )
    return instructions


def _find_value_by_suffix(payload: dict[str, Any], suffix: str) -> Any:
    matches = [value for key, value in payload.items() if key.endswith(suffix)]
    if len(matches) != 1:
        matching_keys = [key for key in payload if key.endswith(suffix)]
        raise KeyError(
            f"Expected exactly one observation key ending with {suffix!r}; "
            f"found {matching_keys}"
        )
    return matches[0]


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def memlite_initial_memory_for_task_id(task_id: int) -> str:
    """Return the trained five-task first-memory prefix for an official ID."""
    normalized_task_id = int(task_id)
    try:
        return MEMLITE_INITIAL_MEMORY_BY_TASK_ID[normalized_task_id]
    except KeyError as error:
        raise ValueError(
            "No canonical MEM-Lite initial memory is configured for official "
            f"task_id={normalized_task_id}; refusing to infer task semantics from text"
        ) from error


def _explicit_memory_from_payload(payload: dict[str, Any]) -> str:
    """Read only an explicitly supplied memory field, never task GT metadata."""
    value = payload.get("memory")
    return value.strip() if isinstance(value, str) else ""


def _to_chw_uint8(image: Any, camera_name: str) -> np.ndarray:
    """Convert OmniGibson HWC RGB/RGBA (or CHW) to G0.5 CHW uint8."""
    array = _as_numpy(image)
    if array.ndim != 3:
        raise ValueError(f"{camera_name} must be 3D, got shape {array.shape}")

    if array.shape[-1] in (3, 4):
        array = array[..., :3].transpose(2, 0, 1)
    elif array.shape[0] in (3, 4):
        array = array[:3]
    else:
        raise ValueError(
            f"{camera_name} must be HWC/CHW RGB(A), got shape {array.shape}"
        )

    if np.issubdtype(array.dtype, np.floating):
        finite_max = float(np.nanmax(array)) if array.size else 0.0
        if finite_max <= 1.0:
            array = array * 255.0
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def behavior_obs_to_g05(
    evaluator_obs: dict[str, Any], task_instructions: dict[int, str]
) -> dict[str, Any]:
    """Translate flattened official evaluator observations to G0.5 raw_obs."""
    if not isinstance(evaluator_obs, dict):
        raise TypeError(f"Evaluator observation must be a dict, got {type(evaluator_obs)}")

    proprio = _as_numpy(_find_value_by_suffix(evaluator_obs, "::proprio"))
    proprio = np.asarray(proprio, dtype=np.float32).reshape(-1)
    if proprio.shape != (61,):
        raise ValueError(f"R1Pro proprio must have 61 values, got {proprio.shape}")

    if "task_id" not in evaluator_obs:
        raise KeyError("Evaluator observation is missing task_id")
    task_values = _as_numpy(evaluator_obs["task_id"]).reshape(-1)
    if task_values.size != 1:
        raise ValueError(f"task_id must contain one value, got shape {task_values.shape}")
    task_id = int(task_values[0])
    if task_id not in task_instructions:
        raise KeyError(f"Unknown BEHAVIOR task_id: {task_id}")

    images = {
        output_name: _to_chw_uint8(
            _find_value_by_suffix(evaluator_obs, input_suffix), output_name
        )
        for output_name, input_suffix in CAMERA_SUFFIXES.items()
    }
    state = {
        output_name: np.ascontiguousarray(proprio[index_slice], dtype=np.float32)
        for output_name, index_slice in STATE_SLICES.items()
    }
    return {
        "images": images,
        "state": state,
        "task": task_instructions[task_id],
        "embodiment_type": "galaxea_r1pro",
        "frequency": 30.0,
    }


def _action_part(action: dict[str, Any], key: str, expected_dim: int) -> np.ndarray:
    if key not in action:
        raise KeyError(f"G0.5 action is missing {key!r}; available keys: {sorted(action)}")
    value = np.asarray(action[key], dtype=np.float32).reshape(-1)
    if value.shape != (expected_dim,):
        raise ValueError(
            f"Action {key!r} must have {expected_dim} values, got {value.shape}"
        )
    return value


def behavior_action_to_vector(action: dict[str, Any]) -> np.ndarray:
    """Translate G0.5 actions to the official R1Pro 23D controller order.

    Official order is base(3), trunk(4), left_arm(7), left_gripper(1),
    right_arm(7), right_gripper(1).

    Normal inference returns split ``base_qvel`` and ``trunk_qpos`` keys after
    postprocessing. The grouped fallback follows the current training layout:
    ``lower_body = trunk_qpos(4) + base_qvel(3)``.
    """
    if "base_qvel" in action and "trunk_qpos" in action:
        base = _action_part(action, "base_qvel", 3)
        trunk = _action_part(action, "trunk_qpos", 4)
    elif "lower_body" in action:
        lower_body = _action_part(action, "lower_body", 7)
        trunk = lower_body[:4]
        base = lower_body[4:]
    else:
        raise KeyError(
            "G0.5 action must contain base_qvel + trunk_qpos or lower_body"
        )

    vector = np.concatenate(
        [
            base,
            trunk,
            _action_part(action, "left_arm", 7),
            _action_part(action, "left_gripper", 1),
            _action_part(action, "right_arm", 7),
            _action_part(action, "right_gripper", 1),
        ]
    ).astype(np.float32, copy=False)
    if vector.shape != (23,):
        raise AssertionError(f"Internal error: expected a 23D action, got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError("G0.5 produced NaN or infinite action values")
    return vector


def _hold_gripper_action(state_value: Any) -> np.ndarray:
    """Convert two-finger state width to the model's 1D gripper command."""
    state = np.asarray(state_value, dtype=np.float32).reshape(-1)
    if state.shape != (2,):
        raise ValueError(f"Gripper state must have 2 finger values, got {state.shape}")
    return np.asarray([2.0 * (float(state.sum()) / 0.1) - 1.0], dtype=np.float32)


def fill_missing_behavior_action(
    action: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    """Fill action groups omitted by ActionCodecV2 with safe hold commands."""
    completed = dict(action)
    defaults = {
        "base_qvel": np.zeros(3, dtype=np.float32),
        "trunk_qpos": state["trunk_qpos"],
        "left_arm": state["left_arm"],
        "left_gripper": _hold_gripper_action(state["left_gripper"]),
        "right_arm": state["right_arm"],
        "right_gripper": _hold_gripper_action(state["right_gripper"]),
    }
    if "lower_body" not in completed:
        for key in ("base_qvel", "trunk_qpos"):
            completed.setdefault(key, defaults[key])
    for key in ("left_arm", "left_gripper", "right_arm", "right_gripper"):
        completed.setdefault(key, defaults[key])
    return completed


def _json_safe(value: Any) -> Any:
    """Convert diagnostic values to JSON without touching policy inputs."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


_TRACE_SENSITIVE_KEY_PARTS = ("authorization", "credential", "password", "secret", "token", "cookie")


def _trace_key_is_sensitive(key: Any) -> bool:
    if isinstance(key, bytes):
        key = key.decode("utf-8", errors="replace")
    normalized = str(key).lower()
    return any(part in normalized for part in _TRACE_SENSITIVE_KEY_PARTS)


def _trace_payload_has_sensitive_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            _trace_key_is_sensitive(key) or _trace_payload_has_sensitive_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_trace_payload_has_sensitive_key(item) for item in value)
    return False


def _trace_redact_sensitive(value: Any) -> Any:
    """JSON-safe payload copy that does not expose authentication fields."""
    if isinstance(value, dict):
        return {
            str(key): "<redacted>" if _trace_key_is_sensitive(key) else _trace_redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_trace_redact_sensitive(item) for item in value]
    return _json_safe(value)


def _safe_path_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("._") or "unknown"


class MemLiteTraceRecorder:
    """Write per-task / per-episode diagnostics without affecting inference."""

    def __init__(self, root: str | Path | None):
        self.root = Path(root) if root else None
        self._handle = None
        self._current_task: str | None = None
        self._episode_counts: dict[str, int] = {}
        self._episode_dir: Path | None = None
        self._pending: list[dict[str, Any]] = []
        self._first_official_observation_saved = False

    @property
    def episode_dir(self) -> Path | None:
        return self._episode_dir

    def _write(self, payload: dict[str, Any]) -> None:
        if self._handle is None:
            self._pending.append(payload)
            return
        try:
            enriched = {"wall_time_unix": time.time(), **_json_safe(payload)}
            self._handle.write(json.dumps(enriched, ensure_ascii=False, allow_nan=False) + "\n")
            self._handle.flush()
        except Exception:
            LOGGER.exception("Failed to write MEM-Lite diagnostic trace")

    def begin_episode(self, task: str, task_id: int | None = None) -> None:
        if self.root is None:
            return
        if self._handle is not None and self._current_task == task:
            return
        try:
            self.close()
            task_key = _safe_path_component(task)
            episode_number = self._episode_counts.get(task_key, 0) + 1
            self._episode_counts[task_key] = episode_number
            self._episode_dir = self.root / task_key / f"episode_{episode_number:04d}"
            self._episode_dir.mkdir(parents=True, exist_ok=False)
            self._handle = (self._episode_dir / "policy_trace.jsonl").open("a", encoding="utf-8")
            self._current_task = task
            self._write(
                {
                    "source": "bridge",
                    "event": "episode_start",
                    "task": task,
                    "task_id": task_id,
                }
            )
            pending, self._pending = self._pending, []
            for payload in pending:
                self._write(payload)
        except Exception:
            LOGGER.exception("Failed to initialize MEM-Lite diagnostic trace")
            self.close()
            self.root = None

    def policy_event(self, payload: dict[str, Any]) -> None:
        if self.root is None:
            return
        self._write(payload)
        if payload.get("event") == "reset" and self._handle is not None:
            self.close()

    def record(self, event: str, **fields: Any) -> None:
        if self.root is not None:
            self._write({"source": "bridge", "event": event, **fields})

    def save_first_official_observation(self, raw_message: Any, decoded: dict[str, Any]) -> None:
        """Persist one inbound official observation for offline failure replay.

        This captures an already-received websocket message only when tracing
        is enabled; it never changes the policy input or response.  A message
        carrying authentication-like keys is never written verbatim: a
        redacted decoded JSON fallback is used instead.
        """
        if (
            self.root is None
            or self._episode_dir is None
            or self._first_official_observation_saved
        ):
            return
        self._first_official_observation_saved = True
        try:
            contains_sensitive_key = _trace_payload_has_sensitive_key(decoded)
            if isinstance(raw_message, (bytes, bytearray)) and not contains_sensitive_key:
                path = self._episode_dir / "first_official_observation.msgpack"
                path.write_bytes(bytes(raw_message))
                storage = "raw_msgpack"
            else:
                path = self._episode_dir / "first_official_observation.decoded.json"
                path.write_text(
                    json.dumps(_trace_redact_sensitive(decoded), ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                storage = "redacted_decoded_json"
            self.record(
                "first_official_observation_saved",
                path=path.name,
                storage=storage,
                offline_debug_only=True,
            )
        except Exception:
            # Tracing cannot invalidate an official request.
            LOGGER.exception("Failed to save first official observation diagnostic")

    def snapshot_rgb(
        self,
        raw_obs: dict[str, Any],
        *,
        request_count: int,
        trigger: str,
        label: str,
    ) -> None:
        """Save an already-received official RGB observation as a JPEG diagnostic."""
        if self._episode_dir is None:
            return
        try:
            from PIL import Image

            image_dir = self._episode_dir / "images"
            image_dir.mkdir(exist_ok=True)
            saved = []
            for name, image in raw_obs.get("images", {}).items():
                array = np.asarray(image)
                if array.ndim != 3:
                    continue
                if array.shape[0] in (3, 4):
                    array = array[:3].transpose(1, 2, 0)
                elif array.shape[-1] in (3, 4):
                    array = array[..., :3]
                else:
                    continue
                path = image_dir / f"{_safe_path_component(label)}_r{request_count:05d}_{_safe_path_component(name)}.jpg"
                Image.fromarray(np.ascontiguousarray(array, dtype=np.uint8)).save(path, format="JPEG", quality=85)
                saved.append(str(path.name))
            self.record(
                "rgb_snapshot",
                request_count=request_count,
                trigger=trigger,
                label=label,
                images=saved,
            )
        except Exception as exc:
            self.record(
                "trace_exception",
                stage="jpeg_snapshot",
                exception_type=type(exc).__name__,
                message=str(exc),
            )

    def snapshot_exception(self, raw_obs: dict[str, Any] | None) -> None:
        """Capture the last official RGB observation when an inference call fails."""
        if self._episode_dir is None or not raw_obs:
            return
        try:
            from PIL import Image

            image_dir = self._episode_dir / "images"
            image_dir.mkdir(exist_ok=True)
            saved = []
            for name, image in raw_obs.get("images", {}).items():
                array = np.asarray(image)
                if array.ndim != 3:
                    continue
                if array.shape[0] in (3, 4):
                    array = array[:3].transpose(1, 2, 0)
                elif array.shape[-1] in (3, 4):
                    array = array[..., :3]
                else:
                    continue
                path = image_dir / f"exception_{_safe_path_component(name)}.jpg"
                Image.fromarray(np.ascontiguousarray(array, dtype=np.uint8)).save(path, format="JPEG", quality=85)
                saved.append(str(path.name))
            self.record("exception_rgb_snapshot", images=saved)
        except Exception as exc:
            self.record(
                "trace_exception",
                stage="exception_jpeg_snapshot",
                exception_type=type(exc).__name__,
                message=str(exc),
            )

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._current_task = None
        self._episode_dir = None


def _health_check(connection: Any, request: Any) -> Any:
    if getattr(request, "path", None) == "/healthz":
        if hasattr(connection, "respond"):
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return http.HTTPStatus.OK, {"Content-Type": "text/plain"}, b"OK\n"
    return None


async def behavior_handler(
    websocket: Any,
    *,
    inferencer: Any,
    processor: Any,
    task_instructions: dict[int, str],
    action_steps: int,
    require_memlite: bool,
    memlite_replan_every_chunks: int = 0,
    memlite_high_level_max_new_tokens: int = 160,
    trace_root: str | Path | None = None,
) -> None:
    """Handle one official BEHAVIOR evaluator connection with fresh MEM state."""
    from scripts.serve_policy_mem import ChunkedPolicyWrapper

    client = websocket.remote_address
    LOGGER.info("BEHAVIOR evaluator connected: %s", client)
    await websocket.send(
        official_packb(
            {
                "policy": "G0.5",
                "embodiment": "R1Pro",
                "action_dim": 23,
                "action_steps": action_steps,
            }
        )
    )
    trace = MemLiteTraceRecorder(trace_root)
    policy = ChunkedPolicyWrapper(
        inferencer,
        processor,
        action_steps=action_steps,
        strict_memlite=require_memlite,
        memlite_replan_every_chunks=memlite_replan_every_chunks,
        memlite_high_level_max_new_tokens=memlite_high_level_max_new_tokens,
        trace_hook=trace.policy_event,
    )
    # Be explicit even though a new wrapper is initialized empty: every new
    # official evaluator connection is a new episode and must never inherit a
    # semantic memory, intent, cached action, or temporal frame buffer.
    policy.reset()
    LOGGER.info("MEMLITE_STATE_RESET reason=connection_open client=%s", client)

    try:
        async for message in websocket:
            g05_obs: dict[str, Any] | None = None
            try:
                payload = unpackb(message)
                # The official client sends reset without waiting for a reply.
                if isinstance(payload, dict) and (
                    payload.get("reset") or payload.get("__reset__")
                ):
                    policy.reset()
                    trace.record("bridge_reset", reason="official_reset")
                    LOGGER.info("MEMLITE_STATE_RESET reason=episode_reset client=%s", client)
                    continue

                task_values = _as_numpy(payload["task_id"]).reshape(-1)
                if task_values.size != 1:
                    raise ValueError(
                        "Official MEM-Lite evaluation requires exactly one task_id per observation"
                    )
                task_id = int(task_values[0])
                explicit_memory = _explicit_memory_from_payload(payload)
                # Task changes are an episode boundary even if an evaluator
                # reuses the websocket. Reset before opening the next trace so
                # old memory/intent/chunk buffers cannot cross radio -> trash.
                if (
                    policy.mem_state.memory_initialized
                    and policy.mem_state.task_id != task_id
                ):
                    previous_task_id = policy.mem_state.task_id
                    policy.reset()
                    trace.record(
                        "bridge_reset",
                        reason="official_task_switch",
                        previous_task_id=previous_task_id,
                        task_id=task_id,
                    )
                canonical_memory = ""
                # Once initialized, preserve state (including a model-produced
                # empty value) instead of looking up/replacing a prior again.
                # This also permits a caller that supplied explicit memory for
                # an otherwise unsupported task ID to continue that episode.
                if not explicit_memory and not policy.mem_state.memory_initialized:
                    canonical_memory = memlite_initial_memory_for_task_id(task_id)
                g05_obs = behavior_obs_to_g05(payload, task_instructions)
                trace.begin_episode(g05_obs["task"], task_id=task_id)
                policy.initialize_memlite_memory(
                    task_id=task_id,
                    canonical_memory=canonical_memory,
                    explicit_memory=explicit_memory,
                )
                trace.save_first_official_observation(message, payload)
                action_dict, cot_text = await policy.get_action(g05_obs)
                if policy._last_memlite_updated:
                    trace.snapshot_rgb(
                        g05_obs,
                        request_count=policy.request_count,
                        trigger=policy._last_high_level_reason,
                        label="high_level",
                    )
                if policy.served_action_count % 128 == 0:
                    trace.snapshot_rgb(
                        g05_obs,
                        request_count=policy.request_count,
                        trigger="periodic_128_steps",
                        label=f"step_{policy.served_action_count:05d}",
                    )
                model_action_keys = sorted(action_dict)
                action_dict = fill_missing_behavior_action(
                    action_dict, g05_obs["state"]
                )
                vector = behavior_action_to_vector(action_dict)
                filled_action_keys = sorted(set(action_dict) - set(model_action_keys))
                trace.record(
                    "official_action",
                    request_count=policy.request_count,
                    chunk_index=policy.chunk_index,
                    chunk_position=policy._chunk_step,
                    served_action_count=policy.served_action_count,
                    low_level_intent=policy.mem_state.intent_text,
                    model_action_keys=model_action_keys,
                    hold_or_zero_fallback_keys=filled_action_keys,
                    actual_action_23d=vector,
                    base3=vector[:3],
                    trunk4=vector[3:7],
                    action_execution_start_index=policy.action_execution_start_index,
                    decoded_action_horizon=policy.decoded_action_horizon,
                    postprocess_action_horizon=policy.decoded_action_horizon,
                    executed_action_steps=policy.action_steps,
                    finite=bool(np.isfinite(vector).all()),
                    l2_norm=float(np.linalg.norm(vector)),
                )
                response = {"action": vector}
                if cot_text is not None:
                    response["cot_text"] = cot_text
                await websocket.send(official_packb(response))
            except Exception as exc:
                error_message = "G0.5 BEHAVIOR adapter error:\n" + traceback.format_exc()
                trace.record(
                    "bridge_exception",
                    exception_type=type(exc).__name__,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                    strict_memlite=require_memlite,
                )
                trace.snapshot_exception(g05_obs)
                LOGGER.exception("BEHAVIOR inference request failed")
                if require_memlite:
                    # Let the official evaluator's websocket fail rather than
                    # accepting a text-frame error as a valid policy action.
                    raise
                # Official client treats text frames as inference failures.
                await websocket.send(error_message)
    except websockets.exceptions.ConnectionClosed:
        LOGGER.info("BEHAVIOR evaluator disconnected: %s", client)
    finally:
        policy.reset()
        trace.close()
        LOGGER.info("MEMLITE_STATE_RESET reason=connection_close client=%s", client)


async def serve_behavior(
    policy: Any,
    processor: Any,
    *,
    host: str,
    port: int,
    device: str,
    action_steps: int,
    task_instructions: dict[int, str],
    require_memlite: bool,
    memlite_replan_every_chunks: int = 0,
    memlite_high_level_max_new_tokens: int = 160,
    trace_root: str | Path | None = None,
) -> None:
    from g05.models.g05.inferencer import PolicyInferencer
    from scripts.serve_policy_mem import ChunkedPolicyWrapper

    inferencer = PolicyInferencer(policy, processor, device=device)
    if require_memlite:
        capability_probe = ChunkedPolicyWrapper(
        inferencer, processor, action_steps=action_steps, strict_memlite=True
        )
        if not capability_probe._memlite_enabled:
            raise RuntimeError(
                "MEM-Lite evaluation requires infer_high_level + "
                "infer_low_level_action with predict_cot enabled"
            )
    handler = functools.partial(
        behavior_handler,
        inferencer=inferencer,
        processor=processor,
        task_instructions=task_instructions,
        action_steps=action_steps,
        require_memlite=require_memlite,
        memlite_replan_every_chunks=memlite_replan_every_chunks,
        memlite_high_level_max_new_tokens=memlite_high_level_max_new_tokens,
        trace_root=trace_root,
    )
    async with websockets.serve(
        handler,
        host,
        port,
        max_size=None,
        # OmniGibson can spend more than 20 seconds inside a physics/render
        # step without servicing a server-initiated ping.  Disable the
        # server's aggressive default keepalive; the official client keeps
        # its own 60 s ping interval and 300 s timeout.
        ping_interval=None,
        process_request=_health_check,
    ):
        LOGGER.info(
            "MEM-Lite BEHAVIOR policy server listening on ws://%s:%d "
            "(action_steps=%d, device=%s, require_memlite=%s)",
            host,
            port,
            action_steps,
            device,
            require_memlite,
        )
        await asyncio.Future()


def _safe_apply_action_tokenizer_sidecar(config: Any, run_dir: Path) -> bool:
    """Patch the tokenizer sidecar without breaking OmegaConf aliases.

    In this training config ``model.tokenizer`` aliases the root ``tokenizer``
    node. Updating a child through the alias replaces the alias with a partial
    dictionary and drops ``_target_``. Patch the real root node once instead;
    the model aliases then resolve to the same updated value.
    """
    from omegaconf import OmegaConf

    local_tokenizer = Path(run_dir) / "action_tokenizer.pt"
    if not local_tokenizer.exists():
        return False

    if OmegaConf.select(config, "tokenizer.vq_config.ckpt_dir") is not None:
        OmegaConf.update(
            config,
            "tokenizer.vq_config.ckpt_dir",
            str(local_tokenizer),
            merge=False,
        )
        return True

    # Legacy exported configs may contain concrete copies instead of a root
    # tokenizer node. These paths are safe to update only in that case.
    patched = False
    for key in (
        "model.tokenizer.vq_config.ckpt_dir",
        "model.model_arch.AT_CONFIG.ckpt_dir",
    ):
        if OmegaConf.select(config, key) is not None:
            OmegaConf.update(config, key, str(local_tokenizer), merge=False)
            patched = True
    return patched


def load_runtime_config(ckpt_path: str, overrides: list[str]) -> Any:
    """Load a saved training config without loading checkpoint tensors."""
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.utils.checkpoint import ckpt_utils

    # Saved configs contain custom expressions such as ``obs_image_steps`` and
    # ``oc.load``. They must be registered before OmegaConf resolves the file.
    register_default_resolvers()
    run_dir = ckpt_utils.find_run_dir(ckpt_path)
    original_sidecar_patch = ckpt_utils._apply_action_tokenizer_sidecar
    ckpt_utils._apply_action_tokenizer_sidecar = _safe_apply_action_tokenizer_sidecar
    try:
        return ckpt_utils.load_config_from_run_dir(run_dir, ckpt_path, overrides)
    finally:
        ckpt_utils._apply_action_tokenizer_sidecar = original_sidecar_patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--tasks_path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action_steps", type=int, default=16)
    parser.add_argument(
        "--pure-ar",
        action="store_true",
        help="Require discrete autoregressive action-token inference.",
    )
    parser.add_argument(
        "--predict-cot",
        action="store_true",
        help="Require the MEM-Lite high-/low-level CoT route.",
    )
    parser.add_argument(
        "--require-memlite",
        action="store_true",
        help="Fail closed if MEM high-/low-level inference is unavailable or errors.",
    )
    parser.add_argument(
        "--memlite-replan-every-chunks",
        type=int,
        default=0,
        help="Recompute the high-level branch every N low-level action chunks; 0 disables periodic replanning.",
    )
    parser.add_argument(
        "--memlite-high-level-max-new-tokens",
        type=int,
        default=160,
        help="Maximum AR tokens for one MEM-Lite high-level generation; default 160 preserves prior serving.",
    )
    parser.add_argument(
        "--trace-root",
        default=None,
        help="Optional root for per-task / per-episode MEM-Lite JSONL and sparse RGB diagnostics.",
    )
    args, remaining = parser.parse_known_args()
    if args.memlite_replan_every_chunks < 0:
        parser.error("--memlite-replan-every-chunks must be >= 0")
    if args.memlite_high_level_max_new_tokens < 1:
        parser.error("--memlite-high-level-max-new-tokens must be >= 1")
    overrides = [item for item in remaining if "=" in item]

    from g05.utils.eval.eval_utils import filter_embodiment
    from g05.utils.logging.banner import print_banner
    from g05.utils.logging.logging_config import setup_logging
    from scripts.serve_policy_mem import setup

    config = load_runtime_config(args.ckpt_path, overrides)
    if args.pure_ar:
        config.model.model_arch.discrete_action = True
        config.model.model_arch.continuous_action = False
        config.model.model_arch.return_continuous_action = False
        config.model.processor.discrete_action = True
    if args.predict_cot:
        config.model.model_arch.predict_cot = True
        config.model.model_arch.input_preprocessor.pred_eov = True
    action_horizon = int(config.data.action_size)
    if args.action_steps < 1 or args.action_steps > action_horizon:
        parser.error(
            f"--action_steps must be in 1-{action_horizon} for this checkpoint, "
            f"got {args.action_steps}"
        )

    eval_embodiment = config.get("eval_embodiment", None)
    if eval_embodiment and "embodiment_datasets" in config.data:
        filter_embodiment(config, eval_embodiment)

    setup_logging(log_level=logging.INFO, is_main_process=True)
    print_banner(subtitle="BEHAVIOR Policy Server")
    task_instructions = load_task_instructions(args.tasks_path)
    policy, processor = setup(config, device=args.device)
    LOGGER.info("Loaded MEM-Lite G0.5 checkpoint: %s", args.ckpt_path)

    asyncio.run(
        serve_behavior(
            policy,
            processor,
            host=args.host,
            port=args.port,
            device=args.device,
            action_steps=args.action_steps,
            task_instructions=task_instructions,
            require_memlite=args.require_memlite,
            memlite_replan_every_chunks=args.memlite_replan_every_chunks,
            memlite_high_level_max_new_tokens=args.memlite_high_level_max_new_tokens,
            trace_root=args.trace_root,
        )
    )


if __name__ == "__main__":
    main()
