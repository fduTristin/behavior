#!/usr/bin/env python3
"""Send synthetic official-protocol observations to the stage-1 server.

This exercises the real websocket handshake, reset semantics, planner AR,
low-level FM chunk, and 23-dimensional action response without starting
OmniGibson.  It is a connectivity/runtime smoke test, not a task-success test.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

import numpy as np
import websockets

BRIDGE_DIR = Path(__file__).resolve().parent / "behavior_bridge"
if str(BRIDGE_DIR) not in sys.path:
    sys.path.insert(0, str(BRIDGE_DIR))

import serve_behavior_policy_mem as bridge  # noqa: E402


def synthetic_observation(task_id: int) -> dict:
    """Build one zero-valued observation in the official flattened layout."""
    return {
        "robot_r1::proprio": np.zeros(61, dtype=np.float32),
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.zeros(
            (720, 720, 4), dtype=np.uint8
        ),
        "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": np.zeros(
            (480, 480, 4), dtype=np.uint8
        ),
        "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": np.zeros(
            (480, 480, 4), dtype=np.uint8
        ),
        "task_id": np.asarray([task_id], dtype=np.int64),
    }


async def run(uri: str, task_id: int, requests: int, timeout: float) -> dict:
    async with websockets.connect(uri, max_size=None, ping_interval=None) as client:
        handshake = bridge.unpackb(
            await asyncio.wait_for(client.recv(), timeout=min(timeout, 30.0))
        )
        expected = {
            "policy": "G0.5",
            "embodiment": "R1Pro",
            "action_dim": 23,
            "action_steps": 16,
        }
        for key, value in expected.items():
            if handshake.get(key) != value:
                raise RuntimeError(
                    f"handshake {key!r} mismatch: {handshake.get(key)!r} != {value!r}"
                )

        # Official reset is fire-and-forget; the next frame receives the first
        # response after planner generation and low-level chunk construction.
        await client.send(bridge.official_packb({"reset": True}))
        norms = []
        for _ in range(requests):
            await client.send(bridge.official_packb(synthetic_observation(task_id)))
            response = bridge.unpackb(
                await asyncio.wait_for(client.recv(), timeout=timeout)
            )
            action = np.asarray(response.get("action"), dtype=np.float32)
            if action.shape != (23,) or not np.isfinite(action).all():
                raise RuntimeError(
                    f"server returned invalid official action: shape={action.shape}"
                )
            norms.append(float(np.linalg.norm(action)))
        return {"handshake": handshake, "requests": requests, "action_l2": norms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10110)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if not 0 <= args.task_id < 100:
        parser.error("--task-id must be in [0, 99]")
    if args.requests < 1:
        parser.error("--requests must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    result = asyncio.run(
        run(
            f"ws://{args.host}:{args.port}",
            task_id=args.task_id,
            requests=args.requests,
            timeout=args.timeout,
        )
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
