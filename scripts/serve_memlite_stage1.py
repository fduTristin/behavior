"""Serve a MEM-Lite stage-1 checkpoint (high planner or low FM policy) over WebSocket.

Stage-1 runs (scripts/train_memlite_stage1.py) ship no .hydra config, so
serve_policy.py / serve_policy_mem.py cannot load them directly. Everything
needed at inference is recovered from the run's recipe.json plus the runtime
assets mirrored under one deploy ROOT:

    ROOT/
      high/                      # recipe.json, latest.json, step_*_save_0027.pt
      low/                       # recipe.json, latest.json, step_*_save_0021.pt
      manifests/memlite-stage1-v4-action-bounds/stats.json   # low 归一化
      models/memlite-b-final-20260910/B-dataset-stats.json   # high 归一化
      models/action_tokenizer.pt                             # 动作 VQ codec
      models/qwen3_5_2b_base_processor/                      # Qwen3.5 处理器

Protocol (msgpack over WebSocket) is identical to scripts/serve_policy_mem.py;
see docs/deployment/serve_policy_mem_zh.md for the client contract and
docs/deployment/serve_memlite_stage1_zh.md for the full deployment guide.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from g05.utils.training.stage1_model import make_processor, restore_model  # noqa: E402
from serve_policy_mem import handler, serve  # noqa: E402

logger = logging.getLogger("serve_memlite_stage1")

ORIGINAL_ROOT = "/data/workspace/wsy/behavior2026"


def _remap(path: str, root: Path) -> str:
    """Rewrite absolute training-side paths onto the deploy root.

    recipe.json records where each asset lived on the training cluster. On the
    deployment server the same files sit under <root>/<relative path>.
    """
    if path.startswith(ORIGINAL_ROOT + "/"):
        return str(root / path[len(ORIGINAL_ROOT) + 1:])
    return path


def load_stage1(root: Path, branch: str, ckpt: Path | None, device: str):
    run = root / branch
    recipe = json.loads((run / "recipe.json").read_text())
    if recipe["config"]["component"] != branch:
        raise ValueError(f"recipe component mismatch: {recipe['config']['component']} != {branch}")
    model_config = recipe["model"]
    model_config["stats_path"] = _remap(model_config["stats_path"], root)
    if Path(model_config["stats_path"]).name != "stats.json" and branch == "low":
        raise ValueError("Low stats must be the stage1-v4-action-bounds sidecar")
    if ckpt is None:
        ckpt = run / json.loads((run / "latest.json").read_text())["path"]
    logger.info("Loading %s checkpoint %s", branch, ckpt)
    saved = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
    processor = make_processor(model_config, training=False)
    model, receipt = restore_model(model_config, branch, state=saved["model_state_dict"])
    logger.info("Exact restore verified: %s", receipt)
    model = model.to(device).to(torch.bfloat16)
    model.apply_fp32_params()
    model.eval()
    if hasattr(model, "action_tokenizer"):
        model.action_tokenizer.to(device)
    model.requires_grad_(False)
    return model, processor


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch", choices=("high", "low"), required=True)
    ap.add_argument("--root", type=Path, default=Path(ORIGINAL_ROOT),
                    help="Deploy root holding high/ low/ manifests/ models/")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="Checkpoint .pt; defaults to <root>/<branch>/latest.json")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--action_steps", type=int, default=16,
                    help="Steps served per inference; 1 = real-time-chunking mode")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    model, processor = load_stage1(args.root, args.branch, args.ckpt, args.device)
    logger.info("%s policy ready on %s, serving ws://%s:%d",
                args.branch, args.device, args.host, args.port)
    asyncio.run(
        serve(
            handler,
            model,
            processor,
            args.host,
            args.port,
            device=args.device,
            action_steps=args.action_steps,
        )
    )


if __name__ == "__main__":
    main()
