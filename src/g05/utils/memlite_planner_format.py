# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
"""Planner-only AR generation format constants and receipt.

The stage-1 planner (``G05PolicyMEMLitePlannerOutcome.generate_high_level``)
wraps its ``generate_text`` call in :func:`planner_only_format_constants` so the
serving-side decoding format (the fixed ``<HL_END>`` delimitation, batching
shape, configured token budget) is checked once per generation instead of being
implicit.  The manager changes nothing about decoding; it validates the
constants and yields a receipt that the generation result carries back to the
caller for audit.
"""
from __future__ import annotations

import contextlib
from typing import Any

from g05.utils.memlite_skill_protocol import MEMLITE_SKILL_SCHEMA_VERSION

HL_END_TEXT = "<HL_END>"


@contextlib.contextmanager
def planner_only_format_constants(
    ar_helper: Any,
    processor: Any,
    *,
    batch_size: int,
    hl_end_id: Any,
    enabled: bool,
):
    """Validate planner-only generation constants; yield an audit receipt.

    Args:
        ar_helper: model AR helper (must expose a ``<HL_END>``-consistent vocab).
        processor: model-side input processor holding the tokenizer.
        batch_size: number of target-free prefixes in this AR call.
        hl_end_id: configured ``<HL_END>`` token id used as the stop condition.
        enabled: planner-only profile flag; the target-free stage-1 planner
            requires it.  ``False`` would mean the legacy mixed MEM-Lite route,
            which this context manager deliberately does not service.
    """
    if not enabled:
        raise ValueError(
            "planner_only_format_constants services the target-free planner-only route; "
            "a non-planner-only stage1 planner profile is a config error"
        )
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError(f"planner batch_size must be a positive integer, got {batch_size!r}")
    if not isinstance(hl_end_id, int) or isinstance(hl_end_id, bool) or hl_end_id < 0:
        raise ValueError(f"planner generation needs a concrete <HL_END> token id, got {hl_end_id!r}")

    configured = getattr(processor, "hl_end_token_id", None)
    if configured is not None and int(configured) != int(hl_end_id):
        raise ValueError(
            f"<HL_END> mismatch: processor configured {configured} but generation stops at {hl_end_id}"
        )
    tokenizer = getattr(processor, "tokenizer", None)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        from_tokenizer = convert(HL_END_TEXT)
        if isinstance(from_tokenizer, int) and from_tokenizer >= 0 and int(from_tokenizer) != int(hl_end_id):
            raise ValueError(
                f"<HL_END> mismatch: tokenizer maps it to {from_tokenizer} but generation stops at {hl_end_id}"
            )

    receipt = {
        "planner_only": True,
        "schema_version": MEMLITE_SKILL_SCHEMA_VERSION,
        "batch_size": int(batch_size),
        "hl_end_id": int(hl_end_id),
        "hl_end_text": HL_END_TEXT,
    }
    yield receipt
