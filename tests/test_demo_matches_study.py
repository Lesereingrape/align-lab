"""The live demo must be one seed of the published experiment, not a smaller stand-in.

``alignlab demo`` used to draw 1200/400 examples, run 400 steps at a hard-coded
learning rate and evaluate on a cadence the study never used — so its four printed
lines described an unpublished experiment while sitting directly above a table of
the published one.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from alignlab import cli
from alignlab.study import (
    BASE_STEPS,
    EVAL_EVERY,
    HI,
    LO,
    LR_ALIGN,
    LR_BASE,
    METHODS,
    N_EVAL,
    N_PAIRS,
    SEEDS,
    STEPS,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    data = json.loads((ROOT / "results" / "alignment.json").read_text(encoding="utf-8"))
    return data["config"]


def test_study_constants_are_the_committed_settings():
    config = _config()
    assert config["n_pairs"] == N_PAIRS
    assert config["n_eval"] == N_EVAL
    assert config["operand_lo"] == LO
    assert config["operand_hi"] == HI
    assert config["base_steps"] == BASE_STEPS
    assert config["align_steps"] == STEPS
    assert config["eval_every"] == EVAL_EVERY
    assert config["lr_base"] == LR_BASE
    assert config["lr_align"] == LR_ALIGN
    assert config["seeds"] == list(SEEDS)
    assert config["methods"] == list(METHODS)


def test_demo_defaults_are_the_published_budget():
    args = cli._parser().parse_args(["demo"])
    assert args.steps == STEPS
    assert args.seed == SEEDS[0]


def test_demo_does_not_retune_the_study():
    source = inspect.getsource(cli)
    for literal in ("lr=2e-3", "lr=1e-3", "lr=0.002", "lr=0.001", "batch=128",
                    "steps=400", "eval_every=max"):
        assert literal not in source, f"{literal} bypasses alignlab.study"
