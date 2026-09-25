"""The published experiment's configuration, in one importable place.

These constants live in the package rather than only in ``experiments/run_study.py``
so that ``alignlab demo`` cannot quietly train a smaller, different experiment than
the one the README table reports: the demo draws its dataset sizes, step budget,
batch, learning rate and eval cadence from here, and a test pins them to
``results/alignment.json``.
"""

from __future__ import annotations

SEEDS = (0, 1, 2)
METHODS = ("sft", "dpo", "orpo", "simpo")
N_PAIRS = 2500
N_EVAL = 600
LO, HI = 10, 99
BASE_STEPS = 200
STEPS = 600
EVAL_EVERY = 150
BATCH = 128
LR_BASE = 2e-3
LR_ALIGN = 1e-3
DPO_BETAS = (0.05, 0.1, 0.5)
