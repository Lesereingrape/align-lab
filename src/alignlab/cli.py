"""Tiny CLI: ``alignlab demo`` trains the four objectives live at the published budget.

    python -m alignlab.cli demo            # seed 0, the 600 steps the table reports
    python -m alignlab.cli demo --steps 200  # explicitly shorter than the published run

The committed results come from ``experiments/run_study.py``; this command is one
seed of *that* experiment. Dataset sizes, base budget, batch, learning rates and
eval cadence all come from :mod:`alignlab.study`, so the four lines it prints are
the seed-0 versions of published columns rather than a smaller, differently-tuned
run standing in front of a table it does not match.
"""

from __future__ import annotations

import argparse

from .study import (
    BASE_STEPS,
    BATCH,
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
from .train import build_base, evaluate_answer, evaluate_pairs, make_dataset, run_method


def demo(steps: int = STEPS, seed: int = SEEDS[0]) -> None:
    train_examples, train_pairs, eval_examples, eval_pairs = make_dataset(
        N_PAIRS, N_EVAL, seed, LO, HI)
    base = build_base(steps=BASE_STEPS, batch=BATCH, lr=LR_BASE,
                      seed=seed, examples=train_examples)
    m = {**evaluate_answer(base, eval_examples), **evaluate_pairs(base, eval_pairs)}
    note = "" if steps == STEPS else f"   [truncated: {steps} of {STEPS} published steps]"
    print(f"seed {seed}  train_pairs={N_PAIRS}  eval={N_EVAL}{note}")
    print(f"base   answer_acc={m['answer_acc']:.3f} win_rate={m['win_rate']:.3f} "
          f"margin={m['margin']:.2f}")
    for name in METHODS:
        curve = run_method(name, base=base, train_pairs=train_pairs,
                           eval_examples=eval_examples, eval_pairs=eval_pairs,
                           steps=steps, batch=BATCH, lr=LR_ALIGN,
                           seed=seed + 1, eval_every=EVAL_EVERY)
        last = curve[-1]
        print(f"{name:5s} final answer_acc={last['answer_acc']:.3f} "
              f"win_rate={last['win_rate']:.3f} margin={last['margin']:.2f}")


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="alignlab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="live run of the published objectives")
    d.add_argument("--steps", type=int, default=STEPS)
    d.add_argument("--seed", type=int, default=SEEDS[0])
    return ap


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.cmd == "demo":
        demo(args.steps, args.seed)


if __name__ == "__main__":
    main()
