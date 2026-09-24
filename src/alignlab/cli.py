"""Tiny CLI: ``alignlab demo`` trains the four objectives briefly and prints curves.

The full, committed results come from ``experiments/run_study.py``; this is just a
quick, self-contained smoke run to see the objectives move live.
"""

from __future__ import annotations

import argparse

from .train import build_base, evaluate_answer, evaluate_pairs, make_dataset, run_method


def demo(steps: int, seed: int) -> None:
    torch_seed = seed
    te, tp, ee, ep = make_dataset(1200, 400, torch_seed, 10, 99)
    base = build_base(steps=200, batch=128, lr=2e-3, seed=torch_seed, examples=te)
    m = {**evaluate_answer(base, ee), **evaluate_pairs(base, ep)}
    print(f"base   answer_acc={m['answer_acc']:.3f} win_rate={m['win_rate']:.3f} "
          f"margin={m['margin']:.2f}")
    for name in ("sft", "dpo", "orpo", "simpo"):
        c = run_method(name, base=base, train_pairs=tp, eval_examples=ee,
                       eval_pairs=ep, steps=steps, batch=128, lr=1e-3,
                       seed=torch_seed + 1, eval_every=max(1, steps // 3))
        last = c[-1]
        print(f"{name:5s} final answer_acc={last['answer_acc']:.3f} "
              f"win_rate={last['win_rate']:.3f} margin={last['margin']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="alignlab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="quick live comparison of the objectives")
    d.add_argument("--steps", type=int, default=400)
    d.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "demo":
        demo(args.steps, args.seed)


if __name__ == "__main__":
    main()
