"""Run the SFT-vs-DPO-vs-ORPO-vs-SimPO comparison and write results JSON.

Usage:  PYTHONPATH=src python experiments/run_study.py
Writes results/alignment.json; the README tables are rendered from that file by
make_report.py, so the numbers shipped in the README are exactly these.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from alignlab.model import count_parameters
from alignlab.train import build_base, evaluate_answer, evaluate_pairs, make_dataset, run_method

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


def _agg(curves: list[list[dict]]) -> list[dict]:
    """Mean/std across seeds at each recorded step index."""
    out = []
    for i in range(len(curves[0])):
        step = curves[0][i]["step"]
        aa = [c[i]["answer_acc"] for c in curves]
        wr = [c[i]["win_rate"] for c in curves]
        out.append({
            "step": step,
            "answer_acc": round(statistics.fmean(aa), 4),
            "answer_acc_std": round(statistics.pstdev(aa), 4),
            "win_rate": round(statistics.fmean(wr), 4),
        })
    return out


def run_seed(seed: int) -> dict:
    train_examples, train_pairs, eval_examples, eval_pairs = make_dataset(
        N_PAIRS, N_EVAL, seed, LO, HI)
    base = build_base(steps=BASE_STEPS, batch=BATCH, lr=LR_BASE,
                      seed=seed, examples=train_examples)
    base_m = {**evaluate_answer(base, eval_examples),
              **evaluate_pairs(base, eval_pairs)}
    base_m = {k: round(v, 4) for k, v in base_m.items()}

    curves: dict[str, list[dict]] = {}
    for name in METHODS:
        curves[name] = run_method(
            name, base=base, train_pairs=train_pairs,
            eval_examples=eval_examples, eval_pairs=eval_pairs,
            steps=STEPS, batch=BATCH, lr=LR_ALIGN, seed=seed + 1,
            eval_every=EVAL_EVERY)

    dpo_beta = {}
    for beta in DPO_BETAS:
        c = run_method("dpo", base=base, train_pairs=train_pairs,
                       eval_examples=eval_examples, eval_pairs=eval_pairs,
                       steps=STEPS, batch=BATCH, lr=LR_ALIGN, seed=seed + 1,
                       eval_every=EVAL_EVERY, beta=beta)
        dpo_beta[str(beta)] = round(c[-1]["answer_acc"], 4)

    return {"seed": seed, "params": count_parameters(base),
            "base": base_m, "curves": curves, "dpo_beta": dpo_beta}


def main() -> None:
    t0 = time.time()
    per_seed = [run_seed(s) for s in SEEDS]

    methods = {}
    for name in METHODS:
        seed_curves = [ps["curves"][name] for ps in per_seed]
        curve = _agg(seed_curves)
        finals = [ps["curves"][name][-1] for ps in per_seed]
        methods[name] = {
            "curve": curve,
            "final_answer_acc": round(statistics.fmean(f["answer_acc"] for f in finals), 4),
            "final_answer_acc_std": round(statistics.pstdev(
                [f["answer_acc"] for f in finals]), 4),
            "final_win_rate": round(statistics.fmean(f["win_rate"] for f in finals), 4),
            "final_margin": round(statistics.fmean(f["margin"] for f in finals), 4),
        }

    dpo_beta = {b: round(statistics.fmean(ps["dpo_beta"][b] for ps in per_seed), 4)
                for b in map(str, DPO_BETAS)}

    out = {
        "config": {
            "methods": list(METHODS), "seeds": list(SEEDS), "n_pairs": N_PAIRS,
            "n_eval": N_EVAL, "operand_lo": LO, "operand_hi": HI,
            "base_steps": BASE_STEPS, "align_steps": STEPS, "eval_every": EVAL_EVERY,
            "lr_base": LR_BASE, "lr_align": LR_ALIGN,
            "params": per_seed[0]["params"],
        },
        "base": {
            "answer_acc": round(statistics.fmean(ps["base"]["answer_acc"] for ps in per_seed), 4),
            "answer_acc_std": round(statistics.pstdev(
                [ps["base"]["answer_acc"] for ps in per_seed]), 4),
            "win_rate": round(statistics.fmean(ps["base"]["win_rate"] for ps in per_seed), 4),
            "margin": round(statistics.fmean(ps["base"]["margin"] for ps in per_seed), 4),
        },
        "methods": methods,
        "dpo_beta_ablation": dpo_beta,
        "runtime_sec": round(time.time() - t0, 1),
    }
    dest = Path("results/alignment.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest} in {out['runtime_sec']}s")


if __name__ == "__main__":
    main()
