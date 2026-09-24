"""Render the README results block directly from results/alignment.json.

The README numbers are mechanically tied to the committed artifact: run
``python experiments/run_study.py`` then ``python experiments/make_report.py`` and
paste the output between the RESULTS markers. A test asserts the README already
equals this, so nothing is hand-copied.
"""

from __future__ import annotations

import json
from pathlib import Path

ORDER = ("sft", "orpo", "dpo", "simpo")


def _curve_rows(data: dict) -> list[str]:
    steps = [p["step"] for p in data["methods"]["sft"]["curve"]]
    header = "| step | " + " | ".join(f"{m} acc" for m in ORDER) + " |"
    sep = "|-----:|" + "|".join(["-------:"] * len(ORDER)) + "|"
    lines = [header, sep]
    for i, step in enumerate(steps):
        cells = [f"{data['methods'][m]['curve'][i]['answer_acc']:.3f}" for m in ORDER]
        lines.append(f"| {step} | " + " | ".join(cells) + " |")
    return lines


def build(data: dict) -> str:
    cfg = data["config"]
    base = data["base"]
    methods = data["methods"]
    out: list[str] = []

    out.append("*Every figure below is produced by `experiments/run_study.py` on CPU "
               "and committed as [`results/alignment.json`](results/alignment.json); "
               "the tables are rendered by `experiments/make_report.py`. All four "
               f"objectives start from the same {cfg['base_steps']}-step SFT base and "
               f"train {cfg['align_steps']} steps on {cfg['n_pairs']} preference pairs; "
               f"mean over {len(cfg['seeds'])} seeds.*")
    out.append("")
    out.append(f"- tiny model: **{cfg['params']:,}** parameters (decoder-only "
               "transformer, CPU-only)")
    out.append(f"- weak SFT base: greedy answer accuracy **{base['answer_acc']:.3f}** "
               f"(±{base['answer_acc_std']:.3f}), offline pair win-rate "
               f"{base['win_rate']:.3f}")
    out.append("")

    out.append("### Final held-out: generation accuracy vs offline preference fit\n")
    out.append("| objective | greedy answer acc | offline win-rate | mean margin |")
    out.append("|-----------|------------------:|-----------------:|------------:|")
    out.append(f"| SFT base (starting point) | {base['answer_acc']:.3f} | "
               f"{base['win_rate']:.3f} | {base['margin']:.2f} |")
    for m in ORDER:
        v = methods[m]
        out.append(f"| **{m.upper()}** | **{v['final_answer_acc']:.3f}** "
                   f"(±{v['final_answer_acc_std']:.3f}) | {v['final_win_rate']:.3f} | "
                   f"{v['final_margin']:.2f} |")
    out.append("")
    dpo = methods["dpo"]
    simpo = methods["simpo"]
    orpo = methods["orpo"]
    out.append(f"Read the two accuracy/win-rate columns *together*. On a task where "
               f"the preferred response is the fully correct chain, plain **SFT** and "
               f"the SFT-coupled **ORPO** stay at ceiling ({orpo['final_answer_acc']:.3f}). "
               f"The pure preference objectives do the opposite: **DPO** drives the "
               f"offline win-rate to {dpo['final_win_rate']:.3f} and the margin to "
               f"{dpo['final_margin']:.1f} yet its *greedy* answer accuracy collapses to "
               f"{dpo['final_answer_acc']:.3f} - **below the {base['answer_acc']:.2f} base** "
               f"- and SimPO shows the same ({simpo['final_answer_acc']:.3f} acc at "
               f"win-rate {simpo['final_win_rate']:.3f}).")
    out.append("")

    out.append("### Held-out greedy answer accuracy vs training step\n")
    out.extend(_curve_rows(data))
    out.append("")
    d0 = methods["dpo"]["curve"][1]["answer_acc"]
    out.append(f"The collapse is immediate: DPO is already at {d0:.3f} by the first "
               "eval. This is the documented *likelihood-displacement* pathology - a "
               "margin-only objective widens chosen-vs-rejected by pushing both "
               "absolute log-probs around, so it can raise the pair win-rate while the "
               "argmax chain drifts off the correct answer. Win-rate alone is a "
               "misleading alignment metric; measuring real generations is what "
               "exposes it. ORPO's built-in SFT term anchors the chosen likelihood and "
               "avoids the collapse.")
    out.append("")

    out.append("### Ablation: DPO preference strength (beta) vs accuracy\n")
    out.append("| beta | greedy answer acc |")
    out.append("|-----:|------------------:|")
    for b in sorted(data["dpo_beta_ablation"], key=float):
        out.append(f"| {b} | {data['dpo_beta_ablation'][b]:.3f} |")
    out.append("")
    lo = data["dpo_beta_ablation"][min(data["dpo_beta_ablation"], key=float)]
    hi = data["dpo_beta_ablation"][max(data["dpo_beta_ablation"], key=float)]
    out.append(f"Stronger preference pressure (larger beta) is not gentler here - it "
               f"displaces the policy *more* ({lo:.3f} -> {hi:.3f}), the opposite of the "
               "usual alignment intuition, again because the base is already "
               "near-saturated on win-rate and the only room left is to inflate the "
               "margin.")
    out.append("")

    out.append("### Honest limitations\n")
    out.append("- This is a deliberately toy, fully-verifiable task. Because the "
               "preferred response *is* the correct chain, SFT/ORPO have a structural "
               "advantage on raw accuracy that real human preference data (correct-but-"
               "verbose vs better) would not give them.")
    out.append("- The DPO/SimPO collapse is a real, reproducible failure mode, not a "
               "claim that these methods are generally bad: with a less-saturated base, "
               "shorter schedules, or an SFT anchor they behave normally.")
    out.append("- It is an elicitation/behavioural study of the objectives on a tiny "
               "model, not a claim about frontier-scale alignment.")
    return "\n".join(out)


if __name__ == "__main__":
    print(build(json.loads(Path("results/alignment.json").read_text(encoding="utf-8"))))
