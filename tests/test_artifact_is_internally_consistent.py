"""The committed artifact must agree with itself, not just with the README.

``methods[m]["curve"]`` is a mean over seeds and ``answer_acc_per_seed`` is the raw
material behind it, so the two have to reconcile at every checkpoint. A mean and a
per-seed list that disagree would mean one of them was produced by a different run —
exactly the failure mode that is invisible in a table and obvious in a test.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _data() -> dict:
    return json.loads((ROOT / "results" / "alignment.json").read_text(encoding="utf-8"))


def test_per_seed_curves_average_to_the_published_curve():
    data = _data()
    for name, method in data["methods"].items():
        per_seed = method["answer_acc_per_seed"]
        assert len(per_seed) == len(data["config"]["seeds"]), name
        for i, point in enumerate(method["curve"]):
            across = statistics.fmean(v[i] for v in per_seed.values())
            assert abs(across - point["answer_acc"]) < 2e-4, f"{name} checkpoint {i}"
        finals = [v[-1] for v in per_seed.values()]
        assert abs(statistics.fmean(finals) - method["final_answer_acc"]) < 2e-4, name
        assert abs(statistics.pstdev(finals) - method["final_answer_acc_std"]) < 2e-4, name


def test_base_accuracy_is_the_mean_of_its_per_seed_values():
    data = _data()
    per_seed = data["base"]["answer_acc_per_seed"]
    assert abs(statistics.fmean(per_seed.values()) - data["base"]["answer_acc"]) < 2e-4
    assert abs(statistics.pstdev(per_seed.values())
               - data["base"]["answer_acc_std"]) < 2e-4


def test_the_run_records_the_environment_it_is_bit_exact_under():
    env = _data()["environment"]
    assert env["device"] == "cpu"
    assert env["threads"] >= 1
    assert env["torch"].startswith("2.")
    assert env["python"] and env["platform"]
