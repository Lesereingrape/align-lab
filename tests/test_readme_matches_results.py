"""The README must not drift from the repository, in either of its two claims.

The results block is exactly what ``make_report.py`` renders from
``results/alignment.json``, and the "~N-line" size claim in the opening paragraph is
checked against the package's actual line count. Nothing in the README is
hand-copied or hand-rounded.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_renderer():
    path = ROOT / "experiments" / "make_report.py"
    spec = importlib.util.spec_from_file_location("make_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _block(text: str) -> str:
    m = re.search(r"<!-- RESULTS:START -->\n(.*?)\n<!-- RESULTS:END -->", text, re.DOTALL)
    assert m, "README is missing the RESULTS:START/END block"
    return m.group(1).strip()


def test_readme_matches_committed_results():
    data = json.loads((ROOT / "results" / "alignment.json").read_text(encoding="utf-8"))
    rendered = _load_renderer().build(data).strip()
    readme_block = _block((ROOT / "README.md").read_text(encoding="utf-8"))
    assert readme_block == rendered, (
        "README results drift: run `python experiments/make_report.py --write` to splice the "
        "output into the RESULTS block."
    )


def test_readme_size_claim_matches_the_package():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"~(\d+)-line", text)
    assert m, "README no longer states its package size in lines"
    claimed = int(m.group(1))
    measured = sum(
        len(p.read_text(encoding="utf-8").splitlines())
        for p in sorted((ROOT / "src" / "alignlab").glob("*.py"))
    )
    assert abs(claimed - measured) <= 0.05 * measured, (
        f"README claims ~{claimed} lines; src/alignlab is {measured}. Update the prose."
    )
