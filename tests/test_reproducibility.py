"""Runs every lesson script and checks its stdout against the lesson doc's
"Expected output" block (mirrored in tests/expected/). If a script's output
legitimately changes, update the doc AND the matching tests/expected/*.txt
file in the same PR — see CONTRIBUTING.md.

The comparison is tolerant of tiny floating-point differences: the non-numeric
structure of every line must match exactly, but numbers only have to agree
within ABS_TOL. Neural-network / deep-learning lessons train with mini-batch
SGD, and the last decimals of a loss or accuracy depend on the CPU (which BLAS
kernel it picks, whether it fuses multiply-adds), so bit-identical output across
different machines is not achievable. Deterministic lessons still match exactly
because their numbers are identical — the tolerance simply never has to fire.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

# How far a single printed number may drift before the test fails. 0.05 easily
# covers observed cross-CPU wobble (a handful of borderline predictions flipping
# an accuracy by <1%) while still catching a genuinely broken script.
ABS_TOL = 0.05

# Matches an integer or decimal, optionally signed: 42, -0.5, 0.7080, 101770.
_NUMBER = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_DIR = Path(__file__).resolve().parent / "expected"

SCRIPTS = [
    ("lesson01_from_scratch", "src/lesson01_descriptive_stats/from_scratch.py"),
    ("lesson01_with_library", "src/lesson01_descriptive_stats/with_library.py"),
    ("lesson02_from_scratch", "src/lesson02_probability/from_scratch.py"),
    ("lesson02_with_library", "src/lesson02_probability/with_library.py"),
    ("lesson03_from_scratch", "src/lesson03_hypothesis_testing/from_scratch.py"),
    ("lesson03_with_library", "src/lesson03_hypothesis_testing/with_library.py"),
    ("lesson04_from_scratch", "src/lesson04_linear_regression/from_scratch.py"),
    ("lesson04_with_library", "src/lesson04_linear_regression/with_library.py"),
    ("lesson05_from_scratch", "src/lesson05_logistic_regression/from_scratch.py"),
    ("lesson05_with_library", "src/lesson05_logistic_regression/with_library.py"),
    ("lesson06_from_scratch", "src/lesson06_model_evaluation/from_scratch.py"),
    ("lesson06_with_library", "src/lesson06_model_evaluation/with_library.py"),
    ("lesson07_from_scratch", "src/lesson07_decision_trees/from_scratch.py"),
    ("lesson07_with_library", "src/lesson07_decision_trees/with_library.py"),
    ("lesson08_from_scratch", "src/lesson08_ensembles/from_scratch.py"),
    ("lesson08_with_library", "src/lesson08_ensembles/with_library.py"),
    ("lesson09_from_scratch", "src/lesson09_clustering/from_scratch.py"),
    ("lesson09_with_library", "src/lesson09_clustering/with_library.py"),
    ("lesson10_from_scratch", "src/lesson10_pca/from_scratch.py"),
    ("lesson10_with_library", "src/lesson10_pca/with_library.py"),
    ("lesson11_from_scratch", "src/lesson11_neural_networks/from_scratch.py"),
    ("lesson12_with_library", "src/lesson12_deep_learning/with_library.py"),
    ("lesson12_cnn", "src/lesson12_deep_learning/cnn.py"),
]

NAME_MAP = {"scikit-learn": "sklearn"}


def assert_output_matches(got: str, expected: str, script: str, name: str) -> None:
    """Compare stdout to the expected block line by line: the text around the
    numbers must be identical, and each number must agree within ABS_TOL."""
    got_lines = got.rstrip("\n").splitlines()
    exp_lines = expected.rstrip("\n").splitlines()
    hint = (
        f"{script} no longer matches its lesson doc's Expected output block. "
        f"If the change is intentional, update the doc and tests/expected/{name}.txt together."
    )
    assert len(got_lines) == len(exp_lines), (
        f"{hint}\nLine count differs: got {len(got_lines)}, expected {len(exp_lines)}."
    )
    for i, (g, e) in enumerate(zip(got_lines, exp_lines), start=1):
        # Blank out the numbers; the surrounding text must match character for character.
        assert _NUMBER.sub("#", g) == _NUMBER.sub("#", e), (
            f"{hint}\nLine {i} text differs:\n  got:      {g!r}\n  expected: {e!r}"
        )
        g_nums = [float(x) for x in _NUMBER.findall(g)]
        e_nums = [float(x) for x in _NUMBER.findall(e)]
        for gn, en in zip(g_nums, e_nums):
            assert abs(gn - en) <= ABS_TOL, (
                f"{hint}\nLine {i} number differs by more than {ABS_TOL}:\n"
                f"  got:      {g!r}\n  expected: {e!r}"
            )


def _pinned_versions():
    reqs = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        pkg, _, version = line.partition("==")
        reqs[pkg] = version
    return reqs


@pytest.mark.parametrize("name,script", SCRIPTS, ids=[s[0] for s in SCRIPTS])
def test_script_matches_doc(name, script):
    expected = (EXPECTED_DIR / f"{name}.txt").read_text()
    result = subprocess.run(
        [sys.executable, script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"{script} exited {result.returncode}:\n{result.stderr}"
    assert_output_matches(result.stdout, expected, script, name)


def test_verify_setup_reports_pinned_versions():
    result = subprocess.run(
        [sys.executable, "scripts/verify_setup.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "Setup OK" in result.stdout
    for pkg, version in _pinned_versions().items():
        printed_name = NAME_MAP.get(pkg, pkg)
        assert f"{printed_name} {version}" in result.stdout, (
            f"verify_setup.py did not report {printed_name} {version} — "
            f"is the environment installed from the pinned requirements.txt?"
        )
