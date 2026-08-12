from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "cluster/templates"

EXPECTED_ENTRYPOINTS = {
    "slurm_preflight.sbatch": [
        ("cluster/preflight.py", {"--check", "--project-root", "--out"}),
    ],
    "slurm_calibration_108.sbatch": [
        ("cluster/launcher.py", {"--project-root", "--entrypoint", "--preflight-out"}),
        ("prepare_benchmark_assets.py", {"--dataset", "--out-root", "--smoke-doc", "--execute"}),
    ],
    "slurm_prepare_val_shard.sbatch": [
        ("cluster/launcher.py", {"--project-root", "--entrypoint", "--preflight-out"}),
        ("cluster/prepare_assets.py", {"--split", "--shard-index", "--num-shards", "--out-root", "--resume"}),
    ],
    "slurm_pilot_baseline.sbatch": [
        ("cluster/run_baselines.py", {"--project-root", "--dataset", "--split", "--max-examples", "--resume"}),
    ],
}

LAUNCHER_FLAGS = {"--project-root", "--entrypoint", "--preflight-out", "--cpu-only", "--faar-python"}


def _template_invocations(template: Path) -> list[tuple[str, set[str]]]:
    lines = template.read_text(encoding="utf-8").splitlines()
    starts = [index for index, line in enumerate(lines) if '"${FAAR_PYTHON}"' in line]
    blocks: list[list[str]] = []
    for start in starts:
        block = [lines[start]]
        for line in lines[start + 1 :]:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "export", "mkdir", "if", "done", "cd", "set")):
                break
            block.append(line)
        blocks.append(block)

    invocations: list[tuple[str, set[str]]] = []
    for block in blocks:
        tokens: list[str] = []
        for line in block:
            tokens.extend(shlex.split(line.replace('"${FAAR_PYTHON}"', "python").replace("\\", " ")))
        if len(tokens) < 2:
            continue
        script = tokens[1]
        flags = {t for t in tokens[2:] if t.startswith("--")}
        if script == "cluster/launcher.py":
            at = tokens.index("--entrypoint") if "--entrypoint" in tokens else None
            child_flags = {
                t
                for t in tokens[at + 2 :]
                if t.startswith("--") and t not in LAUNCHER_FLAGS
            } if at is not None else set()
            invocations.append((script, {t for t in flags if t in LAUNCHER_FLAGS}))
            if at is not None and tokens[at + 1]:
                invocations.append((tokens[at + 1], child_flags))
        else:
            invocations.append((script, flags))
    return invocations


@pytest.mark.parametrize("name", sorted(EXPECTED_ENTRYPOINTS))
def test_template_invokes_real_supported_arguments(name: str) -> None:
    expected = EXPECTED_ENTRYPOINTS[name]
    invocations = _template_invocations(TEMPLATES / name)
    assert invocations, f"{name} contains no python invocation"
    for script, flags in invocations:
        completed = subprocess.run(
            [sys.executable, str(ROOT / script), "--help"],
            check=False,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert completed.returncode == 0, f"{script} --help failed: {completed.stderr}"
        help_text = completed.stdout + completed.stderr
        for flag in flags:
            assert flag in help_text, f"{script} --help does not document {flag}"
    found = [(script, flags) for script, flags in invocations if any(flags)]
    for script, expected_flags in expected:
        actual = next((flags for s, flags in found if s == script), set())
        assert expected_flags.issubset(actual), f"{name} is missing {expected_flags - actual} for {script}"


def test_template_dry_run_flags_are_safe() -> None:
    for name, flags in _template_invocations(TEMPLATES / "slurm_preflight.sbatch"):
        assert "--dry-run" not in flags  # preflight --check must not be muted


def test_templates_never_hardcode_private_values() -> None:
    sensitive = re.compile(
        r"(haseeb|raza|\/Users\/|\/home\/|OPENAI_API_KEY=sk|sk-[A-Za-z0-9]{10,}|AKIA|\b[\w.+-]+@[\w-]+\.[\w.]+\b)",
        re.IGNORECASE,
    )
    for name in (
        "slurm_preflight.sbatch",
        "slurm_calibration_108.sbatch",
        "slurm_prepare_val_shard.sbatch",
        "slurm_pilot_baseline.sbatch",
    ):
        text = (TEMPLATES / name).read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            assert not sensitive.search(line), f"{name}:{line_no} looks hard-coded: {line.strip()}"
            if "PARTITION" in line and "EDIT" not in line:
                assert False, f"{name}:{line_no} has a non-placeholder partition"


def test_templates_include_signal_and_resume_behavior() -> None:
    calibration = (TEMPLATES / "slurm_calibration_108.sbatch").read_text()
    sharded = (TEMPLATES / "slurm_prepare_val_shard.sbatch").read_text()
    pilot = (TEMPLATES / "slurm_pilot_baseline.sbatch").read_text()
    assert "--signal=TERM@120" in calibration
    assert "--signal=TERM@120" in sharded
    assert "--signal=TERM@120" in pilot
    assert "--resume" in sharded
    assert "--resume" in pilot
