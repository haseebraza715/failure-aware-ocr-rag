#!/usr/bin/env python3
"""Launch one resource-bounded FAAR process after a hardware preflight."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

GIB = 1024**3
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import preflight


class LaunchError(Exception):
    pass


def resolve_root(explicit: Path | None) -> Path:
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
    else:
        root = SCRIPT_DIR.parent
    if not root.is_dir():
        raise LaunchError(f"project root is not a directory: {root}")
    return root


def load_preflight(root: Path) -> dict[str, Any]:
    override = os.getenv("FAAR_PREFLIGHT_JSON")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise LaunchError(f"FAAR_PREFLIGHT_JSON does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LaunchError(f"FAAR_PREFLIGHT_JSON is not readable JSON: {path}") from exc
        if not isinstance(payload, dict) or "schema_version" not in payload:
            raise LaunchError(f"FAAR_PREFLIGHT_JSON is not a preflight payload: {path}")
        return payload
    return preflight.collect(root)


def check_one_gpu(preflight_payload: dict[str, Any], *, cpu_only: bool) -> None:
    if cpu_only:
        return
    torch_info = preflight_payload.get("torch") or {}
    devices = torch_info.get("devices") or []
    if torch_info.get("available") is False:
        raise LaunchError("torch unavailable in preflight; cannot confirm a single logical GPU.")
    count = len(devices)
    if count == 0:
        raise LaunchError(
            "No CUDA GPU found by preflight. Run with --cpu-only, or fix the allocation."
        )
    if count > 1:
        raise LaunchError(
            f"Expected exactly one logical GPU, but preflight found {count}. "
            "Request one scheduler GPU, or limit visibility before a non-scheduler launch."
        )
    total = int(devices[0].get("total_memory_bytes") or 0)
    free = int(devices[0].get("free_memory_bytes") or 0)
    if total and free / total < 0.3:
        raise LaunchError(
            f"Only {free / GIB:.2f} GiB of {total / GIB:.2f} GiB VRAM is free. "
            "Wait for at least 30% free VRAM so the job can retain a 20% co-tenant reserve."
        )


def derive_max_rss_gb(preflight_payload: dict[str, Any]) -> float:
    memory = preflight_payload.get("memory") or {}
    budget: int | None = memory.get("cgroup_limit_bytes")
    if not budget:
        scheduler = preflight_payload.get("scheduler") or {}
        budget = scheduler.get("slurm_mem_per_node_bytes") or scheduler.get("pbs_mem_bytes")
        per_cpu = scheduler.get("slurm_mem_per_cpu_bytes")
        cpus = scheduler.get("slurm_cpus_per_task") or scheduler.get("slurm_cpus_on_node") or scheduler.get("pbs_ncpus")
        if per_cpu and cpus:
            budget = per_cpu * int(cpus)
    if budget:
        return round(budget * 0.9 / GIB, 1)
    physical = memory.get("physical_bytes")
    if physical:
        return round(physical * 0.5 / GIB, 1)
    return 8.0


def derive_min_gpu_free_gb(preflight_payload: dict[str, Any], reserve_fraction: float = 0.2) -> float:
    devices = (preflight_payload.get("torch") or {}).get("devices") or []
    if not devices:
        return 0.0
    total = devices[0].get("total_memory_bytes") or 0
    if not total:
        return 0.0
    return round(total * reserve_fraction / GIB, 2)


def derive_gpu_memory_fraction(preflight_payload: dict[str, Any]) -> float:
    devices = (preflight_payload.get("torch") or {}).get("devices") or []
    if devices:
        total = devices[0].get("total_memory_bytes") or 0
        free = devices[0].get("free_memory_bytes") or 0
        if total:
            free_fraction = free / total
            return round(min(0.5, max(0.1, free_fraction - 0.2)), 3)
    return 0.5


def derive_threads(preflight_payload: dict[str, Any]) -> int:
    scheduler = preflight_payload.get("scheduler") or {}
    alloc = (
        scheduler.get("slurm_cpus_per_task")
        or scheduler.get("slurm_cpus_on_node")
        or scheduler.get("pbs_ncpus")
        or scheduler.get("pbs_num_ppn")
        or preflight_payload.get("cpu_count")
    )
    try:
        alloc = int(alloc)
    except (TypeError, ValueError):
        alloc = preflight_payload.get("cpu_count")
    if not alloc or alloc < 1:
        alloc = 1
    return max(1, alloc // 2)


def set_derived_env(preflight_payload: dict[str, Any], *, cpu_only: bool = False) -> dict[str, str]:
    derived: dict[str, str] = {}
    if os.getenv("FAAR_MAX_RSS_GB") is None:
        derived["FAAR_MAX_RSS_GB"] = str(derive_max_rss_gb(preflight_payload))
    if not cpu_only:
        if os.getenv("FAAR_MIN_GPU_FREE_GB") is None:
            derived["FAAR_MIN_GPU_FREE_GB"] = str(derive_min_gpu_free_gb(preflight_payload))
        if os.getenv("FAAR_MAX_GPU_MEMORY_FRACTION") is None:
            derived["FAAR_MAX_GPU_MEMORY_FRACTION"] = str(derive_gpu_memory_fraction(preflight_payload))
    threads = derive_threads(preflight_payload)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.getenv(name) is None:
            derived[name] = str(threads)
    for name, value in derived.items():
        os.environ[name] = value
    return derived


def hf_cache_candidates() -> list[Path]:
    explicit = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")
    if explicit:
        return [Path(explicit).expanduser()]
    return [Path.home() / ".cache" / "huggingface"]


def validate_hf_cache() -> None:
    for path in hf_cache_candidates():
        explicit = path != Path.home() / ".cache" / "huggingface"
        if not path.exists():
            if explicit:
                raise LaunchError(f"Configured HF cache does not exist: {path}")
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise LaunchError(f"Could not create HF cache directory {path}: {exc}") from exc
        if not os.access(path, os.W_OK):
            raise LaunchError(f"HF cache is not writable: {path}")


def required_key_for_backend(backend: str) -> str | None:
    if backend in {"openai", "oai"}:
        return "OPENAI_API_KEY"
    if backend in {"claude-sonnet-4-5", "anthropic", "claude"}:
        return "ANTHROPIC_API_KEY"
    return None


def _dotenv_has_nonempty_value(root: Path, required: str) -> bool:
    env_file = root / ".env"
    if not env_file.is_file():
        return False
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == required and value.strip().strip("'\""):
                return True
    except OSError:
        return False
    return False


def run_requires_vlm(run_args: list[str]) -> bool:
    values: dict[str, str] = {}
    for index, token in enumerate(run_args):
        if token in {"--mode", "--recovery", "--ablate"} and index + 1 < len(run_args):
            values[token] = run_args[index + 1]
        elif token.startswith("--mode="):
            values["--mode"] = token.split("=", 1)[1]
        elif token.startswith("--recovery="):
            values["--recovery"] = token.split("=", 1)[1]
        elif token.startswith("--ablate="):
            values["--ablate"] = token.split("=", 1)[1]
    return (
        values.get("--mode") in {"faar", "colpali", "visrag"}
        or "--ablate" in values
        or values.get("--recovery") in {"always_vlm", "random_type"}
    )


def validate_required_keys(root: Path, run_args: list[str] | None = None) -> None:
    if run_args is not None and not run_requires_vlm(run_args):
        return
    backend = os.getenv("VLM_BACKEND", "openai")
    required = required_key_for_backend(backend)
    if required is None:
        return
    present = bool(os.getenv(required, "").strip()) or _dotenv_has_nonempty_value(root, required)
    if not present:
        raise LaunchError(
            f"Missing required credential {required} for VLM_BACKEND={backend}. "
            f"Export it or add it to {root}/.env before starting; values are never printed."
        )


def build_child_command(root: Path, run_args: list[str], python_executable: str | None = None) -> list[str]:
    python = python_executable or os.getenv("FAAR_PYTHON") or sys.executable
    run_py = root / "run.py"
    if not run_py.is_file():
        raise LaunchError(f"run.py not found under project root: {run_py}")
    return [python, str(run_py), *run_args]


_CHILD: subprocess.Popen[str] | None = None


def run_child(argv: list[str], *, cwd: Path) -> int:
    global _CHILD

    def forward(signum: int, frame: Any) -> None:
        child = _CHILD
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    previous = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, forward)
    try:
        _CHILD = subprocess.Popen(argv, cwd=str(cwd), env=os.environ.copy())
        return _CHILD.wait()
    finally:
        _CHILD = None
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def print_summary(root: Path, cpu_only: bool, run_args: list[str]) -> None:
    mode = "cpu-only" if cpu_only else "single-gpu"
    print(f"[faar-launcher] mode={mode} project_root={root}", flush=True)
    for name in (
        "FAAR_MAX_RSS_GB",
        "FAAR_MIN_GPU_FREE_GB",
        "FAAR_MAX_GPU_MEMORY_FRACTION",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        print(f"[faar-launcher] {name}={os.getenv(name)}", flush=True)


def write_preflight_report(payload: dict[str, Any], path: Path) -> None:
    preflight._atomic_write_text(path, preflight.render(payload) + "\n")


def default_preflight_path(root: Path, payload: dict[str, Any]) -> Path:
    scheduler = payload.get("scheduler") or {}
    identifier = scheduler.get("slurm_job_id") or scheduler.get("pbs_job_id") or str(os.getpid())
    safe_identifier = str(identifier).replace("/", "_").replace("\\", "_")
    return root / f"results/environment/preflight_{safe_identifier}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project-root", type=Path, default=None, help="Repository root (default: parent of this file)")
    parser.add_argument("--cpu-only", action="store_true", help="Allow a CPU-only run without any CUDA GPU")
    parser.add_argument("--faar-python", default=None, help="Python used to launch run.py (default: this interpreter)")
    parser.add_argument("--preflight-out", type=Path, default=None, help="Saved redacted preflight JSON path")
    args, run_args = parser.parse_known_args(argv)
    try:
        root = resolve_root(args.project_root)
        payload = load_preflight(root)
        report_path = args.preflight_out or default_preflight_path(root, payload)
        if not report_path.is_absolute():
            report_path = root / report_path
        write_preflight_report(payload, report_path)
        check_one_gpu(payload, cpu_only=args.cpu_only)
        set_derived_env(payload, cpu_only=args.cpu_only)
        validate_hf_cache()
        validate_required_keys(root, run_args)
        print_summary(root, args.cpu_only, run_args)
        child_argv = build_child_command(root, run_args, args.faar_python)
        return run_child(child_argv, cwd=root)
    except LaunchError as exc:
        print(f"launcher: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
