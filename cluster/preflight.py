#!/usr/bin/env python3
"""Collect cluster hardware and quota facts without loading FAAR models."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:
    resource = None

_SECRET_PATTERNS = tuple(
    re.compile(pattern) for pattern in (
        r"(sk-[A-Za-z0-9_-]{8,})",
        r"(AKIA[0-9A-Z]{16})",
        r"(AIza[0-9A-Za-z_-]{20,})",
        r"(ghp_[0-9A-Za-z]{20,})",
        r"(xox[baprs]-[0-9A-Za-z-]{10,})",
        r"([\w-]*API[_-]KEY[=:\s]+[^\s,}]+)",
        r"([\w-]*SECRET[=:\s]+[^\s,}]+)",
    )
)


def _run(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _bytes_from_text(raw: str) -> int | None:
    raw = raw.strip()
    if not raw or raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None


def _parse_memory_bytes(raw: str | None, default_multiplier: int = 1024**2) -> int | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text or text.lower() in {"max", "unlimited", "0"}:
        return None
    match = re.fullmatch(r"(\d+)\s*([KMGTP])?i?[bB]?", text, flags=re.IGNORECASE)
    if not match:
        return None
    value = int(match.group(1))
    suffix = (match.group(2) or "").upper()
    multiplier = default_multiplier if not suffix else {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}[suffix]
    return value * multiplier


def _parse_walltime_seconds(raw: str | None) -> int | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    if "-" in text:
        days_part, clock = text.split("-", 1)
        try:
            days = int(days_part)
        except ValueError:
            return None
    else:
        days = 0
        clock = text
    parts = clock.split(":")
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        try:
            return int(text) * 60
        except ValueError:
            return None
    if len(numbers) == 3:
        return days * 86400 + numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    if len(numbers) == 2:
        return days * 86400 + numbers[0] * 60 + numbers[1]
    if len(numbers) == 1:
        return numbers[0] * 60
    return None


def _parse_gpu_count(raw: str | None, *, id_list: bool = False) -> int | None:
    if raw is None:
        return None
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        return None

    def token_count(token: str, id_is_one: bool) -> int:
        parts = token.split(":")
        if token.isdigit():
            return 1 if id_is_one else int(token)
        if parts[-1].isdigit():
            return int(parts[-1])
        return 0

    if len(tokens) == 1:
        return token_count(tokens[0], id_is_one=id_list) or None
    total = sum(token_count(token, id_is_one=True) for token in tokens)
    return total or None


def _cgroup_memory() -> dict[str, int | None]:
    v2 = {
        "current": Path("/sys/fs/cgroup/memory.current"),
        "peak": Path("/sys/fs/cgroup/memory.peak"),
        "limit": Path("/sys/fs/cgroup/memory.max"),
    }
    v1 = {
        "current": Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        "peak": Path("/sys/fs/cgroup/memory/memory.max_usage_in_bytes"),
        "limit": Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    }
    source = "none"
    payload: dict[str, int | None] = {"current_bytes": None, "peak_bytes": None, "limit_bytes": None}
    for mapping, version in ((v2, "v2"), (v1, "v1")):
        if any(path.is_file() for path in mapping.values()):
            source = version
            for key, path in mapping.items():
                if path.is_file():
                    try:
                        payload[f"{key}_bytes"] = _bytes_from_text(path.read_text(encoding="utf-8"))
                    except OSError:
                        pass
            break
    payload["controller"] = source
    return payload


def _memory_bytes() -> dict[str, int | None]:
    total: int | None = None
    if hasattr(os, "sysconf"):
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            pass
    cgroup = _cgroup_memory()
    cgroup_limit = cgroup.get("limit_bytes")
    if total and cgroup_limit and cgroup_limit > total:
        cgroup_limit = None
    return {
        "physical_bytes": total,
        "cgroup_limit_bytes": cgroup_limit,
        "cgroup_current_bytes": cgroup.get("current_bytes"),
        "cgroup_peak_bytes": cgroup.get("peak_bytes"),
        "cgroup_controller": cgroup.get("controller"),
    }


def _ulimit() -> dict[str, Any]:
    payload: dict[str, Any] = {"available": resource is not None}
    if resource is not None:
        for name, constant in (
            ("RLIMIT_AS", resource.RLIMIT_AS),
            ("RLIMIT_DATA", resource.RLIMIT_DATA),
            ("RLIMIT_RSS", resource.RLIMIT_RSS),
            ("RLIMIT_NOFILE", resource.RLIMIT_NOFILE),
            ("RLIMIT_NPROC", resource.RLIMIT_NPROC),
            ("RLIMIT_STACK", resource.RLIMIT_STACK),
        ):
            try:
                soft, hard = resource.getrlimit(constant)
                payload[name] = {"soft": soft, "hard": hard}
            except (ValueError, OSError):
                continue
    return payload


def _disk(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {"total_bytes": usage.total, "used_bytes": usage.total - usage.free, "free_bytes": usage.free}


def _writable_free(path: Path) -> dict[str, Any]:
    try:
        exists = path.exists()
    except OSError:
        exists = False
    writable = exists and os.access(path, os.W_OK)
    free_bytes: int | None = None
    total_bytes: int | None = None
    if exists:
        try:
            usage = shutil.disk_usage(path)
            free_bytes, total_bytes = usage.free, usage.total
        except OSError:
            pass
    return {
        "path": str(path),
        "exists": exists,
        "writable": writable,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
    }


def _locations_from_env(names: tuple[str, ...]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in names:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            continue
        path = Path(raw).expanduser()
        key = os.path.normcase(str(path))
        if key in seen:
            continue
        seen.add(key)
        results.append({"env": name, **_writable_free(path)})
    return results


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except (ImportError, OSError) as exc:
        return {"available": False, "error": f"torch import failed: {exc}"}

    payload: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "devices": [],
    }
    if torch.cuda.is_available():
        driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"])
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            payload["devices"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "free_memory_bytes": free_bytes,
                    "driver_version": driver.get("stdout") if driver.get("available") else None,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "device_count": torch.cuda.device_count(),
                }
            )
    return payload


def _scheduler() -> dict[str, Any]:
    is_slurm = os.getenv("SLURM_JOB_ID") is not None
    is_pbs = os.getenv("PBS_JOBID") is not None
    slurm_job_gpus = os.getenv("SLURM_JOB_GPUS")
    return {
        "scheduler_detected": "slurm" if is_slurm else "pbs" if is_pbs else None,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_job_name": os.getenv("SLURM_JOB_NAME"),
        "slurm_partition": os.getenv("SLURM_JOB_PARTITION") or os.getenv("SLURM_PARTITION"),
        "slurm_qos": os.getenv("SLURM_JOB_QOS"),
        "slurm_num_nodes": _int_env("SLURM_JOB_NUM_NODES"),
        "slurm_cpus_on_node": _int_env("SLURM_CPUS_ON_NODE"),
        "slurm_cpus_per_task": _int_env("SLURM_CPUS_PER_TASK"),
        "slurm_mem_per_node": os.getenv("SLURM_MEM_PER_NODE"),
        "slurm_mem_per_node_bytes": _parse_memory_bytes(os.getenv("SLURM_MEM_PER_NODE")),
        "slurm_mem_per_cpu": os.getenv("SLURM_MEM_PER_CPU"),
        "slurm_mem_per_cpu_bytes": _parse_memory_bytes(os.getenv("SLURM_MEM_PER_CPU")),
        "slurm_time_limit_seconds": _parse_walltime_seconds(os.getenv("SLURM_TIME_LIMIT")),
        "slurm_gpus_requested": (
            _parse_gpu_count(slurm_job_gpus, id_list=True)
            if slurm_job_gpus is not None
            else _parse_gpu_count(os.getenv("SLURM_GPUS"))
        ),
        "slurm_gpus_on_node": _parse_gpu_count(os.getenv("SLURM_GPUS_ON_NODE")),
        "pbs_job_id": os.getenv("PBS_JOBID"),
        "pbs_job_name": os.getenv("PBS_JOBNAME"),
        "pbs_queue": os.getenv("PBS_O_QUEUE") or os.getenv("PBS_QUEUE"),
        "pbs_num_nodes": _int_env("PBS_NUM_NODES"),
        "pbs_num_ppn": _int_env("PBS_NUM_PPN"),
        "pbs_ncpus": _int_env("PBS_NCPUS"),
        "pbs_mem": _pbs_mem_raw(),
        "pbs_mem_bytes": _parse_memory_bytes(_pbs_mem_raw()),
        "pbs_gpus_requested": _parse_gpu_count(os.getenv("PBS_GPUS")),
        "pbs_gpus_on_node": _pbs_gpufile_count(),
        "pbs_walltime_seconds": _parse_walltime_seconds(os.getenv("PBS_WALLTIME")),
    }


def _pbs_mem_raw() -> str | None:
    return next((os.getenv(name) for name in ("PBS_RESC_MEM", "PBS_MEM", "PBS_MEMORY") if os.getenv(name)), None)


def _pbs_gpufile_count() -> int | None:
    raw = os.getenv("PBS_GPUFILE")
    if not raw:
        return None
    gpufile = Path(raw).expanduser()
    try:
        if gpufile.is_file():
            return sum(1 for line in gpufile.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        pass
    return _parse_gpu_count(raw)


def redact(value: Any) -> Any:
    if isinstance(value, str):
        masked = value
        for pattern in _SECRET_PATTERNS:
            masked = pattern.sub("<redacted>", masked)
        return masked
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    return value


def render(payload: dict[str, Any]) -> str:
    return json.dumps(redact(payload), indent=2, sort_keys=True)


def collect(path: Path) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "collected_at_utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory": _memory_bytes(),
        "disk": {"path": str(path), **_disk(path)},
        "scheduler": _scheduler(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "torch": _torch_info(),
        "nvidia_smi": _run(["nvidia-smi", "-L"]) if shutil.which("nvidia-smi") else {"available": False},
        "ulimit": _ulimit(),
        "scratch": _locations_from_env(("TMPDIR", "TMP", "TEMP", "SCRATCH", "SCRATCH_DIR", "LOCAL_SCRATCH", "TMPFS")),
        "cache": _locations_from_env(("HF_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME")),
        "relevant_environment": {
            key: os.getenv(key)
            for key in (
                "HF_HOME",
                "TRANSFORMERS_CACHE",
                "FAAR_VISUAL_BATCH_SIZE",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "FAAR_MAX_RSS_GB",
                "FAAR_MIN_GPU_FREE_GB",
                "FAAR_MAX_GPU_MEMORY_FRACTION",
            )
            if os.getenv(key) is not None
        },
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            temp.unlink()
        except OSError:
            pass
        raise
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path.cwd(), help="Filesystem path whose quota should be measured")
    parser.add_argument("--out", type=Path, help="Optional JSON output path (written atomically)")
    args = parser.parse_args()
    payload = collect(args.path.resolve())
    rendered = render(payload)
    print(rendered)
    if args.out:
        _atomic_write_text(args.out, rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
