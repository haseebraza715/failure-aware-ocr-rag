#!/usr/bin/env python3
"""Collect cluster hardware and quota facts without loading FAAR or models."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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


def _memory_bytes() -> dict[str, int | None]:
    total: int | None = None
    if hasattr(os, "sysconf"):
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            pass

    cgroup_limit: int | None = None
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        if path.is_file():
            cgroup_limit = _bytes_from_text(path.read_text(encoding="utf-8"))
            if cgroup_limit is not None:
                break
    return {"physical_bytes": total, "cgroup_limit_bytes": cgroup_limit}


def _disk(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {"total_bytes": usage.total, "used_bytes": usage.total - usage.free, "free_bytes": usage.free}


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
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


def collect(path: Path) -> dict[str, Any]:
    scheduler = {
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_mem_per_node": os.getenv("SLURM_MEM_PER_NODE"),
        "slurm_mem_per_cpu": os.getenv("SLURM_MEM_PER_CPU"),
        "pbs_job_id": os.getenv("PBS_JOBID"),
        "scheduler_detected": "slurm" if os.getenv("SLURM_JOB_ID") else "pbs" if os.getenv("PBS_JOBID") else None,
    }
    return {
        "schema_version": 1,
        "collected_at_utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory": _memory_bytes(),
        "disk": {"path": str(path), **_disk(path)},
        "scheduler": scheduler,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "torch": _torch_info(),
        "nvidia_smi": _run(["nvidia-smi", "-L"]) if shutil.which("nvidia-smi") else {"available": False},
        "relevant_environment": {
            key: os.getenv(key)
            for key in (
                "HF_HOME",
                "TRANSFORMERS_CACHE",
                "FAAR_VISUAL_BATCH_SIZE",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
            if os.getenv(key) is not None
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path.cwd(), help="Filesystem path whose quota should be measured")
    parser.add_argument("--out", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    payload = collect(args.path.resolve())
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
