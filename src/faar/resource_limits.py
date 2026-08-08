from __future__ import annotations

import os
import sys
from typing import Any

try:
    import resource
except ImportError:
    resource = None


def _optional_gb(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number of GiB; received {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive number of GiB; received {value}.")
    return value


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if value <= 0:
        return None
    if sys.platform == "darwin":
        return value
    return value * 1024


def enforce_memory_budget(stage: str, torch_module: Any | None = None) -> None:
    """Fail before the next inference step when an optional budget is exceeded.

    The cluster scheduler remains the hard CPU-RAM limit. These checks provide a
    portable, fail-fast guard inside the process once the cluster-specific
    values are supplied through the environment.
    """

    max_rss_gb = _optional_gb("FAAR_MAX_RSS_GB")
    if max_rss_gb is not None:
        rss = _peak_rss_bytes()
        if rss is not None and rss > max_rss_gb * 1024**3:
            raise MemoryError(
                f"FAAR_MAX_RSS_GB exceeded before {stage}: "
                f"peak RSS is {rss / 1024**3:.2f} GiB, limit is {max_rss_gb:.2f} GiB."
            )

    min_gpu_free_gb = _optional_gb("FAAR_MIN_GPU_FREE_GB")
    if min_gpu_free_gb is None:
        return
    if torch_module is None:
        try:
            import torch as torch_module
        except Exception:
            return
    if not torch_module.cuda.is_available():
        return
    free_bytes, _ = torch_module.cuda.mem_get_info()
    if free_bytes < min_gpu_free_gb * 1024**3:
        raise MemoryError(
            f"FAAR_MIN_GPU_FREE_GB not available before {stage}: "
            f"free GPU memory is {free_bytes / 1024**3:.2f} GiB, "
            f"required reserve is {min_gpu_free_gb:.2f} GiB."
        )
