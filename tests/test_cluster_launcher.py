from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

CLUSTER = Path(__file__).resolve().parents[1] / "cluster"
PREFLIGHT_SPEC = spec_from_file_location("preflight", CLUSTER / "preflight.py")
assert PREFLIGHT_SPEC and PREFLIGHT_SPEC.loader
preflight = module_from_spec(PREFLIGHT_SPEC)
sys.modules["preflight"] = preflight
PREFLIGHT_SPEC.loader.exec_module(preflight)
LAUNCHER_SPEC = spec_from_file_location("launcher", CLUSTER / "launcher.py")
assert LAUNCHER_SPEC and LAUNCHER_SPEC.loader
launcher = module_from_spec(LAUNCHER_SPEC)
sys.modules["launcher"] = launcher
LAUNCHER_SPEC.loader.exec_module(launcher)

GIB = 1024**3


@pytest.fixture(autouse=True)
def isolate_launcher_env(monkeypatch) -> None:
    for name in (
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "VLM_BACKEND",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "FAAR_PREFLIGHT_JSON",
        "FAAR_PYTHON",
        "CUDA_VISIBLE_DEVICES",
        "TMPDIR",
        "TMP",
        "TEMP",
        "SCRATCH",
        "SCRATCH_DIR",
        "LOCAL_SCRATCH",
    ):
        monkeypatch.delenv(name, raising=False)


def _gpu_payload(*, devices: list[dict], cpu_count: int = 8, physical_bytes: int = 64 * GIB) -> dict:
    return {
        "schema_version": 2,
        "cpu_count": cpu_count,
        "memory": {"physical_bytes": physical_bytes, "cgroup_limit_bytes": None},
        "scheduler": {},
        "torch": {"available": True, "cuda_available": bool(devices), "devices": devices},
        "scratch": [],
        "cache": [],
    }


def test_required_key_for_backend() -> None:
    assert launcher.required_key_for_backend("openai") == "OPENAI_API_KEY"
    assert launcher.required_key_for_backend("oai") == "OPENAI_API_KEY"
    assert launcher.required_key_for_backend("claude-sonnet-4-5") == "ANTHROPIC_API_KEY"
    assert launcher.required_key_for_backend("anthropic") == "ANTHROPIC_API_KEY"
    assert launcher.required_key_for_backend("mock") is None


def test_build_child_command_plain_argv_no_shell(tmp_path: Path) -> None:
    (tmp_path / "run.py").write_text("", encoding="utf-8")
    argv = launcher.build_child_command(tmp_path, ["--gate", "on", "--out", "results/x.json"], "/usr/bin/python3")
    assert argv == ["/usr/bin/python3", str(tmp_path / "run.py"), "--gate", "on", "--out", "results/x.json"]
    assert all(isinstance(item, str) for item in argv)


def test_build_child_command_missing_runpy_fails(tmp_path: Path) -> None:
    with pytest.raises(launcher.LaunchError, match="run.py not found"):
        launcher.build_child_command(tmp_path, [], "/usr/bin/python3")


def test_default_preflight_path_is_job_namespaced(tmp_path: Path) -> None:
    payload = _gpu_payload(devices=[])
    payload["scheduler"] = {"pbs_job_id": "123.cluster"}
    assert launcher.default_preflight_path(tmp_path, payload) == (
        tmp_path / "results/environment/preflight_123.cluster.json"
    )


def test_check_one_gpu_rejects_zero_and_multiple() -> None:
    with pytest.raises(launcher.LaunchError, match="No CUDA GPU"):
        launcher.check_one_gpu(_gpu_payload(devices=[]), cpu_only=False)
    many = [{"index": 0}, {"index": 1}]
    with pytest.raises(launcher.LaunchError, match="exactly one logical GPU"):
        launcher.check_one_gpu(_gpu_payload(devices=many), cpu_only=False)
    launcher.check_one_gpu(_gpu_payload(devices=[{"index": 0}]), cpu_only=False)


def test_check_one_gpu_cpu_only_skips_zero_gpus() -> None:
    launcher.check_one_gpu(_gpu_payload(devices=[]), cpu_only=True)


def test_check_one_gpu_rejects_unavailable_torch() -> None:
    payload = _gpu_payload(devices=[])
    payload["torch"] = {"available": False, "cuda_available": False, "devices": []}
    with pytest.raises(launcher.LaunchError, match="torch unavailable"):
        launcher.check_one_gpu(payload, cpu_only=False)


def test_derive_limits_from_preflight() -> None:
    payload = _gpu_payload(
        devices=[{"index": 0, "total_memory_bytes": 16 * GIB, "free_memory_bytes": 16 * GIB}],
        cpu_count=8,
        physical_bytes=64 * GIB,
    )
    assert launcher.derive_max_rss_gb(payload) == 32.0
    assert launcher.derive_min_gpu_free_gb(payload) == 3.2
    assert launcher.derive_gpu_memory_fraction(payload) == 0.5
    assert launcher.derive_threads(payload) == 4


def test_derive_limits_respect_scheduler_budget() -> None:
    payload = _gpu_payload(devices=[{"total_memory_bytes": 16 * GIB, "free_memory_bytes": 16 * GIB}])
    payload["memory"]["cgroup_limit_bytes"] = 20 * GIB
    assert launcher.derive_max_rss_gb(payload) == 18.0
    payload["scheduler"] = {"slurm_mem_per_node_bytes": 40 * GIB}
    payload["memory"]["cgroup_limit_bytes"] = None
    assert launcher.derive_max_rss_gb(payload) == 36.0


def test_derive_gpu_fraction_bounded_when_busy() -> None:
    payload = _gpu_payload(devices=[{"total_memory_bytes": 16 * GIB, "free_memory_bytes": 4 * GIB}])
    assert launcher.derive_gpu_memory_fraction(payload) == 0.1
    payload = _gpu_payload(devices=[{"total_memory_bytes": 16 * GIB, "free_memory_bytes": 2 * GIB}])
    assert launcher.derive_gpu_memory_fraction(payload) == 0.1


def test_check_one_gpu_rejects_busy_shared_device() -> None:
    payload = _gpu_payload(devices=[{"total_memory_bytes": 16 * GIB, "free_memory_bytes": 4 * GIB}])
    with pytest.raises(launcher.LaunchError, match="at least 30% free VRAM"):
        launcher.check_one_gpu(payload, cpu_only=False)


def test_derive_threads_falls_back_to_cpu_count() -> None:
    payload = _gpu_payload(devices=[], cpu_count=12)
    assert launcher.derive_threads(payload) == 6
    payload["scheduler"] = {"slurm_cpus_per_task": 2}
    assert launcher.derive_threads(payload) == 1


def test_set_derived_env_only_when_absent(monkeypatch) -> None:
    payload = _gpu_payload(devices=[{"total_memory_bytes": 16 * GIB, "free_memory_bytes": 16 * GIB}])
    monkeypatch.delenv("FAAR_MAX_RSS_GB", raising=False)
    monkeypatch.delenv("FAAR_MIN_GPU_FREE_GB", raising=False)
    monkeypatch.delenv("FAAR_MAX_GPU_MEMORY_FRACTION", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    derived = launcher.set_derived_env(payload)
    assert derived["FAAR_MAX_RSS_GB"] == "32.0"
    assert derived["FAAR_MIN_GPU_FREE_GB"] == "3.2"
    assert derived["FAAR_MAX_GPU_MEMORY_FRACTION"] == "0.5"
    assert derived["OMP_NUM_THREADS"] == "4"
    assert derived["MKL_NUM_THREADS"] == "4"

    monkeypatch.setenv("FAAR_MAX_RSS_GB", "99")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    derived2 = launcher.set_derived_env(payload)
    assert "FAAR_MAX_RSS_GB" not in derived2
    assert "OMP_NUM_THREADS" not in derived2
    assert os.environ["FAAR_MAX_RSS_GB"] == "99"
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_cpu_only_does_not_derive_gpu_guards(monkeypatch) -> None:
    payload = _gpu_payload(devices=[])
    derived = launcher.set_derived_env(payload, cpu_only=True)
    assert "FAAR_MIN_GPU_FREE_GB" not in derived
    assert "FAAR_MAX_GPU_MEMORY_FRACTION" not in derived
    assert "FAAR_MIN_GPU_FREE_GB" not in os.environ
    assert "FAAR_MAX_GPU_MEMORY_FRACTION" not in os.environ


def test_validate_hf_cache_default_created(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cache_dir = tmp_path / ".cache" / "huggingface"
    launcher.validate_hf_cache()
    assert cache_dir.is_dir()


def test_validate_hf_cache_explicit_missing_fails(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing-hf"
    monkeypatch.setenv("HF_HOME", str(missing))
    with pytest.raises(launcher.LaunchError, match="does not exist"):
        launcher.validate_hf_cache()


def test_validate_hf_cache_unwritable_fails(monkeypatch, tmp_path: Path) -> None:
    cache_dir = tmp_path / "hf"
    cache_dir.mkdir()
    monkeypatch.setenv("HF_HOME", str(cache_dir))
    monkeypatch.setattr(launcher.os, "access", lambda path, mode: False)
    with pytest.raises(launcher.LaunchError, match="not writable"):
        launcher.validate_hf_cache()


def test_validate_required_keys_env_and_dotenv(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VLM_BACKEND", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(launcher.LaunchError, match="OPENAI_API_KEY"):
        launcher.validate_required_keys(tmp_path)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    launcher.validate_required_keys(tmp_path)

    monkeypatch.delenv("OPENAI_API_KEY")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-from-dotenv\nANTHROPIC_API_KEY=sk-ant-from-dotenv\n", encoding="utf-8")
    launcher.validate_required_keys(tmp_path)

    monkeypatch.setenv("VLM_BACKEND", "claude-sonnet-4-5")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / ".env").unlink()
    with pytest.raises(launcher.LaunchError, match="ANTHROPIC_API_KEY"):
        launcher.validate_required_keys(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    launcher.validate_required_keys(tmp_path)


def test_empty_key_values_are_rejected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    with pytest.raises(launcher.LaunchError, match="OPENAI_API_KEY"):
        launcher.validate_required_keys(tmp_path, ["--mode", "faar"])
    monkeypatch.delenv("OPENAI_API_KEY")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=\n", encoding="utf-8")
    with pytest.raises(launcher.LaunchError, match="OPENAI_API_KEY"):
        launcher.validate_required_keys(tmp_path, ["--mode", "faar"])


def test_b0_does_not_require_a_vlm_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = ["--gate", "off", "--recovery", "off", "--out", "results/b0.json"]
    assert launcher.run_requires_vlm(args) is False
    launcher.validate_required_keys(tmp_path, args)


@pytest.mark.parametrize(
    "args",
    [
        ["--mode", "faar"],
        ["--mode=colpali"],
        ["--ablate", "no_gate"],
        ["--recovery", "always_vlm"],
        ["--recovery=random_type"],
    ],
)
def test_vlm_profiles_require_a_key(monkeypatch, tmp_path: Path, args: list[str]) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert launcher.run_requires_vlm(args) is True
    with pytest.raises(launcher.LaunchError, match="OPENAI_API_KEY"):
        launcher.validate_required_keys(tmp_path, args)


def test_run_child_propagates_exit_code(tmp_path: Path) -> None:
    code = launcher.run_child([sys.executable, "-c", "import sys; sys.exit(7)"], cwd=tmp_path)
    assert code == 7


def test_signal_with_no_child_yet_is_remembered(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "_CHILD", None)
    monkeypatch.setattr(launcher, "_pending_signal", None)
    monkeypatch.setattr(launcher, "_forwarded_signal", None)
    launcher._forward(signal.SIGINT, None)
    assert launcher._pending_signal == signal.SIGINT
    assert launcher._forwarded_signal == signal.SIGINT
    assert launcher._CHILD is None


def test_pending_signal_before_popen_prevents_spawn(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "_pending_signal", signal.SIGTERM)
    monkeypatch.setattr(launcher, "_forwarded_signal", None)
    spawns = []
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: spawns.append(args))
    code = launcher.run_child([sys.executable, "-c", "pass"], cwd=tmp_path)
    assert code == 128 + signal.SIGTERM
    assert spawns == []


def test_forwarded_signal_child_exits_zero_yields_128_plus_signal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "_CHILD", None)
    monkeypatch.setattr(launcher, "_pending_signal", None)
    monkeypatch.setattr(launcher, "_forwarded_signal", None)
    forwarded = []

    class _FakeProc:
        def poll(self) -> None:
            return None

        def send_signal(self, signum: int) -> None:
            forwarded.append(signum)

        def wait(self) -> int:
            return 0

    def _popen_with_delayed_signal(*args, **kwargs) -> _FakeProc:
        os.kill(os.getpid(), signal.SIGTERM)
        return _FakeProc()

    monkeypatch.setattr(launcher.subprocess, "Popen", _popen_with_delayed_signal)
    code = launcher.run_child([sys.executable, "-c", "pass"], cwd=tmp_path)
    assert code == 128 + signal.SIGTERM
    assert forwarded == [signal.SIGTERM]
    assert launcher._pending_signal == signal.SIGTERM
    assert launcher._forwarded_signal == signal.SIGTERM


def _write_fake_runpy(tmp_path: Path) -> None:
    (tmp_path / "run.py").write_text(
        "\n".join(
            [
                "import os, signal, sys, time",
                "def handle(signum, frame):",
                "    with open('sig_caught', 'w', encoding='utf-8') as fh:",
                "        fh.write(str(signum))",
                "    with open('env_dump.txt', 'w', encoding='utf-8') as fh:",
                "        for name in ('FAAR_MAX_RSS_GB','FAAR_MIN_GPU_FREE_GB','FAAR_MAX_GPU_MEMORY_FRACTION','OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES'):",
                "            fh.write(f'{name}={os.environ.get(name, \"UNSET\")}\\n')",
                "    sys.exit(0)",
                "signal.signal(signal.SIGTERM, handle)",
                "print('READY', flush=True)",
                "while True:",
                "    time.sleep(0.1)",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _wait_for_ready(out_file: Path, proc: subprocess.Popen, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if out_file.is_file() and "READY" in out_file.read_text(encoding="utf-8", errors="replace"):
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.05)
    return False


def test_launcher_forwards_sigterm_and_propagates_exit(tmp_path: Path) -> None:
    _write_fake_runpy(tmp_path)
    preflight_file = tmp_path / "preflight.json"
    preflight_file.write_text(
        __import__("json").dumps(
            _gpu_payload(devices=[{"index": 0, "total_memory_bytes": 16 * GIB, "free_memory_bytes": 16 * GIB}])
        ),
        encoding="utf-8",
    )
    (tmp_path / "hf").mkdir()
    env = os.environ.copy()
    env["FAAR_PREFLIGHT_JSON"] = str(preflight_file)
    env["HF_HOME"] = str(tmp_path / "hf")
    env["VLM_BACKEND"] = "openai"
    env["OPENAI_API_KEY"] = "sk-dummy-secret-never-printed"
    out = tmp_path / "out.log"
    err = tmp_path / "err.log"
    proc = subprocess.Popen(
        [sys.executable, str(CLUSTER / "launcher.py"), "--cpu-only", "--project-root", str(tmp_path), "arg-one"],
        cwd=str(tmp_path),
        env=env,
        stdout=open(out, "w", encoding="utf-8"),
        stderr=open(err, "w", encoding="utf-8"),
    )
    try:
        assert _wait_for_ready(out, proc), f"child never became ready; stderr={err.read_text()}"
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=20) == 128 + signal.SIGTERM
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    assert (tmp_path / "sig_caught").exists()
    assert (tmp_path / "sig_caught").read_text(encoding="utf-8") == str(signal.SIGTERM)

    env_dump = (tmp_path / "env_dump.txt").read_text(encoding="utf-8")
    assert "FAAR_MAX_RSS_GB=32.0" in env_dump
    assert "FAAR_MIN_GPU_FREE_GB=UNSET" in env_dump
    assert "FAAR_MAX_GPU_MEMORY_FRACTION=UNSET" in env_dump
    assert "OMP_NUM_THREADS=4" in env_dump
    assert "MKL_NUM_THREADS=4" in env_dump
    assert "CUDA_VISIBLE_DEVICES=UNSET" in env_dump

    logs = out.read_text(encoding="utf-8") + err.read_text(encoding="utf-8")
    assert "sk-dummy-secret-never-printed" not in logs
    assert "sk-dummy-secret-never-printed" not in env_dump


def test_launcher_rejects_multiple_gpus_end_to_end(tmp_path: Path) -> None:
    _write_fake_runpy(tmp_path)
    preflight_file = tmp_path / "preflight.json"
    preflight_file.write_text(
        __import__("json").dumps(
            _gpu_payload(
                devices=[
                    {"index": 0, "total_memory_bytes": 16 * GIB, "free_memory_bytes": 16 * GIB},
                    {"index": 1, "total_memory_bytes": 16 * GIB, "free_memory_bytes": 16 * GIB},
                ]
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "hf").mkdir()
    env = os.environ.copy()
    env["FAAR_PREFLIGHT_JSON"] = str(preflight_file)
    env["HF_HOME"] = str(tmp_path / "hf")
    env["OPENAI_API_KEY"] = "sk-dummy-secret"
    env.pop("CUDA_VISIBLE_DEVICES", None)
    result = subprocess.run(
        [sys.executable, str(CLUSTER / "launcher.py"), "--project-root", str(tmp_path), "--mode", "faar"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "exactly one logical GPU" in result.stderr
    assert "arg-one" not in result.stderr


def test_launcher_missing_key_fails_before_child(tmp_path: Path) -> None:
    _write_fake_runpy(tmp_path)
    preflight_file = tmp_path / "preflight.json"
    preflight_file.write_text(
        __import__("json").dumps(_gpu_payload(devices=[{"index": 0, "total_memory_bytes": 16 * GIB, "free_memory_bytes": 16 * GIB}])),
        encoding="utf-8",
    )
    (tmp_path / "hf").mkdir()
    env = os.environ.copy()
    env["FAAR_PREFLIGHT_JSON"] = str(preflight_file)
    env["HF_HOME"] = str(tmp_path / "hf")
    env.pop("OPENAI_API_KEY", None)
    env.pop("VLM_BACKEND", None)
    result = subprocess.run(
        [sys.executable, str(CLUSTER / "launcher.py"), "--project-root", str(tmp_path), "--mode", "faar"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "OPENAI_API_KEY" in result.stderr
    assert not (tmp_path / "sig_caught").exists()


def _write_fake_faar_demo(root: Path) -> Path:
    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    fake = venv_bin / "faar-demo"
    fake.write_text("#!/bin/sh\nfor arg in \"$@\"; do printf '%s\\n' \"$arg\"; done\n", encoding="utf-8")
    fake.chmod(0o755)
    return fake


def test_demo_script_path_resolution_portable(tmp_path: Path) -> None:
    _write_fake_faar_demo(tmp_path)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    demo = scripts_dir / "run_phase1_demo.sh"
    demo.write_text((Path(__file__).resolve().parents[1] / "scripts" / "run_phase1_demo.sh").read_text(encoding="utf-8"), encoding="utf-8")
    demo.chmod(0o755)
    result = subprocess.run(
        ["sh", str(demo), "446d159e", "--seed", "7"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    lines = result.stdout.splitlines()
    assert "run-example" in lines
    assert "--project-root" in lines
    assert str(tmp_path) in lines
    assert "--example-id" in lines
    assert "446d159e" in lines
    assert "--seed" in lines
    assert "7" in lines


def test_demo_script_usage_without_args(tmp_path: Path) -> None:
    _write_fake_faar_demo(tmp_path)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    demo = scripts_dir / "run_phase1_demo.sh"
    demo.write_text((Path(__file__).resolve().parents[1] / "scripts" / "run_phase1_demo.sh").read_text(encoding="utf-8"), encoding="utf-8")
    demo.chmod(0o755)
    result = subprocess.run(["sh", str(demo)], cwd=str(tmp_path), capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_launcher_syntax_compiles() -> None:
    import py_compile

    py_compile.compile(str(CLUSTER / "launcher.py"), doraise=True)
    py_compile.compile(str(CLUSTER / "preflight.py"), doraise=True)


@pytest.mark.parametrize(
    "script",
    [
        Path(__file__).resolve().parents[1] / "scripts" / "run_phase1_demo.sh",
        Path(__file__).resolve().parents[1] / "cluster" / "templates" / "slurm_one_gpu.sbatch",
        Path(__file__).resolve().parents[1] / "cluster" / "templates" / "pbs_one_gpu.pbs",
    ],
)
def test_shell_template_syntax(script: Path) -> None:
    if "run_phase1_demo.sh" in script.name:
        checker = ["sh", "-n"]
    else:
        checker = ["bash", "-n"]
    result = subprocess.run(checker + [str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_templates_never_override_cuda_visible_devices() -> None:
    for template in (Path(__file__).resolve().parents[1] / "cluster" / "templates" / "slurm_one_gpu.sbatch",
                     Path(__file__).resolve().parents[1] / "cluster" / "templates" / "pbs_one_gpu.pbs"):
        text = template.read_text(encoding="utf-8")
        assert "CUDA_VISIBLE_DEVICES=" not in text
        assert "--gpus=1" in text or "ngpus=1" in text
