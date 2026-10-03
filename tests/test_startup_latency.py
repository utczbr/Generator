"""
tests/test_startup_latency.py

Cold startup compilation latency benchmark (T207).
Launches an isolated Python subprocess to measure wall-clock compilation time
of synth.semantics.loader.registry from a completely cold process state.
Asserts that elapsed time is strictly <= 50.0 ms and verifies zero top-level pydantic imports.
"""
import subprocess
import sys
import pytest


def test_cold_startup_latency():
    cmd = [
        sys.executable,
        "-c",
        (
            "import time; "
            "t0 = time.perf_counter(); "
            "from synth.semantics.loader import registry; "
            "t1 = time.perf_counter(); "
            "import sys; "
            "has_pydantic = 'pydantic' in sys.modules; "
            "elapsed_ms = (t1 - t0) * 1000.0; "
            "print(f'ELAPSED:{elapsed_ms:.4f}'); "
            "print(f'METRICS:{len(registry.metrics_by_label)}'); "
            "print(f'HAS_PYDANTIC:{has_pydantic}')"
        ),
    ]

    # Take best of 3 runs to guard against transient OS scheduling / IO spikes
    samples = []
    for _ in range(3):
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        stdout = result.stdout
        lines = stdout.strip().splitlines()
        data = {}
        for line in lines:
            if ":" in line:
                key, val = line.split(":", 1)
                data[key.strip()] = val.strip()

        assert "ELAPSED" in data, f"Benchmark did not output ELAPSED. Stdout: {stdout}"
        assert "METRICS" in data, f"Benchmark did not output METRICS. Stdout: {stdout}"
        assert "HAS_PYDANTIC" in data, f"Benchmark did not output HAS_PYDANTIC. Stdout: {stdout}"

        elapsed_ms = float(data["ELAPSED"])
        metrics_count = int(data["METRICS"])
        has_pydantic = data["HAS_PYDANTIC"] == "True"
        samples.append((elapsed_ms, metrics_count, has_pydantic))

    best_ms = min(s[0] for s in samples)
    metrics_count = samples[0][1]
    has_pydantic = any(s[2] for s in samples)

    # Strict performance and isolation assertions
    assert not has_pydantic, "pydantic was imported during cold startup of loader.py"
    assert metrics_count >= 400, f"Expected >= 400 metrics in registry, found {metrics_count}"
    assert best_ms <= 50.0, (
        f"Cold startup latency exceeded 50.0 ms budget: measured best of {len(samples)} was {best_ms:.2f} ms"
    )
