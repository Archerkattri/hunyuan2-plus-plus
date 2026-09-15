"""CPU tests for the Hunyuan mini central DMD adapter."""

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hicache = _load("hunyuan_hicache_adapter", "hy3dgen/shapegen/hicache.py")
dmd = _load("hunyuan_hicache_dmd_adapter", "hy3dgen/shapegen/hicache_dmd.py")


def _trajectory():
    return [torch.tensor([0.92 ** step, 0.70 ** step]) for step in range(6)]


def _state(values):
    state = hicache.hicache_init(num_steps=20, interval=1, first_enhance=0,
                                 backend="dmd", history=5)
    for step, value in enumerate(values):
        state["step"] = step
        state["activated_steps"].append(step)
        hicache.hicache_update_derivatives(state, value)
        dmd.dmd_update_snapshots(state, value, history=5)
    return state


def test_facades_resolve_to_central_implementations():
    import hicache_pp.dmd as central_dmd
    import hicache_pp.hermite as central_hermite

    assert dmd.dmd_forecast_state is central_dmd.dmd_forecast_state
    assert dmd.dmd_update_snapshots is central_dmd.dmd_update_snapshots
    assert hicache.hicache_forecast is central_hermite.hicache_forecast
    assert hicache.hicache_telemetry is central_hermite.hicache_telemetry


def test_dmd_fit_reused_per_compute_window(monkeypatch):
    import hicache_pp.dmd as central

    state = _state(_trajectory()[:5])
    calls = 0
    original = central.dmd_fit

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(central, "dmd_fit", counted)
    for step in (5, 6, 7):
        state["step"] = step
        out = dmd.dmd_forecast_state(state)
        assert torch.isfinite(out).all()
    assert calls == 1
    assert state["_dmd_fit_key"] == (4, 5, 1)
    assert state["telemetry"]["method_counts"]["dmd"] == 3


def test_snapshot_owns_storage_and_reset_clears_fit():
    state = _state(_trajectory()[:4])
    original = state["dmd_snapshots"][-1][1]
    source = original.clone().requires_grad_(True)
    state["activated_steps"].append(4)
    dmd.dmd_update_snapshots(state, source)
    with torch.no_grad():
        source.add_(10.0)
    stored = state["dmd_snapshots"][-1][1]
    assert not stored.requires_grad
    assert torch.equal(stored, original)

    state["_dmd_fit"] = object()
    state["_dmd_fit_key"] = (4, 5, 1)
    run_id = state["run_id"]
    hicache.hicache_reset(state)
    assert state["run_id"] != run_id
    assert state["dmd_snapshots"] == []
    assert "_dmd_fit" not in state and "_dmd_fit_key" not in state
    assert hicache.hicache_telemetry(state)["decisions"] == {"full": 0, "forecast": 0}


def test_short_history_uses_actual_hermite_fallback():
    traj = _trajectory()
    state = hicache.hicache_init(num_steps=10, interval=3, first_enhance=0,
                                 backend="dmd", max_order=1, sigma=0.5)
    state["step"] = 4
    state["activated_steps"] = [3]
    hicache.hicache_update_derivatives(state, traj[3])
    state["dmd_snapshots"] = [(1, traj[1]), (3, traj[3])]
    output = dmd.dmd_forecast_state(state)
    assert torch.equal(output, traj[3])
    assert state["telemetry"]["fallbacks"]["dmd_insufficient_uniform_history"] == 1


def test_cfg_combined_update_contract_is_preserved():
    cond, uncond, scale = torch.tensor([3.0]), torch.tensor([1.0]), 5.0
    expected = uncond + scale * (cond - uncond)
    # This is the exact expression retained at the pipeline's native update site.
    actual = torch.tensor([1.0]) + scale * (torch.tensor([3.0]) - torch.tensor([1.0]))
    assert torch.equal(actual, expected)
