"""Compatibility facade for the central cached-fit DMD/Prony forecaster.

The public names remain local for existing pipeline imports. The direct-script
smoke at the bottom deliberately avoids importing ``hy3dgen.shapegen`` so it is
safe to run before optional pipeline dependencies are installed.
"""

try:
    from hicache_pp.dmd import (
        dmd_eval,
        dmd_fit,
        dmd_forecast,
        dmd_forecast_state,
        dmd_update_snapshots,
    )
except ImportError as exc:  # pragma: no cover - installation failure path
    raise ImportError(
        "Hunyuan HiCache++ requires hicache-pp>=1.2.1; install requirements.txt"
    ) from exc


__all__ = [
    "dmd_eval", "dmd_fit", "dmd_forecast", "dmd_forecast_state",
    "dmd_update_snapshots",
]


if __name__ == "__main__":
    import torch

    # A compact direct-execution check: scalar DMD stays finite and the stateful
    # path reuses one fit across multiple skip horizons.
    snapshots = [torch.tensor([0.8 ** step, 0.6 ** step]) for step in range(4)]
    state = {
        "step": 5,
        "history": 5,
        "dmd_snapshots": [(step, value) for step, value in enumerate(snapshots)],
    }
    state["derivatives"] = {0: snapshots[-1]}
    state["activated_steps"] = [3]
    prediction = dmd_forecast_state(state)
    assert torch.isfinite(prediction).all()
    state["step"] = 6
    assert torch.isfinite(dmd_forecast_state(state)).all()
    assert state.get("_dmd_fit_key") == (3, 4, 1)
    print("hunyuan2-plus-plus central DMD smoke passed")
