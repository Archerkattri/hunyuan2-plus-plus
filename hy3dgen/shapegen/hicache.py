"""Compatibility facade for the shared scalar HiCache state and Hermite arm.

The pipeline keeps its native CFG-combined update site, while the model-agnostic
state, schedule, Hermite forecast, reset, and telemetry are maintained centrally
by ``hicache-pp``. The DMD arm is selected explicitly with ``backend='dmd'`` and
implemented by :mod:`hicache_dmd`.
"""

try:
    from hicache_pp.hermite import (
        hicache_decide,
        hicache_forecast,
        hicache_init,
        hicache_reset,
        hicache_telemetry,
        hicache_update_derivatives,
        physicists_hermite,
        scaled_hermite,
    )
except ImportError as exc:  # pragma: no cover - installation failure path
    raise ImportError(
        "Hunyuan HiCache requires hicache-pp>=1.2.1; install requirements.txt"
    ) from exc


__all__ = [
    "hicache_decide", "hicache_forecast", "hicache_init", "hicache_reset",
    "hicache_telemetry", "hicache_update_derivatives", "physicists_hermite",
    "scaled_hermite",
]
