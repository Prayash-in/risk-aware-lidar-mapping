"""Prediction: resolution bubbles and constant-velocity trajectory rollouts.

Two distinct products, both consumed by the risk controller:

* **Resolution bubble** (§9) — a disc around a tracked object that must stay
  finely resolved:  r_bubble = r_min(class) + kappa · |v| · T_horizon
  with r_min = 2 m (pedestrian), 3 m (cyclist), 5 m (vehicle), kappa = 1.5,
  T_horizon = 2 s. A faster object gets a larger bubble, so resolution is spent
  where the object can reach next.
* **Predicted corridor** (§10) — the rollout of the current velocity over
  `pred_horizon`, sampled at `pred_samples` points. Resolution is pre-allocated
  along it.

Both are *estimates from a constant-velocity model*; nothing here is ground
truth, and the docs say so wherever the numbers are reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Minimum bubble radius per class (m) — §9.
DEFAULT_R_MIN: dict[str, float] = {
    "pedestrian": 2.0,
    "cyclist": 3.0,
    "vehicle": 5.0,
    "unknown": 2.0,
}


@dataclass
class PredictionConfig:
    """All prediction parameters (§9, §10)."""

    r_min: dict = field(default_factory=lambda: dict(DEFAULT_R_MIN))
    kappa: float = 1.5
    horizon: float = 2.0
    pred_horizon: float = 2.0
    pred_samples: int = 8

    def bubble_radius(self, label: str, speed: float) -> float:
        """r_bubble = r_min(label) + kappa · |v| · T_horizon."""
        base = float(self.r_min.get(label, self.r_min.get("unknown", 2.0)))
        return base + float(self.kappa) * float(speed) * float(self.horizon)

    def corridor_radius(self, bubble_radius: float) -> float:
        """Corridor half-width used for the prediction importance factor."""
        return float(min(4.0, max(1.0, 0.4 * float(bubble_radius))))

    def to_dict(self) -> dict:
        return {
            "r_min": dict(self.r_min), "kappa": self.kappa, "horizon": self.horizon,
            "pred_horizon": self.pred_horizon, "pred_samples": self.pred_samples,
        }


def constant_velocity_path(x: float, y: float, vx: float, vy: float,
                            horizon: float = 2.0, samples: int = 8) -> list[list[float]]:
    """Sample the constant-velocity rollout pos(t) = pos(0) + v·t.

    Returns [] when prediction is disabled (horizon <= 0) or samples < 1.
    """
    if horizon <= 0 or samples < 1:
        return []
    dt = horizon / samples
    return [[round(x + vx * dt * k, 2), round(y + vy * dt * k, 2)]
            for k in range(1, samples + 1)]


def bubble_radius(label: str, speed: float, config: PredictionConfig | None = None) -> float:
    """Convenience wrapper around `PredictionConfig.bubble_radius`."""
    return (config or PredictionConfig()).bubble_radius(label, speed)


def path_endpoints(path: list[list[float]]) -> tuple[tuple[float, float] | None,
                                                      tuple[float, float] | None]:
    """First and last sample of a rollout (for tests and visualization)."""
    if not path:
        return None, None
    return (float(path[0][0]), float(path[0][1])), (float(path[-1][0]), float(path[-1][1]))


def corridor_length(path: list[list[float]]) -> float:
    """Polyline length of a rollout (m)."""
    return float(sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1)))
