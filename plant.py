# plant.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple, Union, Dict

import jax.numpy as jnp

# ---------------- Registry for dynamic plant creation ----------------
PLANT_REGISTRY = {}


def register_plant(name: str):
    """Decorator to register a Plant subclass by name."""

    def _decorator(cls):
        PLANT_REGISTRY[name] = cls
        return cls

    return _decorator


PlantState = Union[float, Tuple[Any, ...], Dict[str, Any]]


class Plant(ABC):
    """
    Abstract base class for a "plant" (system dynamics).

    Contract (functional / JAX-friendly):
      - init_state() -> PlantState
      - get_output(state) -> scalar output Y (used to compute error)
      - update_state(state, control_signal, disturbance, dt) -> new PlantState

    Design note:
      All methods should be free of side effects so they can be used inside
      JAX-transformed code (grad/scan/jit).
    """

    # Target setpoint/target profit is provided by the plant configuration.
    target: float

    @abstractmethod
    def init_state(self) -> PlantState:
        """Return the initial plant state for a new epoch."""
        raise NotImplementedError

    @abstractmethod
    def get_output(self, plant_state: PlantState):
        """Return the current plant output Y given plant_state."""
        raise NotImplementedError

    @abstractmethod
    def update_state(
        self, plant_state: PlantState, control_signal, disturbance, dt: float
    ) -> PlantState:
        """Compute the next plant_state from current state, control, disturbance, and timestep dt."""
        raise NotImplementedError


@register_plant("bathtub")
class Bathtub(Plant):
    """
    Bathtub level control (discrete-time approximation).

    Dynamics:
      H_{t+1} = H_t + ( (U + D - Q(H_t)) / A ) * dt
      Q(H) = C * sqrt(2*g*H)

    Where:
      - H is water height
      - U is control inflow
      - D is disturbance inflow
      - Q is outflow
      - A is cross-sectional area
      - C is drain coefficient
    """

    def __init__(self, config: dict):
        self.area = config["area"]
        self.drain = config["drain"]
        self.init_height = config["init_height"]
        self.g = config["gravity"]
        self.target = config["target"]

    def init_state(self) -> float:
        return float(self.init_height)

    def get_output(self, plant_state: float):
        return plant_state

    def update_state(
        self, plant_state: float, control_signal_U, disturbance_D, dt: float
    ) -> float:
        velocity_V = jnp.sqrt(jnp.maximum(0.0, 2.0 * self.g * plant_state))
        outflow_Q = velocity_V * self.drain
        net_inflow = control_signal_U + disturbance_D
        dH_dt = net_inflow - outflow_Q
        return plant_state + (dH_dt / self.area) * dt


@register_plant("cournot")
class Cournot(Plant):
    """
    Cournot duopoly control.
    Controller U influences dq1/dt, disturbance D influences dq2/dt.

    State: (q1, q2), both clamped to [0, 1].
    Output Y: profit_1 = q1 * (p - c_m), where p = p_max - (q1 + q2).
    Error: target - output.
    """

    def __init__(self, config: dict):
        self.p_max = config["max_price"]
        self.c_m = config["marginal_cost"]
        self.init_q1 = config["q1"]
        self.init_q2 = config["q2"]
        self.target = config["target"]

    def init_state(self):
        return (float(self.init_q1), float(self.init_q2))

    def get_output(self, plant_state):
        q1, q2 = plant_state
        q_total = q1 + q2
        p = self.p_max - q_total
        profit_1 = q1 * (p - self.c_m)
        return profit_1

    def update_state(self, plant_state, control_signal, disturbance, dt: float):
        q1, q2 = plant_state
        q1_next = clamp(q1 + control_signal * dt, 0.0, 1.0)
        q2_next = clamp(q2 + disturbance * dt, 0.0, 1.0)
        return (q1_next, q2_next)


@register_plant("predator_prey")
class PredatorPrey(Plant):
    """
    Simple predator-prey (Lotka–Volterra-style) with control and disturbance.

    State: (x, y)
      x = prey population
      y = predator population

    Controller U adds to prey growth, disturbance D adds to predator dynamics.
    Output Y: prey population x (setpoint tracking).
    """

    def __init__(self, config: dict):
        self.init_x = config["init_prey"]
        self.init_y = config["init_predator"]
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.delta = config["delta"]
        self.gamma = config["gamma"]
        self.target = config["target"]

    def init_state(self):
        return (float(self.init_x), float(self.init_y))

    def get_output(self, plant_state):
        x, _y = plant_state
        return x

    def update_state(self, plant_state, control_signal, disturbance, dt: float):
        x, y = plant_state
        x_next = jnp.maximum(
            0.0, x + dt * (self.alpha * x - self.beta * x * y + control_signal)
        )
        y_next = jnp.maximum(
            0.0, y + dt * (self.delta * x * y - self.gamma * y + disturbance)
        )
        return (x_next, y_next)


def clamp(val, low, high):
    """Clip val into [low, high] using JAX operations."""
    return jnp.maximum(low, jnp.minimum(high, val))
