# controller.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Sequence, Tuple

import jax
import jax.numpy as jnp

# ---------------- Registry for dynamic controller creation ----------------
CONTROLLER_REGISTRY = {}


def register_controller(name: str, cfg_key: str):
    """Decorator to register controller classes by name and config section key."""

    def _decorator(cls):
        CONTROLLER_REGISTRY[name] = (cls, cfg_key)
        return cls

    return _decorator


ControllerState = Tuple[Any, Any]  # (integral_err, prev_err) for both controllers
Params = Any


class Controller(ABC):
    """
    Abstract base class for controllers.

    Contract (functional / JAX-friendly):
      - get_params() -> params pytree
      - update_params(new_params) -> None
      - init_state() -> controller_state
      - compute_control_signal(params, state, error, dt) -> (u, new_state)

    Design note:
      compute_control_signal must not contain Python control-flow based on JAX values,
      and must avoid side effects (prints, exits) so it can be used under grad/scan/jit.
    """

    @abstractmethod
    def get_params(self) -> Params:
        raise NotImplementedError

    @abstractmethod
    def update_params(self, new_params: Params) -> None:
        raise NotImplementedError

    @abstractmethod
    def init_state(self) -> ControllerState:
        raise NotImplementedError

    @abstractmethod
    def compute_control_signal(
        self, ctrl_params: Params, ctrl_state: ControllerState, error, dt: float
    ):
        raise NotImplementedError


@register_controller("classic", "classic_controller")
class PIDController(Controller):
    """
    Classic PID controller with windup-guarded integral term.

    Params: [kp, ki, kd]
    State: (integral_err, prev_err)
    """

    def __init__(self, config: dict):
        self.kp = config["kp"]
        self.ki = config["ki"]
        self.kd = config["kd"]
        self.windup_guard = config["windup_guard"]

    def get_params(self):
        return jnp.array([self.kp, self.ki, self.kd], dtype=jnp.float32)

    def update_params(self, new_params):
        self.kp = float(new_params[0])
        self.ki = float(new_params[1])
        self.kd = float(new_params[2])

    def init_state(self):
        return (0.0, 0.0)

    def compute_control_signal(self, ctrl_params, ctrl_state, error, dt: float):
        kp, ki, kd = ctrl_params
        integral_err, prev_err = ctrl_state

        integral_err = integral_err + error * dt
        integral_err = clamp(integral_err, -self.windup_guard, self.windup_guard)

        derivative = (error - prev_err) / dt
        u = kp * error + ki * integral_err + kd * derivative
        return u, (integral_err, error)


@register_controller("nn", "nn_controller")
class NeuralNetworkController(Controller):
    """
    Neural-network "PID-like" controller.

    Input features: [error, integral(error), derivative(error)] (dimension 3)
    Output: a single scalar control signal (dimension 1)

    Architecture controlled by config["nn_architecture"], e.g. [3, 8, 8, 1].
    Activation functions are provided per layer (same length as number of layers),
    with a sensible fallback to "identity" if fewer are provided.
    """

    def __init__(self, config: dict):
        self.layer_sizes: List[int] = list(config["nn_architecture"])
        self.activation_functions: List[str] = list(config["activation_functions"])
        self.init_weight_range = tuple(config["init_weight_range"])
        self.seed = int(config["seed"])
        self.windup_guard = config["windup_guard"]

        self.activation_map = {
            "relu": jax.nn.relu,
            "tanh": jnp.tanh,
            "sigmoid": jax.nn.sigmoid,
            "identity": lambda x: x,
        }

        self._validate_config()
        self.params, self.act_fns = self._init_params()

    def _validate_config(self) -> None:
        """Validate configuration early so errors are explicit and user-friendly."""
        if len(self.layer_sizes) < 2:
            raise ValueError(
                f"nn_architecture must have at least 2 entries, got {self.layer_sizes}"
            )

        if self.layer_sizes[0] != 3:
            raise ValueError(
                f"nn_architecture must start with 3 (inputs: [error, integral, derivative]), got {self.layer_sizes}"
            )
        if self.layer_sizes[-1] != 1:
            raise ValueError(
                f"nn_architecture must end with 1 (single control output), got {self.layer_sizes}"
            )

        low, high = self.init_weight_range
        if not (
            isinstance(low, (int, float))
            and isinstance(high, (int, float))
            and low < high
        ):
            raise ValueError(
                f"init_weight_range must be (low, high) with low < high, got {self.init_weight_range}"
            )

        # Normalize activation names
        self.activation_functions = [str(a).lower() for a in self.activation_functions]
        for a in self.activation_functions:
            if a not in self.activation_map:
                raise ValueError(
                    f"Unknown activation '{a}'. Allowed: {sorted(self.activation_map.keys())}"
                )

    def _init_params(self):
        """
        Initialize MLP parameters.

        Returns:
          - params: list[(W, b)] as a JAX pytree
          - act_fns: list[callable] same length as number of layers
        """
        layer_params = []
        act_fns: List[Callable] = []

        low, high = self.init_weight_range
        key = jax.random.PRNGKey(self.seed)

        num_layers = len(self.layer_sizes) - 1
        for i in range(num_layers):
            in_dim = self.layer_sizes[i]
            out_dim = self.layer_sizes[i + 1]

            key, w_key, b_key = jax.random.split(key, 3)
            W = jax.random.uniform(w_key, (in_dim, out_dim), minval=low, maxval=high)
            b = jax.random.uniform(b_key, (out_dim,), minval=low, maxval=high)
            layer_params.append((W, b))

            if i < len(self.activation_functions):
                act_name = self.activation_functions[i]
            else:
                act_name = "identity"
            act_fns.append(self.activation_map[act_name])

        return layer_params, act_fns

    def get_params(self):
        return self.params

    def update_params(self, new_params):
        self.params = new_params

    def init_state(self):
        return (0.0, 0.0)

    def compute_control_signal(self, ctrl_params, ctrl_state, error, dt: float):
        """
        Compute control output u from the current error using an MLP over PID features.

        The controller state stores (integral_err, prev_err). From these it derives the
        feature vector [error, integral_err, derivative_err] and evaluates the MLP to
        produce a scalar control signal.

        Returns (u, new_state). Designed to be JAX-friendly and side-effect free.
        """
        integral_err, prev_err = ctrl_state

        new_int = integral_err + error * dt
        new_int = clamp(new_int, -self.windup_guard, self.windup_guard)

        deriv = (error - prev_err) / dt

        x = jnp.array([error, new_int, deriv], dtype=jnp.float32)
        x = jnp.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        for (W, b), act_fn in zip(ctrl_params, self.act_fns):
            x = x @ W + b
            x = jnp.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            x = act_fn(x)
            x = jnp.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        u = x[0]
        return u, (new_int, error)


def clamp(val, low, high):
    """Clip val into [low, high] using JAX operations."""
    return jnp.maximum(low, jnp.minimum(high, val))
