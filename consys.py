# consys.py
from __future__ import annotations

import logging
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from utils import load_config, create_plant_from_config, create_controller_from_config
from controller import PIDController


logger = logging.getLogger(__name__)


class CONSYS:
    """
    Control system orchestrator.

    Responsibilities:
      - Load config
      - Build plant + controller
      - Run differentiable rollouts
      - Train controller parameters via gradient descent (JAX autodiff)
    """

    def __init__(self, config_path: str):
        self.cfg = load_config(config_path)

        self.plant = create_plant_from_config(self.cfg)
        self.controller = create_controller_from_config(self.cfg)

        sim_cfg = self.cfg["simulation"]
        self.num_epochs = int(sim_cfg["num_epochs"])
        self.timesteps = int(sim_cfg["timesteps_per_epoch"])
        self.disturbance_range = tuple(sim_cfg["disturbance_range"])
        self.alpha = float(sim_cfg["learning_rate"])
        self.dt = float(sim_cfg["dt"])

        # For plotting PID gains over epochs (only used for classic PID)
        self.pid_k_history: Tuple[List[float], List[float], List[float]] = ([], [], [])

    def run_one_epoch(self, ctrl_params, disturbances) -> jnp.ndarray:
        """
        Differentiable epoch rollout.

        Args:
          ctrl_params: controller parameters pytree
          disturbances: array-like of length T (one disturbance per timestep)

        Returns:
          mse: scalar MSE over the T errors in this epoch
        """
        plant_state0 = self.plant.init_state()
        ctrl_state0 = self.controller.init_state()
        disturbances = jnp.asarray(disturbances, dtype=jnp.float32)

        def step(carry, disturbance_t):
            plant_state, ctrl_state = carry

            y = self.plant.get_output(plant_state)
            error = self.plant.target - y

            u, new_ctrl_state = self.controller.compute_control_signal(
                ctrl_params, ctrl_state, error, self.dt
            )

            new_plant_state = self.plant.update_state(
                plant_state, u, disturbance_t, self.dt
            )
            return (new_plant_state, new_ctrl_state), error

        (_, _), errors = jax.lax.scan(step, (plant_state0, ctrl_state0), disturbances)
        mse = jnp.mean(errors**2)
        return mse

    def train_one_epoch(self, epoch: int) -> jnp.ndarray:
        """
        One training epoch:
          1) sample disturbances
          2) compute mse + grads
          3) fail-fast if mse is non-finite
          4) update controller params

        Returns:
          mse
        """
        disturbances = np.random.uniform(
            self.disturbance_range[0], self.disturbance_range[1], size=self.timesteps
        ).astype(np.float32)

        ctrl_params = self.controller.get_params()

        def loss_fn(params):
            return self.run_one_epoch(params, disturbances)

        mse, grads = jax.value_and_grad(loss_fn)(ctrl_params)

        if not np.isfinite(float(mse)):
            raise RuntimeError(f"MSE became non-finite at epoch {epoch}: {mse}")

        new_params = jax.tree_util.tree_map(
            lambda p, g: p - self.alpha * g, ctrl_params, grads
        )
        self.controller.update_params(new_params)
        return mse

    def train_for_many_epochs(self, print_every: int = 10):
        """
        Train for num_epochs and return MSE history.
        Also stores PID gain history if using PIDController.
        """
        mse_history: List[float] = []
        kp_hist, ki_hist, kd_hist = [], [], []

        for epoch in range(self.num_epochs):
            mse = self.train_one_epoch(epoch)
            mse_f = float(mse)
            mse_history.append(mse_f)

            # Track PID gains if applicable
            if isinstance(self.controller, PIDController):
                kp, ki, kd = self.controller.get_params()
                kp_hist.append(float(kp))
                ki_hist.append(float(ki))
                kd_hist.append(float(kd))

            if print_every and (
                epoch % print_every == 0 or epoch == self.num_epochs - 1
            ):
                logger.info("Epoch %d/%d | MSE=%.6f", epoch + 1, self.num_epochs, mse_f)

        self.pid_k_history = (kp_hist, ki_hist, kd_hist)
        return mse_history
