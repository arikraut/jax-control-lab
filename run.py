# run.py
from __future__ import annotations
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import logging

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from consys import CONSYS
from controller import PIDController


def main():
    """
    Entry point:
      - constructs CONSYS from config.yaml
      - trains controller
      - plots training curves
    """
    logging.basicConfig(level=logging.INFO)

    system = CONSYS(config_path="./config.yaml")
    mse_history = system.train_for_many_epochs(print_every=10)

    controller_type = system.cfg["controller"]["type"]
    plant_type = system.cfg["plant_type"]

    # Plot MSE
    plt.figure()
    plt.plot(mse_history, label="MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"Training Progress (Controller: {controller_type}, Plant: {plant_type})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot PID gains if classic controller
    if isinstance(system.controller, PIDController):
        kp_hist, ki_hist, kd_hist = system.pid_k_history
        plt.figure()
        plt.plot(kp_hist, label="kp")
        plt.plot(ki_hist, label="ki")
        plt.plot(kd_hist, label="kd")
        plt.xlabel("Epoch")
        plt.ylabel("Gain value")
        plt.title("PID Gains over Epochs")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
