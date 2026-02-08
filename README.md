# CONSYS — Differentiable Control Systems in JAX (PID + Neural PID)

This project implements a small, extensible control-systems framework in Python and JAX.
It supports multiple plants (system dynamics) and multiple controllers (classic PID and a neural “PID-like” controller), trained end-to-end via gradient descent using JAX automatic differentiation.

The main goal is modularity: new plants and controllers can be added without modifying the orchestration logic.

## Features

Plants

- Bathtub: nonlinear drain dynamics (outflow proportional to square root of height)
- Cournot duopoly: controller influences firm 1 output, disturbance influences firm 2
- Predator–prey: Lotka–Volterra style dynamics with control and disturbance terms

Controllers

- Classic PID: trainable kp, ki, kd with integral anti-windup
- Neural PID: an MLP over error, integral(error), derivative(error)

Differentiable training

- Per-epoch rollout implemented with JAX lax.scan
- Mean squared error loss computed over an epoch
- Gradients computed via JAX value_and_grad

Configuration-driven

- All major hyperparameters live in config.yaml

Plots

- MSE vs epoch for every run
- PID gain trajectories (kp, ki, kd) for classic PID runs

---

## Repository structure

```text
.
├── config.yaml          # Main configuration (plant/controller/sim params)
├── run.py               # Entry point (train + plot)
├── consys.py            # Orchestrator (training loop, rollout, logging)
├── plant.py             # Plant interface + plant implementations + registry
├── controller.py        # Controller interface + controllers + registry
├── results              # Plots from training
└── utils.py             # Config loading + registry-based factories

```
