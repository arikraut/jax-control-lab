# utils.py
from __future__ import annotations

import yaml

from plant import PLANT_REGISTRY
from controller import CONTROLLER_REGISTRY


def load_config(path: str) -> dict:
    """Load YAML config into a Python dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_plant_from_config(cfg: dict):
    """
    Instantiate a plant based on cfg["plant_type"].
    """
    plant_type = cfg["plant_type"]
    plant_cfg = cfg[plant_type]

    try:
        plant_cls = PLANT_REGISTRY[plant_type]
    except KeyError as e:
        raise ValueError(
            f"Unknown plant_type='{plant_type}'. Available: {sorted(PLANT_REGISTRY.keys())}"
        ) from e

    return plant_cls(plant_cfg)


def create_controller_from_config(cfg: dict):
    """
    Instantiate a controller based on cfg["controller"]["type"].

    Controllers are registered with:
      @register_controller("name", "config_section_key")
    """
    ctrl_type = str(cfg["controller"]["type"]).lower()

    try:
        ctrl_cls, cfg_key = CONTROLLER_REGISTRY[ctrl_type]
    except KeyError as e:
        raise ValueError(
            f"Unknown controller type='{ctrl_type}'. Available: {sorted(CONTROLLER_REGISTRY.keys())}"
        ) from e

    return ctrl_cls(cfg[cfg_key])
