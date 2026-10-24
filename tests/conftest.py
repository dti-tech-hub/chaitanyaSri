"""Shared fixtures for the test suite."""
from __future__ import annotations

import pytest

from src.control.signal_controller import FixedTimeController, SignalController
from src.traffic.intersection import Intersection
from src.utils.config import load_config


@pytest.fixture()
def config():
    return load_config("config/config.yaml")


@pytest.fixture()
def intersection(config):
    return Intersection(config)


@pytest.fixture()
def controller(config, intersection):
    return SignalController(config, intersection)


@pytest.fixture()
def baseline(config, intersection):
    return FixedTimeController(config, intersection)
