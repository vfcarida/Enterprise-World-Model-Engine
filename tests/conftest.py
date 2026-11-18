"""Pytest shared test fixtures and configuration for EWM Engine."""

from __future__ import annotations

import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, Phase, Verbosity, settings
from hypothesis.database import DirectoryBasedExampleDatabase

from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.seed import seed_everything
from ewm_engine.core.state import WorldState

# ---------------------------------------------------------------------------
# Hypothesis Settings Profiles & Anti-Flake Database Setup
# ---------------------------------------------------------------------------
_db = DirectoryBasedExampleDatabase(".hypothesis/examples")

# CI Profile: fast, targeted, replaying any recorded failures from database
settings.register_profile(
    "ci",
    max_examples=100,
    deadline=timedelta(milliseconds=2000),
    suppress_health_check=[HealthCheck.too_slow],
    database=_db,
)

# Nightly Profile: deep search, high examples, targeted fuzzing phase enabled
settings.register_profile(
    "nightly",
    max_examples=1000,
    deadline=None,
    phases=[
        Phase.explicit,
        Phase.reuse,
        Phase.generate,
        Phase.target,
        Phase.shrink,
    ],
    verbosity=Verbosity.normal,
    database=_db,
)

# Dev Profile: rapid local developer iteration
settings.register_profile(
    "dev",
    max_examples=25,
    deadline=timedelta(milliseconds=1000),
    suppress_health_check=[HealthCheck.too_slow],
    database=_db,
)

# Load profile from environment variable (defaults to 'ci' in automated runs)
_active_profile = os.getenv("HYPOTHESIS_PROFILE", "ci")
settings.load_profile(_active_profile)

DEFAULT_TEST_SEED = 42


@pytest.fixture(autouse=True)
def seed_deterministic_test_environment() -> None:
    """Fix global and standard library random seeds before every test for deterministic replay."""
    random.seed(DEFAULT_TEST_SEED)
    np.random.seed(DEFAULT_TEST_SEED)
    seed_everything(DEFAULT_TEST_SEED, deterministic=True, warn_cuda=False)


@pytest.fixture
def isolated_temp_dir(tmp_path: Path) -> Path:
    """Provide an isolated temporary directory for hermetic test file I/O."""
    io_dir = tmp_path / "isolated_io"
    io_dir.mkdir(parents=True, exist_ok=True)
    return io_dir


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator fixture."""
    return np.random.default_rng(DEFAULT_TEST_SEED)


@pytest.fixture
def sample_world_state() -> WorldState:
    """Fixture providing a standard two-warehouse world state."""
    return WorldState(
        entities=[
            Entity(id="wh_1", type="warehouse", attributes={"region": "north"}),
            Entity(id="wh_2", type="warehouse", attributes={"region": "south"}),
        ],
        relationships=[
            Relationship(
                source="wh_1", target="wh_2", type="connected_to", attributes={"distance_km": 50.0}
            ),
        ],
        resources=[
            Resource(
                id="stock_wh1",
                entity_id="wh_1",
                current=100.0,
                min_value=0.0,
                max_value=200.0,
                unit="boxes",
            ),
            Resource(
                id="stock_wh2",
                entity_id="wh_2",
                current=50.0,
                min_value=0.0,
                max_value=200.0,
                unit="boxes",
            ),
        ],
        memory={"prior_shipments": 0},
        active_rules={"safety_stock": 20.0},
        context={"temperature_c": 22.5},
        timestamp=0.0,
        step=0,
    )
