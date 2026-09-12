"""Feature flags, read from gridcheck.toml at the repo root with environment overrides.

    [features]
    raspberry_connected = false      # in gridcheck.toml

    GRIDCHECK_RASPBERRY_CONNECTED=1  # environment variable, wins over the file

A flag that is off makes the corresponding code paths refuse to run (or silently use a
no-op stand-in) so that, for example, nothing ever touches GPIO on a laptop.
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.environ.get("GRIDCHECK_CONFIG", REPO_ROOT / "gridcheck.toml"))

DEFAULTS: dict[str, bool] = {
    "raspberry_connected": False,
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@lru_cache(maxsize=1)
def _file_features() -> dict[str, bool]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("rb") as f:
        data = tomllib.load(f)
    feats = data.get("features", {})
    return {k: bool(v) for k, v in feats.items()}


def env_var(name: str) -> str:
    return f"GRIDCHECK_{name.upper()}"


def feature(name: str) -> bool:
    """Current value of a feature flag: environment variable > gridcheck.toml > default."""
    if name not in DEFAULTS:
        raise KeyError(f"unknown feature flag {name!r}; known: {sorted(DEFAULTS)}")
    raw = os.environ.get(env_var(name))
    if raw is not None:
        v = raw.strip().lower()
        if v in _TRUE:
            return True
        if v in _FALSE:
            return False
    return _file_features().get(name, DEFAULTS[name])


def how_to_enable(name: str) -> str:
    """One-line instruction shown when a disabled feature is requested."""
    return (f"set `{name} = true` under [features] in {CONFIG_PATH.name} "
            f"or run with {env_var(name)}=1")
