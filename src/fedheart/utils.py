"""Shared helpers: configuration, deterministic seeding, file IO and environment capture."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
import zlib
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

# Packages whose versions are reported in every results bundle.
KEY_PACKAGES = ("flwr", "torch", "scikit-learn", "pandas", "numpy", "scipy", "ray", "matplotlib")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def results_dir(cfg: dict[str, Any]) -> Path:
    return resolve(cfg["paths"]["results_dir"])


def derive_seed(*parts: int | str | float) -> int:
    """Deterministic 31-bit seed from a tuple of keys.

    Used so every random choice (split, initial weights, each epoch's shuffle order at each
    hospital) has its own reproducible stream that does not depend on execution order or on
    which process runs it (Flower clients run in separate Ray workers).
    """
    entropy = [p if isinstance(p, int) and p >= 0 else zlib.crc32(str(p).encode()) for p in parts]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0] & 0x7FFFFFFF)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=_json_default)
        f.write("\n")


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


def lr_tag(lr: float) -> str:
    """Filesystem-safe label for a learning rate, e.g. 0.001 -> 'lr1e-03'."""
    return f"lr{lr:.0e}"


def seed_tag(seed: int) -> str:
    return f"seed_{seed:02d}"


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def environment_info() -> dict[str, Any]:
    """Python, platform and package versions for the results bundle."""
    installed = {d.metadata["Name"].lower(): d.version for d in metadata.distributions()}
    return {
        "captured_at": now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_num_threads": torch.get_num_threads(),
        "key_packages": {p: installed.get(p) for p in KEY_PACKAGES},
        "all_packages": dict(sorted(installed.items())),
    }
