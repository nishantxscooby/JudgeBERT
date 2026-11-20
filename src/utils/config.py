"""Configuration loading and global seeding.

Every script in this project obtains its settings through :func:`load_config`,
so that a run is fully described by (config file + code revision + seed).
Nothing downstream of this module should read environment variables or
hard-code paths.

Typical use
-----------
>>> from src.utils.config import load_config
>>> cfg = load_config()
>>> cfg["training"]["batch_size"]
8
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml

# Repository root = two levels up from this file (src/utils/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"

# Cached parsed config, keyed by resolved path, so repeated loads are free.
_CONFIG_CACHE: dict[Path, dict[str, Any]] = {}


class ConfigError(RuntimeError):
    """Raised when the configuration is missing or internally inconsistent."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the YAML config, apply ``overrides``, and validate it.

    Parameters
    ----------
    path
        Path to a YAML config.  Defaults to ``configs/config.yaml``.
    overrides
        Nested mapping merged on top of the file contents.  Used by tests and
        by experiment scripts that need a one-off tweak without writing a new
        file.  Lists replace rather than concatenate.

    Returns
    -------
    dict
        A deep copy, so callers may mutate it freely.
    """
    config_path = Path(path).resolve() if path else DEFAULT_CONFIG_PATH

    if config_path not in _CONFIG_CACHE:
        if not config_path.exists():
            raise ConfigError(
                f"Config file not found: {config_path}\n"
                f"Expected the repository default at {DEFAULT_CONFIG_PATH}."
            )
        with config_path.open("r", encoding="utf-8") as fh:
            parsed = yaml.safe_load(fh)
        if not isinstance(parsed, dict):
            raise ConfigError(f"Config root must be a mapping, got {type(parsed)}")
        _CONFIG_CACHE[config_path] = parsed

    cfg = copy.deepcopy(_CONFIG_CACHE[config_path])
    if overrides:
        cfg = deep_merge(cfg, dict(overrides))

    _resolve_paths(cfg)
    validate_config(cfg)
    return cfg


def deep_merge(base: MutableMapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base``; lists are replaced wholesale."""
    for key, value in overlay.items():
        if key in base and isinstance(base[key], MutableMapping) and isinstance(value, Mapping):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return dict(base)


def _resolve_paths(cfg: dict[str, Any]) -> None:
    """Attach an absolute ``_abs`` sibling to every entry in ``paths``."""
    paths = cfg.get("paths", {})
    for key, value in list(paths.items()):
        if key == "root" or key.endswith("_abs"):
            continue
        paths[f"{key}_abs"] = str((REPO_ROOT / str(value)).resolve())
    paths["root_abs"] = str(REPO_ROOT)


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Fail loudly on configs that would silently produce nonsense.

    These checks are cheap and catch the mistakes that actually happen:
    split fractions that do not sum to 1, a score range that is inverted,
    an unknown annotation status, or a generalisation domain that is not in
    the domain list.
    """
    problems: list[str] = []

    # --- project -----------------------------------------------------------
    if "project" not in cfg:
        problems.append("missing top-level 'project' section")

    # --- dataset -----------------------------------------------------------
    ds = cfg.get("dataset")
    if not ds:
        problems.append("missing top-level 'dataset' section")
    else:
        score_bounds_valid = True
        for field in ("score_min", "score_max"):
            value = ds.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                problems.append(
                    f"dataset.{field} must be a number, not a boolean"
                )
                score_bounds_valid = False
        if score_bounds_valid and ds["score_min"] >= ds["score_max"]:
            problems.append(
                f"dataset.score_min ({ds.get('score_min')}) must be < "
                f"dataset.score_max ({ds.get('score_max')})"
            )
        langs = ds.get("languages") or []
        if not langs:
            problems.append("dataset.languages must not be empty")

        statuses = set(ds.get("annotation_status_values") or [])
        gold = set(ds.get("gold_status_values") or [])
        nongold = set(ds.get("non_gold_status_values") or [])
        if gold & nongold:
            problems.append(
                f"a status cannot be both gold and non-gold: {sorted(gold & nongold)}"
            )
        if not gold <= statuses:
            problems.append(
                f"gold_status_values not in annotation_status_values: "
                f"{sorted(gold - statuses)}"
            )
        if not nongold <= statuses:
            problems.append(
                f"non_gold_status_values not in annotation_status_values: "
                f"{sorted(nongold - statuses)}"
            )

        domains = set(ds.get("domains") or [])
        gen = ds.get("generalization_domain")
        if gen is not None and gen not in domains:
            problems.append(
                f"dataset.generalization_domain '{gen}' is not in dataset.domains"
            )
        prim = ds.get("primary_domain")
        if prim is not None and prim not in domains:
            problems.append(f"dataset.primary_domain '{prim}' is not in dataset.domains")
        if gen is not None and gen == prim:
            problems.append(
                "dataset.generalization_domain must differ from primary_domain "
                "or Experiment 3 is not a held-out-domain test"
            )

    # --- decisions ---------------------------------------------------------
    dec = cfg.get("decision") or {}
    lo = dec.get("partial_min")
    hi = dec.get("preserved_min")
    if lo is not None and hi is not None and lo >= hi:
        problems.append(
            f"decision.partial_min ({lo}) must be < decision.preserved_min ({hi})"
        )

    # --- splits ------------------------------------------------------------
    sp = cfg.get("splits") or {}
    fracs = [sp.get("train"), sp.get("validation"), sp.get("test")]
    if all(f is not None for f in fracs):
        total = sum(float(f) for f in fracs)
        if abs(total - 1.0) > 1e-6:
            problems.append(f"split fractions sum to {total}, expected 1.0")
        if any(float(f) < 0 for f in fracs):
            problems.append("split fractions must be non-negative")

    # --- models ------------------------------------------------------------
    models = cfg.get("models") or {}
    jb = models.get("judgebert") or {}
    if jb and jb.get("target_scaling") not in {None, "minmax", "none"}:
        problems.append(
            f"models.judgebert.target_scaling must be 'minmax' or 'none', "
            f"got {jb.get('target_scaling')!r}"
        )
    if jb and jb.get("loss") not in {None, "smooth_l1", "mse"}:
        problems.append(
            f"models.judgebert.loss must be 'smooth_l1' or 'mse', got {jb.get('loss')!r}"
        )
    if jb and jb.get("pooling") not in {None, "cls", "mean"}:
        problems.append(
            f"models.judgebert.pooling must be 'cls' or 'mean', got {jb.get('pooling')!r}"
        )

    # --- error taxonomy ----------------------------------------------------
    tax = (cfg.get("error_taxonomy") or {}).get("labels") or []
    if tax and "None" not in tax:
        problems.append("error_taxonomy.labels should contain the 'None' class")

    if problems:
        raise ConfigError(
            "Invalid configuration:\n  - " + "\n  - ".join(problems)
        )


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------
def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG we might touch, and optionally force deterministic kernels.

    Called once at the top of every entry point.  ``PYTHONHASHSEED`` cannot be
    set from inside a running interpreter, so it is exported for child
    processes and documented in the README.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # cuDNN autotuning is non-deterministic; disable it when asked.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Best-effort determinism for CPU scatter/gather kernels. Wrapped
            # because the flag name has moved across torch versions.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:  # torch is optional for the data-only scripts
        pass


def config_fingerprint(cfg: Mapping[str, Any] | None = None) -> str:
    """Return a short stable hash of the config.

    Stamped into every results file so a table can always be traced back to
    the exact settings that produced it.
    """
    cfg = cfg if cfg is not None else load_config()
    payload = json.dumps(_strip_abs_paths(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _strip_abs_paths(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Drop machine-specific absolute paths before hashing.

    Otherwise the fingerprint would differ between two machines running the
    identical experiment, which would defeat the point.
    """
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        if isinstance(value, Mapping):
            out[key] = _strip_abs_paths(value)
        elif key.endswith("_abs"):
            continue
        else:
            out[key] = value
    return out


def resolve_path(cfg: Mapping[str, Any], key: str, create: bool = False) -> Path:
    """Resolve a key from the ``paths`` section to an absolute :class:`Path`.

    ``resolve_path(cfg, "processed_dir")`` -> <repo>/data/processed
    """
    paths = cfg.get("paths") or {}
    abs_key = f"{key}_abs"
    if abs_key in paths:
        path = Path(paths[abs_key])
    elif key in paths:
        path = (REPO_ROOT / str(paths[key])).resolve()
    else:
        raise ConfigError(
            f"Unknown path key {key!r}. Known keys: "
            f"{sorted(k for k in paths if not k.endswith('_abs'))}"
        )
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def dump_config(cfg: Mapping[str, Any], destination: str | os.PathLike[str]) -> Path:
    """Write the *effective* config to disk (provenance for a run)."""
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(_strip_abs_paths(cfg), fh, sort_keys=False, allow_unicode=True)
    return dest
