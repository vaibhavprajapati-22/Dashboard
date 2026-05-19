"""
config_loader.py
----------------
Loads and validates the YAML configuration file for the Trajectory Dashboard.
"""

from pathlib import Path
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from a YAML file.

    Returns
    -------
    dict with keys:
        - reference : {"lat_ref": float, "lon_ref": float}
        - paths     : {path_name: {"gt_file": str, "kf_variants": {name: filepath}}}
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    _validate(cfg)
    return cfg


def get_path_names(cfg: dict) -> list[str]:
    return list(cfg["paths"].keys())


def get_kf_names(cfg: dict, path_name: str) -> list[str]:
    return list(cfg["paths"][path_name]["kf_variants"].keys())


def get_strategies(cfg: dict, path_name: str, kf_name: str) -> list[str]:
    return list(cfg["paths"][path_name]["kf_variants"][kf_name]["strategies"].keys())


def get_gt_filepath(cfg: dict, path_name: str) -> str:
    return cfg["paths"][path_name]["gt_file"]


def get_path_velocity(cfg: dict, path_name: str) -> float:
    return float(cfg["paths"][path_name].get("velocity", 0.0))


def get_kf_node_filepaths(cfg: dict, path_name: str, kf_name: str, strategy: str) -> dict[str, str]:
    return cfg["paths"][path_name]["kf_variants"][kf_name]["strategies"][strategy].get("nodes", {})


def get_kf_combination_filepath(cfg: dict, path_name: str, kf_name: str, strategy: str) -> str | None:
    return cfg["paths"][path_name]["kf_variants"][kf_name]["strategies"][strategy].get("combination")


def get_reference(cfg: dict) -> tuple[float, float]:
    ref = cfg["reference"]
    return float(ref["lat_ref"]), float(ref["lon_ref"])


# ── Internal ──────────────────────────────────────────────────────────────────

def _validate(cfg: dict) -> None:
    assert "reference" in cfg, "Config missing 'reference' block."
    assert "lat_ref" in cfg["reference"] and "lon_ref" in cfg["reference"], \
        "Reference block must contain 'lat_ref' and 'lon_ref'."
    assert "paths" in cfg and cfg["paths"], "Config missing 'paths' block."
    for pname, pdata in cfg["paths"].items():
        assert "gt_file" in pdata, f"Path '{pname}' missing 'gt_file'."
        assert "kf_variants" in pdata and pdata["kf_variants"], \
            f"Path '{pname}' missing 'kf_variants'."
        for kfname, kfdata in pdata["kf_variants"].items():
            assert "strategies" in kfdata and kfdata["strategies"], \
                f"KF '{kfname}' missing 'strategies'."
            for sname, sdata in kfdata["strategies"].items():
                assert "nodes" in sdata and sdata["nodes"], \
                    f"Strategy '{sname}' missing 'nodes'."
