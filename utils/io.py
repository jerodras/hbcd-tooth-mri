"""
utils/io.py
-----------
Shared I/O helpers: config loading and common path resolution.
"""

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str) -> dict:
    """Load and return a YAML config file as a nested dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_config_parser(description: str = "") -> argparse.ArgumentParser:
    """Return an ArgumentParser with a standard --config argument."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml relative to CWD)",
    )
    return parser


def resolve_path(base: Path, rel: str) -> Path:
    """Resolve a path relative to a base directory."""
    return (base / rel).resolve()
