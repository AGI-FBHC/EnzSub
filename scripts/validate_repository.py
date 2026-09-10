#!/usr/bin/env python3
"""Run release-oriented static checks without loading models or data."""

import ast
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCAN_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".md", ".toml"}
PRIVATE_PATTERNS = {
    "home directory": re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/"),
    "private server address": re.compile(r"\b172\.18\.144\.9\b"),
    "embedded password": re.compile(
        r"password\s*[:=]\s*['\"]?[^\s'\"]+", re.IGNORECASE
    ),
}
REQUIRED_FILES = (
    "README.md",
    "environment.yml",
    "pyproject.toml",
    "data/manifest.yaml",
    "docs/reproduction.md",
    "configs/cpt/config_esm2_650M_random_cpt.yaml",
    "configs/cpt/config_esm2_3B_random_cpt.yaml",
    "configs/cpt/config_protbert_random_cpt.yaml",
    "configs/downstream/esp/esm2_650M.yaml",
    "configs/downstream/active_site/esm2_650M.yaml",
    "configs/analysis/hcft/esm2_650M_general.yaml",
    "configs/analysis/substrate_coherence/esm2_3B.yaml",
    "configs/analysis/rns/esm2_650M_cpt.yaml",
)

def tracked_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") and part not in {".github"} for part in path.relative_to(ROOT).parts):
            continue
        yield path

def check_required(errors):
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            errors.append("missing required file: {}".format(relative))

def check_python(errors):
    for path in tracked_files():
        if path.suffix != ".py":
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8", errors="strict"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append("invalid Python {}: {}".format(path.relative_to(ROOT), exc))

def check_yaml(errors):
    for path in tracked_files():
        if path.suffix not in {".yaml", ".yml"}:
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                yaml.safe_load(handle)
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            errors.append("invalid YAML {}: {}".format(path.relative_to(ROOT), exc))

def check_shell(errors):
    for path in tracked_files():
        if path.suffix != ".sh":
            continue
        result = subprocess.run(
            ["bash", "-n", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode:
            errors.append("invalid shell {}: {}".format(
                path.relative_to(ROOT), result.stderr.strip()
            ))

def check_private_paths(errors):
    for path in tracked_files():
        if path.suffix not in SCAN_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(text):
                errors.append("{} in {}".format(label, path.relative_to(ROOT)))

def check_generated_files(errors):
    if (ROOT / "paper-results").exists():
        errors.append("unexpected root output directory: paper-results")
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            errors.append("generated Python cache: {}".format(relative))

def check_readme_collisions(errors):
    for directory in (path for path in ROOT.rglob("*") if path.is_dir()):
        names = [path.name for path in directory.iterdir() if path.is_file()]
        readmes = [name for name in names if name.lower() == "readme.md"]
        if len(readmes) > 1:
            errors.append("duplicate README names in {}: {}".format(
                directory.relative_to(ROOT), ", ".join(sorted(readmes))
            ))

def main():
    errors = []
    check_required(errors)
    check_python(errors)
    check_yaml(errors)
    check_shell(errors)
    check_private_paths(errors)
    check_generated_files(errors)
    check_readme_collisions(errors)

    if errors:
        print("Repository validation failed:")
        for error in errors:
            print("- {}".format(error))
        return 1

    print("Repository validation passed.")
    print("This confirms static release structure only; it does not run training or inference.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
