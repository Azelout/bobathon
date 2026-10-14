#!/usr/bin/env python3
"""Zip a team's code for hand-in.

Keeps the work they wrote — experiments, journals, source, tests, and the
EDA files in ``data/`` — and leaves out lab materials, secrets, raw inputs,
and generated stores.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories that are machine-local or generated, at any depth.
SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        ".bob",
        ".pi",
        ".agents",
        ".cursor",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".vscode",
        ".idea",
        ".worktrees",
        "mlruns",
        "mlartifacts",
        "node_modules",
    }
)

# Top-level trees that are lab material or large generated output.
# ``data/`` is not here: the skills keep the EDA deliverables in that
# folder and only the raw inputs are left out.
SKIP_TOP_LEVEL = frozenset(
    {
        "docs",
        "submissions",
        "reports",
        "html",
        "_build",
        "scratch",
        "skills",
    }
)

# Raw inputs under ``data/``. Matches the workspace gitignore: never drop
# the folder itself, because ``data/eda.py``, ``data/eda.md``, and
# ``data/eda_*.html`` are part of the work.
RAW_DATA_SUFFIXES = frozenset(
    {".csv", ".tsv", ".h5", ".hdf5", ".parquet", ".feather", ".npy", ".npz"}
)

SKIP_RELATIVE = frozenset(
    {
        Path("tests/test_setup.py"),
        Path("scripts/skore-agent"),
        Path("scripts/bundle.py"),
    }
)

# Files shipped with the lab. A zip that contains only these is not a hand-in.
STARTER_FILES = frozenset(
    {
        Path("README.md"),
        Path("requirements.txt"),
        Path("pytest.ini"),
        Path(".gitignore"),
    }
)


def _gitignore_rules(root: Path) -> list[tuple[bool, str, bool]]:
    """Return ``(negated, pattern, directory_only)`` rules from ``.gitignore``."""
    try:
        lines = (root / ".gitignore").read_text().splitlines()
    except OSError:
        return []
    rules: list[tuple[bool, str, bool]] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        if negated:
            line = line[1:].strip()
        directory_only = line.endswith("/")
        if directory_only:
            line = line[:-1]
        if line:
            rules.append((negated, line, directory_only))
    return rules


def _segments_match(path: str, pattern: str) -> bool:
    """Return whether ``path`` matches a slash-containing gitignore glob."""
    path_parts = path.split("/")
    pattern_parts = pattern.split("/")
    if len(path_parts) != len(pattern_parts):
        return False
    return all(
        fnmatch.fnmatchcase(part, token)
        for part, token in zip(path_parts, pattern_parts, strict=True)
    )


def _matches_gitignore(
    relative: Path, pattern: str, directory_only: bool, *, is_dir: bool
) -> bool:
    """Return whether one gitignore rule matches ``relative``."""
    if directory_only and not is_dir:
        return False
    anchored = pattern.startswith("/")
    if anchored:
        pattern = pattern[1:]
    path = relative.as_posix()
    if "/" in pattern:
        return _segments_match(path, pattern)
    if anchored:
        return "/" not in path and fnmatch.fnmatchcase(path, pattern)
    return any(fnmatch.fnmatchcase(part, pattern) for part in relative.parts)


def _gitignored(
    relative: Path, rules: list[tuple[bool, str, bool]], *, is_dir: bool
) -> bool:
    """Return whether ``.gitignore`` leaves ``relative`` out. Last match wins."""
    ignored = False
    for negated, pattern, directory_only in rules:
        if _matches_gitignore(relative, pattern, directory_only, is_dir=is_dir):
            ignored = not negated
    return ignored


def _raw_data(relative: Path) -> bool:
    """Return whether ``relative`` is a raw input under ``data/``."""
    if not relative.parts or relative.parts[0] != "data":
        return False
    if "raw" in relative.parts:
        return True
    return relative.suffix.lower() in RAW_DATA_SUFFIXES


def _under_skipped_path(relative: Path) -> bool:
    """Return whether ``relative`` is a lab path or a file inside one."""
    return any(
        relative == skipped or skipped in relative.parents for skipped in SKIP_RELATIVE
    )


def _skip_dir(relative: Path) -> bool:
    """Return whether a directory should be left out of the zip."""
    if _under_skipped_path(relative):
        return True
    if relative.name.endswith(".egg-info") or relative.name == ".pixi":
        return True
    if len(relative.parts) == 1 and relative.name in {"build", "dist"}:
        return True
    if _raw_data(relative):
        return True
    if relative.name == ".skore" or relative.name.startswith(".env"):
        return True
    if any(part in SKIP_DIR_NAMES for part in relative.parts):
        return True
    return bool(relative.parts) and relative.parts[0] in SKIP_TOP_LEVEL


def _skip_file(relative: Path) -> bool:
    """Return whether a file should be left out of the zip."""
    if _under_skipped_path(relative):
        return True
    if relative.name in {".DS_Store", ".skore"} or relative.name.startswith(".env"):
        return True
    if relative.suffix.lower() in {".pyc", ".ipynb", ".db"} or relative.name.endswith(
        ".db-journal"
    ):
        return True
    if _raw_data(relative):
        return True
    return relative.suffix.lower() == ".csv" and relative.name.lower().startswith(
        "submission"
    )


def collect_files(root: Path, destination: Path) -> list[Path]:
    """Return the files to zip, in archive order.

    Parameters
    ----------
    root : Path
        Participant repository root.
    destination : Path
        Zip path. Skipped when it sits inside ``root``.
    """
    rules = _gitignore_rules(root)
    chosen: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = [
            name
            for name in dirnames
            if not _skip_dir(relative := (current / name).relative_to(root))
            and not _gitignored(relative, rules, is_dir=True)
        ]
        for name in filenames:
            path = current / name
            if path.is_symlink() or path.resolve() == destination:
                continue
            relative = path.relative_to(root)
            if _skip_file(relative) or _gitignored(relative, rules, is_dir=False):
                continue
            chosen.append(path)
    chosen.sort(key=lambda path: path.relative_to(root).as_posix())
    return chosen


def write_bundle(root: Path, destination: Path) -> list[str]:
    """Write the hand-in zip and return its member names.

    Parameters
    ----------
    root : Path
        Participant repository root.
    destination : Path
        Where to write the zip.

    Returns
    -------
    list of str
        Archive names, each prefixed with the repository directory name.
    """
    root = root.resolve()
    destination = destination.resolve()
    files = collect_files(root, destination)
    participant_code = [
        path for path in files if path.relative_to(root) not in STARTER_FILES
    ]
    if not participant_code:
        raise SystemExit(
            "no participant code to bundle. Add experiments, journals, "
            "or source files first."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            arcname = f"{root.name}/{path.relative_to(root).as_posix()}"
            archive.write(path, arcname)
            names.append(arcname)
    return names


def main(argv: list[str] | None = None) -> None:
    """Zip this repository, or the repository passed with ``--root``."""
    parser = argparse.ArgumentParser(
        description=(
            "Zip participant code, leaving out lab data, secrets, and generated files."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository to bundle (default: this repo)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="zip path (default: <root>/<root name>.zip)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = (args.output or (root / f"{root.name}.zip")).resolve()
    names = write_bundle(root, output)
    print(f"wrote {output} ({len(names)} files)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
