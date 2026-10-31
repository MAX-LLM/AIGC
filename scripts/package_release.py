#!/usr/bin/env python3
"""Build a source-only GitHub upload ZIP without local state or Git history."""

import argparse
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {".gitignore", "README.md"}
REFERENCE_FILES = (
    "knowledge_base/mnd_knowledge.json",
    "rubrics/rubric_v1.txt",
    "research/config/critical_errors.json",
    "research/config/deterministic_screen.json",
    "research/config/development_experiment_registry.json",
    "research/config/split_policy.json",
    "research/config/study_registry.json",
)
DIRECTORY_SUFFIXES = {
    "docs": {".md"},
    "scripts": {".py"},
    ".github/workflows": {".yml", ".yaml"},
    "knowledge_base": {".md"},
    "evaluation": {".py"},
    "experiments": {".py"},
    "tests": {".py"},
}


def source_files():
    for path in sorted(ROOT.iterdir()):
        if path.is_file() and not path.is_symlink() and (
            path.name in ROOT_FILES or path.suffix in {".py", ".sh"}
        ):
            yield path
    for relative in REFERENCE_FILES:
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing source reference file: {relative}")
        yield path
    for directory, suffixes in DIRECTORY_SUFFIXES.items():
        # Source modules are flat. Never descend into generated run/result
        # directories, even if an experiment writes Python or Markdown files.
        for path in sorted((ROOT / directory).glob("*")):
            relative = path.relative_to(ROOT / directory)
            if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                continue
            if path.name in {"summary.md", "report.md"}:
                continue
            if path.is_file() and not any(parent.is_symlink() for parent in [path, *path.parents] if parent != ROOT) and path.suffix in suffixes:
                yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime" / "release" / "MND_agent_github_ready.zip")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error("The output already exists; choose a new --output path.")
    files = list(source_files())
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(Path("MND_agent") / path.relative_to(ROOT)))
    print(f"Created {output} ({len(files)} source and reference files)")


if __name__ == "__main__":
    main()
