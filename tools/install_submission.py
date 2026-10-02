"""Copy this reviewed submission into a clean Git checkout without pushing anything."""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from audit_submission import issues_for

SOURCE = Path(__file__).resolve().parents[1]
MERGE_FILES = {"README.md", ".gitignore", "requirements.txt"}
MARKER = "MAML_REVIEW_SUBMISSION_20261002"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, encoding="utf-8").strip()


def merge_text(name, original, incoming):
    if MARKER in original:
        raise ValueError(f"{name}: submission marker already exists; do not install twice.")
    if name == "README.md":
        section = re.sub(r"(?m)^(#{1,5}) ", r"\1# ", incoming)
        return original.rstrip() + f"\n\n<!-- {MARKER} -->\n\n" + section.rstrip() + "\n"
    if name == ".gitignore":
        return original.rstrip() + f"\n\n# {MARKER}\n" + incoming.rstrip() + "\n"
    # Do not silently alter a teammate's dependency constraints.
    def requirements(text):
        values = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-") or " @ " in line:
                raise ValueError("Existing requirements use options/includes/URLs; merge requirements.txt manually.")
            m = re.match(r"^([A-Za-z0-9_.-]+)", line)
            if not m:
                raise ValueError("Unsupported requirements.txt entry; manual merge required.")
            key = re.sub(r"[-_.]+", "-", m.group(1)).lower()
            values.setdefault(key, set()).add(re.sub(r"\s+", "", line).lower())
        return values
    old, new = requirements(original), requirements(incoming)
    for name in old.keys() & new.keys():
        if old[name] != new[name]:
            raise ValueError(f"requirements.txt: different constraints for {name}; resolve before copying.")
    present = {line.strip() for line in original.splitlines()}
    extra = [line for line in incoming.splitlines() if line.strip() not in present]
    return original.rstrip() + f"\n\n# {MARKER}\n" + "\n".join(extra).rstrip() + "\n"


def install(repo):
    repo = repo.resolve()
    if repo == SOURCE or repo in SOURCE.parents or SOURCE in repo.parents:
        raise ValueError("Keep the unpacked submission and target repository in separate folders.")
    if Path(git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        raise ValueError("--repo must be the Git repository root.")
    branch = git(repo, "branch", "--show-current")
    if not branch or branch in {"main", "master"}:
        raise ValueError("Switch to your original personal branch before installing.")
    if git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Target checkout has local changes. Use the fresh clone described in the guide.")
    for module in ("meta_maml_exploration", "water_label_missing_ready"):
        if (repo / module).exists():
            raise ValueError(f"{module} already exists on the reset base. Check the PR base/history before proceeding.")

    # Check every destination before making the first change.
    planned = []
    for source in sorted(SOURCE.rglob("*")):
        rel = source.relative_to(SOURCE)
        if any(part in {".git", "__pycache__"} for part in rel.parts):
            continue
        if source.is_symlink():
            raise ValueError(f"Symlink in submission: {rel}")
        if not source.is_file():
            continue
        name = rel.as_posix()
        content = source.read_bytes()
        problems = issues_for(name, content)
        if problems:
            raise ValueError(f"Submission contains forbidden content: {name}: {problems}")
        target = repo / rel
        for parent in [target, *target.parents]:
            if parent == repo:
                break
            if parent.is_symlink():
                raise ValueError(f"Symlink at destination: {name}")
            if parent != target and parent.exists() and not parent.is_dir():
                raise ValueError(f"Destination parent is a file: {name}")
        if target.exists():
            if not target.is_file():
                raise ValueError(f"Destination is not a regular file: {name}")
            previous = target.read_bytes()
            if previous == content:
                continue
            if name not in MERGE_FILES:
                raise ValueError(f"Different existing teammate file: {name}. Resolve it before installation.")
            content = merge_text(name, previous.decode("utf-8-sig"), content.decode("utf-8-sig")).encode("utf-8")
        planned.append((target, content))

    for target, content in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    print(f"Copied/merged {len(planned)} files into branch {branch}.")
    print("No commits, resets or pushes were performed by this copying tool.")
    print("Next: git add ., then python tools/audit_submission.py --git-index")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    try:
        install(args.repo)
    except (ValueError, OSError, UnicodeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"STOP: {exc}\n")
