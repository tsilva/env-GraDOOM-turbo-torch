#!/usr/bin/env python3
"""Build and audit the one-time metadata-only PyPI migration package."""

from __future__ import annotations

import argparse
import email
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REDIRECT_ROOT = ROOT / "redirect"


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def source_contract(version: str) -> tuple[str, str]:
    configuration = tomllib.loads((REDIRECT_ROOT / "pyproject.toml").read_text())
    project = configuration["project"]
    old_name = str(project["name"])
    if str(project["version"]) != version:
        raise ValueError("redirect version does not match the release version")
    dependencies = list(project.get("dependencies", ()))
    if len(dependencies) != 1 or "==" not in dependencies[0]:
        raise ValueError("redirect must have one exact replacement dependency")
    new_name, dependency_version = dependencies[0].rsplit("==", 1)
    if dependency_version != version or normalized(new_name) == normalized(old_name):
        raise ValueError("redirect dependency must target the renamed distribution")
    if project.get("optional-dependencies"):
        raise ValueError("redirect must not declare optional dependencies")
    if configuration.get("tool", {}).get("setuptools", {}).get("packages") != []:
        raise ValueError("redirect must declare an empty package set")
    return old_name, new_name


def audit_dist(dist: Path, version: str) -> None:
    old_name, new_name = source_contract(version)
    all_files = sorted(path for path in dist.iterdir() if path.is_file())
    files = [
        path
        for path in all_files
        if path.suffix == ".whl" or path.name.endswith(".tar.gz")
    ]
    unexpected = [
        path.name
        for path in all_files
        if path not in files and path.name != ".gitignore"
    ]
    if unexpected:
        raise ValueError(f"redirect build contains unexpected files: {unexpected}")
    wheels = [path for path in files if path.suffix == ".whl"]
    sdists = [path for path in files if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1 or len(files) != 2:
        raise ValueError("redirect build must contain exactly one wheel and one sdist")

    with zipfile.ZipFile(wheels[0]) as archive:
        members = archive.namelist()
        if not members or any(".dist-info/" not in member for member in members):
            raise ValueError("redirect wheel contains importable files")
        metadata_name = next(
            member for member in members if member.endswith(".dist-info/METADATA")
        )
        metadata = email.message_from_bytes(archive.read(metadata_name))
        if normalized(str(metadata["Name"])) != normalized(old_name):
            raise ValueError("redirect wheel has the wrong project name")
        if metadata["Version"] != version:
            raise ValueError("redirect wheel has the wrong version")
        requirements = metadata.get_all("Requires-Dist", [])
        if not any(
            normalized(re.split(r"[\s\[<>=!~;]", requirement, maxsplit=1)[0])
            == normalized(new_name)
            and f"=={version}" in requirement.replace(" ", "")
            for requirement in requirements
        ):
            raise ValueError("redirect wheel lacks the exact replacement dependency")
        if any(member.endswith(".dist-info/entry_points.txt") for member in members):
            raise ValueError("redirect wheel must not expose entry points")

    with tarfile.open(sdists[0], "r:gz") as archive:
        unsafe = [
            member.name
            for member in archive.getmembers()
            if member.isfile()
            and Path(member.name).suffix.lower()
            in {".py", ".pyc", ".pyd", ".so", ".dylib"}
        ]
        if unsafe:
            raise ValueError(f"redirect sdist contains importable files: {unsafe}")


def build(dist: Path, version: str) -> None:
    source_contract(version)
    if dist.exists() and any(dist.iterdir()):
        raise ValueError(f"refusing to reuse non-empty output directory: {dist}")
    dist.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pypi-redirect-") as temp:
        build_root = Path(temp)
        shutil.copy2(REDIRECT_ROOT / "pyproject.toml", build_root / "pyproject.toml")
        shutil.copy2(REDIRECT_ROOT / "README.md", build_root / "README.md")
        subprocess.run(
            [
                "uv",
                "build",
                "--no-sources",
                "--config-file",
                str(ROOT / "uv.toml"),
                "--project",
                str(build_root),
                "--out-dir",
                str(dist),
            ],
            cwd=ROOT,
            check=True,
        )
    audit_dist(dist, version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check-source")
    check.add_argument("--version", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--version", required=True)
    build_parser.add_argument("--out-dir", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--version", required=True)
    audit.add_argument("--dist", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check-source":
        source_contract(args.version)
    elif args.command == "build":
        build(args.out_dir, args.version)
    else:
        audit_dist(args.dist, args.version)


if __name__ == "__main__":
    main()
