#!/usr/bin/env python3
"""Build and audit GraDOOM release distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.error
import urllib.request
import zipfile
from email.parser import BytesParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_NAME = "gradoom"
IMPORT_NAME = "gradoom"
VERSION_PATTERN = re.compile(
    r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)"
    r"(?:(?P<pre>a|b|rc)(?P<pre_number>[0-9]+)"
    r"|\.post(?P<post_number>[0-9]+)"
    r"|\.dev(?P<dev_number>[0-9]+))?$"
)
VERSION_FILES = (
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "src" / IMPORT_NAME / "__init__.py",
    REPO_ROOT / "uv.lock",
)


def read_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def project_metadata() -> tuple[str, str]:
    project = read_toml(REPO_ROOT / "pyproject.toml").get("project")
    if not isinstance(project, dict):
        raise SystemExit("pyproject.toml is missing [project]")
    name = project.get("name")
    version = project.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise SystemExit("project name and version must be strings")
    return name, version


def init_version() -> str:
    path = REPO_ROOT / "src" / IMPORT_NAME / "__init__.py"
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']\s*$',
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match is None:
        raise SystemExit(f"could not find __version__ in {path}")
    return match.group(1)


def lock_version() -> str:
    packages = read_toml(REPO_ROOT / "uv.lock").get("package", [])
    if not isinstance(packages, list):
        raise SystemExit("uv.lock has an invalid package table")
    matches = [
        package.get("version")
        for package in packages
        if isinstance(package, dict) and package.get("name") == PACKAGE_NAME
    ]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise SystemExit(f"expected one {PACKAGE_NAME!r} package in uv.lock")
    return matches[0]


def validate_version(version: str) -> None:
    if VERSION_PATTERN.fullmatch(version) is None:
        raise SystemExit(f"unsupported PEP 440 release version: {version!r}")


def next_version(version: str) -> str:
    match = VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise SystemExit(f"unsupported PEP 440 release version: {version!r}")
    base = ".".join(match.group(name) for name in ("major", "minor", "patch"))
    if pre := match.group("pre"):
        return f"{base}{pre}{int(match.group('pre_number')) + 1}"
    if dev := match.group("dev_number"):
        return f"{base}.dev{int(dev) + 1}"
    return (
        f"{match.group('major')}.{match.group('minor')}."
        f"{int(match.group('patch')) + 1}"
    )


def check_version(args: argparse.Namespace) -> None:
    project_name, project_version = project_metadata()
    expected = args.version or project_version
    validate_version(expected)
    actual = {
        "project.name": project_name,
        "pyproject.toml": project_version,
        "src/gradoom/__init__.py": init_version(),
        "uv.lock": lock_version(),
    }
    wanted = {
        "project.name": PACKAGE_NAME,
        "pyproject.toml": expected,
        "src/gradoom/__init__.py": expected,
        "uv.lock": expected,
    }
    failures = {
        key: value for key, value in actual.items() if value != wanted[key]
    }
    if failures:
        details = ", ".join(
            f"{key}={value!r}, expected {wanted[key]!r}"
            for key, value in failures.items()
        )
        raise SystemExit(f"release metadata mismatch for {expected}: {details}")
    print(json.dumps({"package": PACKAGE_NAME, "version": expected}, indent=2))


def fetch_pypi() -> dict[str, object]:
    url = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {}
        raise
    if not isinstance(data, dict):
        raise SystemExit("unexpected PyPI JSON response")
    return data


def pypi_releases() -> dict[str, object]:
    releases = fetch_pypi().get("releases", {})
    if not isinstance(releases, dict):
        raise SystemExit("unexpected PyPI releases payload")
    return releases


def tagged_versions() -> set[str]:
    result = subprocess.run(
        ["git", "tag", "--list", "v*"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        tag.removeprefix("v")
        for tag in result.stdout.splitlines()
        if tag.startswith("v")
    }


def select_release_version(
    current: str,
    releases: dict[str, object],
    tags: set[str],
) -> str:
    validate_version(current)
    candidate = current
    while releases.get(candidate) or candidate in tags:
        candidate = next_version(candidate)
    return candidate


def replace_project_version(path: Path, current: str, target: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf'(?ms)(^\[project\]\n.*?^version\s*=\s*"){re.escape(current)}(")'
    )
    updated, count = pattern.subn(rf"\g<1>{target}\g<2>", text, count=1)
    if count != 1:
        raise SystemExit(f"could not update [project] version in {path}")
    path.write_text(updated, encoding="utf-8")


def replace_init_version(path: Path, current: str, target: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf'(?m)^(?P<prefix>__version__\s*=\s*["\']){re.escape(current)}(?P<suffix>["\'])$'
    )
    updated, count = pattern.subn(
        rf"\g<prefix>{target}\g<suffix>",
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit(f"could not update __version__ in {path}")
    path.write_text(updated, encoding="utf-8")


def replace_lock_version(path: Path, current: str, target: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf'(?m)(^\[\[package\]\]\nname = "{PACKAGE_NAME}"\nversion = ")'
        rf'{re.escape(current)}(")'
    )
    updated, count = pattern.subn(rf"\g<1>{target}\g<2>", text, count=1)
    if count != 1:
        raise SystemExit(f"could not update {PACKAGE_NAME!r} version in {path}")
    path.write_text(updated, encoding="utf-8")


def write_version(target: str) -> None:
    validate_version(target)
    _, current = project_metadata()
    check_version(argparse.Namespace(version=current))
    snapshots = {path: path.read_bytes() for path in VERSION_FILES}
    try:
        replace_project_version(VERSION_FILES[0], current, target)
        replace_init_version(VERSION_FILES[1], current, target)
        replace_lock_version(VERSION_FILES[2], current, target)
        check_version(argparse.Namespace(version=target))
    except BaseException:
        for path, contents in snapshots.items():
            path.write_bytes(contents)
        raise


def prepare_version(args: argparse.Namespace) -> None:
    _, current = project_metadata()
    check_version(argparse.Namespace(version=current))
    releases = pypi_releases()
    tags = tagged_versions()
    if args.to:
        validate_version(args.to)
        target = args.to
        if releases.get(target):
            raise SystemExit(f"{PACKAGE_NAME}=={target} already exists on PyPI")
        if target in tags:
            raise SystemExit(f"release tag already exists: v{target}")
    else:
        target = select_release_version(current, releases, tags)
    if args.write and target != current:
        write_version(target)
    print(
        json.dumps(
            {
                "package": PACKAGE_NAME,
                "current_version": current,
                "selected_version": target,
                "bumped": target != current,
                "written": bool(args.write and target != current),
            },
            indent=2,
        )
    )


def check_pypi(args: argparse.Namespace) -> None:
    validate_version(args.version)
    releases = pypi_releases()
    if releases.get(args.version):
        raise SystemExit(f"{PACKAGE_NAME}=={args.version} already exists on PyPI")
    print(f"{PACKAGE_NAME}=={args.version} is unused on PyPI")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wheel_audit(wheel: Path, version: str) -> dict[str, object]:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
        metadata = (
            BytesParser().parsebytes(archive.read(metadata_names[0]))
            if len(metadata_names) == 1
            else None
        )
        wheel_metadata = (
            BytesParser().parsebytes(archive.read(wheel_names[0]))
            if len(wheel_names) == 1
            else None
        )
    checks = {
        "expected_filename": wheel.name == f"{PACKAGE_NAME}-{version}-py3-none-any.whl",
        "one_metadata_file": len(metadata_names) == 1,
        "one_wheel_file": len(wheel_names) == 1,
        "metadata_name": metadata is not None and metadata.get("Name") == PACKAGE_NAME,
        "metadata_version": metadata is not None and metadata.get("Version") == version,
        "universal_python_wheel": (
            wheel_metadata is not None and wheel_metadata.get("Tag") == "py3-none-any"
        ),
        "has_init": f"{IMPORT_NAME}/__init__.py" in names,
        "has_environment": f"{IMPORT_NAME}/env.py" in names,
        "has_scenario": f"{IMPORT_NAME}/scenario.py" in names,
        "has_package_data": f"{IMPORT_NAME}/assets/zdoom_bullet_chips.json" in names,
        "has_license_files": sum(".dist-info/licenses/" in name for name in names) >= 3,
        "no_cache_files": not any(
            "__pycache__" in Path(name).parts or name.endswith(".pyc") for name in names
        ),
    }
    result = {"wheel": str(wheel), "checks": checks}
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(json.dumps(result, indent=2), file=sys.stderr)
        raise SystemExit(f"wheel audit failed: {failed}")
    return result


def sdist_audit(sdist: Path, version: str) -> dict[str, object]:
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()
    root = f"{PACKAGE_NAME}-{version}"
    checks = {
        "expected_filename": sdist.name == f"{root}.tar.gz",
        "has_pyproject": f"{root}/pyproject.toml" in names,
        "has_readme": f"{root}/README.md" in names,
        "has_license": f"{root}/LICENSE" in names,
        "has_gpl_license": f"{root}/LICENSES/GPL-3.0-only.txt" in names,
        "has_notices": f"{root}/THIRD_PARTY_NOTICES.md" in names,
        "has_package": f"{root}/src/{IMPORT_NAME}/__init__.py" in names,
        "has_package_data": (
            f"{root}/src/{IMPORT_NAME}/assets/zdoom_bullet_chips.json" in names
        ),
        "no_build_outputs": not any(
            part in {".git", ".venv", "__pycache__", "build", "dist"}
            for name in names
            for part in Path(name).parts
        ),
    }
    result = {"sdist": str(sdist), "checks": checks}
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(json.dumps(result, indent=2), file=sys.stderr)
        raise SystemExit(f"sdist audit failed: {failed}")
    return result


def smoke_wheel(wheel: Path, version: str) -> None:
    code = """
import sys
from importlib.metadata import PathDistribution
from pathlib import Path

wheel = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(wheel))
import gradoom

assert gradoom.__version__ == sys.argv[2]
assert len(gradoom.DEATHMATCH_ACTIONS) == 17
assert gradoom.scenario_buttons() == gradoom.DEATHMATCH_BUTTONS
distribution = next(PathDistribution.discover(path=[str(wheel)]))
assert distribution.metadata["Name"] == "gradoom"
assert distribution.version == sys.argv[2]
print("wheel import smoke passed")
"""
    with tempfile.TemporaryDirectory(prefix="gradoom-wheel-smoke-") as directory:
        subprocess.run(
            [sys.executable, "-c", code, str(wheel), version],
            cwd=directory,
            check=True,
            timeout=120,
        )


def audit_directory(directory: Path, version: str) -> dict[str, object]:
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit(
            f"expected one wheel and one sdist in {directory}; "
            f"found {len(wheels)} wheel(s) and {len(sdists)} sdist(s)"
        )
    wheel = wheels[0]
    sdist = sdists[0]
    result = {
        "version": version,
        "audits": [wheel_audit(wheel, version), sdist_audit(sdist, version)],
        "sha256": {wheel.name: sha256(wheel), sdist.name: sha256(sdist)},
    }
    smoke_wheel(wheel, version)
    print(json.dumps(result, indent=2))
    return result


def build(args: argparse.Namespace) -> None:
    check_version(argparse.Namespace(version=args.version))
    output = args.out_dir.resolve()
    if output.exists():
        raise SystemExit(f"release output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv",
            "build",
            "--no-sources",
            "--no-create-gitignore",
            "--out-dir",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    audit_directory(output, args.version)


def audit(args: argparse.Namespace) -> None:
    validate_version(args.version)
    audit_directory(args.dist_dir.resolve(), args.version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    version = commands.add_parser("check-version")
    version.add_argument("--version")
    version.set_defaults(func=check_version)

    prepare = commands.add_parser("prepare-version")
    prepare.add_argument("--to")
    prepare.add_argument("--write", action="store_true")
    prepare.set_defaults(func=prepare_version)

    pypi = commands.add_parser("check-pypi")
    pypi.add_argument("--version", required=True)
    pypi.set_defaults(func=check_pypi)

    candidate = commands.add_parser("build")
    candidate.add_argument("--version", required=True)
    candidate.add_argument("--out-dir", type=Path, required=True)
    candidate.set_defaults(func=build)

    existing = commands.add_parser("audit")
    existing.add_argument("--version", required=True)
    existing.add_argument("--dist-dir", type=Path, required=True)
    existing.set_defaults(func=audit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
