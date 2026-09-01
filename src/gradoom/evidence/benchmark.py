from __future__ import annotations

import ast
import base64
import binascii
import ctypes
import fcntl
import hashlib
import json
import math
import os
import resource
import shutil
import signal
import stat
import statistics
import subprocess
import time
import zipfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, BinaryIO

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .cuda_residency import CUDA_RESIDENCY_CATEGORIES, CUDA_RESIDENCY_CONTRACT
from .report import (
    EvidenceError,
    _canonical_sha256,
    _load_manifest,
    _parse_json_document,
    _resolve_evidence_path,
    _sha256_bytes,
    _validate_code_provenance,
    _validate_declared_inputs,
    _validate_schema_version,
    _validate_sha256,
)
from .time_authority import ReusableTimeAuthority, TimeAuthorityError
from .wad_profile import validate_wad_profile

_WORKFLOW = "development_training_benchmark"
_TRAINER_CONTRACT = "standalone-gradoom-deathmatch-ppo-v2"
_DEFAULT_TRAINING_SEED = 123
_UINT32_MAX = (1 << 32) - 1
_QUALITY_THRESHOLD = 30.0
_EVALUATION_EPISODES = 100
_FIXTURE_ELAPSED_ANCHOR_PUBLIC_KEY = "MfMyLUkj02xBwQm9sAmRkxh77ZmUIJbkkmokx379DS8="
_FIXTURE_ELAPSED_ANCHOR_AUTHORITY = "gradoom-fixture-independent-anchor-v1"
_AUTHORITY_STATE_ENV = "GRADOOM_REUSABLE_TIME_AUTHORITY_STATE"
_AUTHORITY_WITNESS_ENV = "GRADOOM_REUSABLE_TIME_AUTHORITY_WITNESS"
_CONTROLLED_ARGUMENTS = {
    "--checkpoint",
    "--checkpoint-every-rollouts",
    "--config-only",
    "--cuda-residency-acceptance",
    "--evaluate-checkpoint",
    "--evaluation-episodes",
    "--evaluation-seed",
    "--evaluation-seeds-file",
    "--evaluation-stochastic",
    "--evidence-attempt-identity",
    "--evidence-run-identity",
    "--initialize-from",
    "--metrics-jsonl",
    "--no-evaluation-stochastic",
    "--no-cuda-residency-acceptance",
    "--resume",
    "--seed",
    "--timesteps",
}

_MAX_CODE_ROOT_IDENTITY_MARKERS = 4096
_OPEN_DESCRIPTOR_RESERVE = 64
_SEALED_EXECUTION_EXIT = 125
_SEALED_EXECUTION_CONTRACT = "kernel-sealed-python-zip-v1"
_MFD_CLOEXEC = 0x0001
_MFD_ALLOW_SEALING = 0x0002
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_F_SEAL_SEAL = 0x0001
_F_SEAL_SHRINK = 0x0002
_F_SEAL_GROW = 0x0004
_F_SEAL_WRITE = 0x0008
_SEALED_PYTHON_BOOTSTRAP = r"""
import os
import sys
import zipfile

archive_path, code_root, entry_relative, *trainer_arguments = sys.argv[1:]
entry_path = os.path.join(code_root, entry_relative)
violations = []


def inside_code_root(value):
    if not isinstance(value, (str, bytes, os.PathLike)):
        return False
    try:
        candidate = os.path.realpath(os.fsdecode(value))
        return os.path.commonpath((code_root, candidate)) == code_root
    except (OSError, ValueError):
        return False


def reject_code_root_mutation(event, arguments):
    targets = ()
    if event == "open":
        path, mode, flags = arguments
        writing = (isinstance(mode, str) and any(marker in mode for marker in "wax+")) or (
            isinstance(flags, int)
            and flags
            & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        )
        if writing:
            targets = (path,)
    elif event in {
        "os.remove",
        "os.rmdir",
        "os.mkdir",
        "os.chmod",
        "os.chown",
        "os.truncate",
        "os.utime",
        "os.symlink",
    }:
        targets = arguments[:1]
    elif event in {"os.rename", "os.link"}:
        targets = arguments[:2]
    if any(inside_code_root(target) for target in targets):
        violations.append(event)
        raise PermissionError("sealed trainer execution binding forbids code-root mutation")


sys.addaudithook(reject_code_root_mutation)
search_roots = []
entry_parent = os.path.dirname(entry_relative)
for relative in (entry_parent, "", "src"):
    candidate = archive_path if not relative else archive_path + "/" + relative
    if candidate not in search_roots:
        search_roots.append(candidate)
sys.path[:] = search_roots + [
    item
    for item in sys.path
    if item and not inside_code_root(item)
]
sys.argv[:] = [entry_path, *trainer_arguments]
namespace = {
    "__name__": "__main__",
    "__file__": entry_path,
    "__package__": None,
    "__cached__": None,
    "__spec__": None,
}
try:
    with zipfile.ZipFile(archive_path) as archive:
        entry_source = archive.read(entry_relative)
    exec(compile(entry_source, entry_path, "exec"), namespace, namespace)
finally:
    if violations:
        sys.stderr.write("sealed trainer execution binding rejected a code-root mutation\n")
        sys.stderr.flush()
        os._exit(125)
"""

_RESTORABLE_STATE = {
    "policy": True,
    "optimizer": True,
    "rng": True,
    "progress": True,
}


def _formal_time_authority() -> ReusableTimeAuthority:
    raw_path = os.environ.get(_AUTHORITY_STATE_ENV)
    raw_witness = os.environ.get(_AUTHORITY_WITNESS_ENV)
    if not raw_path or not raw_witness:
        raise EvidenceError(
            "formal benchmark evidence requires "
            f"{_AUTHORITY_STATE_ENV} and {_AUTHORITY_WITNESS_ENV} to name the persistent "
            "repository-owned authority state and independently retained monotonic witness"
        )
    try:
        return ReusableTimeAuthority(Path(raw_path), Path(raw_witness))
    except TimeAuthorityError as error:
        raise EvidenceError(f"reusable-time authority state is invalid: {error}") from error


def _attempt_journal_payload(
    attempt: dict[str, Any],
    *,
    run_identity: str,
    generation: int | None = None,
    previous_journal_sha256: str | None = None,
) -> dict[str, Any]:
    recovery = attempt.get("recovery")
    normalized_recovery = recovery
    if isinstance(recovery, dict):
        normalized_recovery = {
            **recovery,
            "accumulated_reusable_elapsed_seconds": None,
        }
    payload = {
        "schema_version": 1,
        "run_identity": run_identity,
        "attempt_identity": attempt.get("attempt_identity"),
        "seed": attempt.get("seed"),
        "status": attempt.get("status"),
        # Final elapsed is issued by the external authority only after this payload is durable.
        "reusable_elapsed_seconds": None,
        "cold_start": attempt.get("cold_start"),
        "checkpoint": attempt.get("checkpoint"),
        "checkpoint_sha256": attempt.get("checkpoint_sha256"),
        "outcomes_sha256": _canonical_sha256(
            attempt.get("outcomes"),
            document="benchmark attempt outcomes",
        ),
        "failures_sha256": _canonical_sha256(
            attempt.get("failures"),
            document="benchmark attempt failures",
        ),
        "recovery_sha256": _canonical_sha256(
            {
                "recovery": normalized_recovery,
                "recovery_history": attempt.get("recovery_history"),
                "recovery_journal": attempt.get("recovery_journal"),
            },
            document="benchmark attempt recovery",
        ),
    }
    if generation is not None:
        payload["generation"] = generation
        payload["previous_journal_sha256"] = previous_journal_sha256
    return payload


def _required_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{field} is required and must be an object")
    return value


def _positive_integer(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise EvidenceError(f"{field} must be a positive integer")
    return value


def _seed_list(value: object, field: str, *, exact_count: int | None = None) -> list[int]:
    if not isinstance(value, list):
        raise EvidenceError(f"{field} must be an array")
    if exact_count is not None and len(value) != exact_count:
        raise EvidenceError(f"{field} must contain exactly {exact_count} seeds")
    if not value:
        raise EvidenceError(f"{field} must contain at least one seed")
    seeds: list[int] = []
    for index, seed in enumerate(value):
        if type(seed) is not int or not 0 <= seed <= _UINT32_MAX:
            raise EvidenceError(f"{field}[{index}] must be an integer in [0, {_UINT32_MAX}]")
        seeds.append(seed)
    if len(set(seeds)) != len(seeds):
        raise EvidenceError(f"{field} must be unique")
    return seeds


def _string_array(value: object, field: str, *, non_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (non_empty and not value):
        qualifier = "a non-empty array" if non_empty else "an array"
        raise EvidenceError(f"{field} must be {qualifier} of strings")
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise EvidenceError(f"{field}[{index}] must be a non-empty string")
    return value


def _validate_trainer(value: object) -> dict[str, Any]:
    trainer = _required_mapping(value, "benchmark.trainer")
    unknown = sorted(set(trainer) - {"command", "arguments", "code_root"})
    if unknown:
        raise EvidenceError(f"benchmark.trainer has undeclared fields: {', '.join(unknown)}")
    command = _string_array(trainer.get("command"), "benchmark.trainer.command", non_empty=True)
    arguments = _string_array(trainer.get("arguments", []), "benchmark.trainer.arguments")
    code_root = trainer.get("code_root")
    if code_root is not None and (not isinstance(code_root, str) or not code_root.strip()):
        raise EvidenceError("benchmark.trainer.code_root must be a non-whitespace path")
    for argument in arguments:
        controlled = next(
            (
                option
                for option in _CONTROLLED_ARGUMENTS
                if argument == option or argument.startswith(f"{option}=")
            ),
            None,
        )
        if controlled is not None:
            raise EvidenceError(
                f"benchmark.trainer.arguments must not control {controlled!r}; "
                "the evidence command owns cold-start, timing, checkpoint, and evaluation flags"
            )
    return {"command": command, "arguments": arguments, "code_root": code_root}


def _validated_python_tree(path: Path) -> ast.AST:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as error:
        raise EvidenceError(f"benchmark trainer Python source is unreadable: {path}") from error
    forbidden_imports = {"ctypes", "multiprocessing", "runpy", "subprocess", "sys"}
    forbidden_builtins = {"__import__", "eval", "exec", "compile"}
    forbidden_attributes = {
        "system",
        "popen",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "posix_spawn",
        "posix_spawnp",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "import_module",
        "find_spec",
        "module_from_spec",
        "__dict__",
        "__getattribute__",
        "__class__",
        "__bases__",
        "__mro__",
        "__subclasses__",
    }

    def static_string(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = static_string(node.left)
            right = static_string(node.right)
            if left is not None and right is not None:
                return left + right
        return None

    restricted_names: dict[str, str] = {}
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                bound = alias.asname or root
                if root == "os":
                    restricted_names[bound] = "os"
                elif alias.name == "importlib.resources" and alias.asname:
                    restricted_names[bound] = "importlib-resources"
                elif root == "importlib":
                    restricted_names[bound] = "importlib"
        elif isinstance(node, ast.ImportFrom):
            if node.module == "os":
                for alias in node.names:
                    if alias.name == "environ":
                        restricted_names[alias.asname or alias.name] = "os-environ"
            elif node.module == "importlib" and all(
                alias.name == "resources" for alias in node.names
            ):
                for alias in node.names:
                    restricted_names[alias.asname or alias.name] = "importlib-resources"

    def permitted_restricted_name(node: ast.Name) -> bool:
        kind = restricted_names[node.id]
        parent = parents.get(node)
        if (
            kind == "os"
            and isinstance(parent, ast.Attribute)
            and parent.value is node
            and parent.attr in {"_exit", "link"}
        ):
            grandparent = parents.get(parent)
            return isinstance(grandparent, ast.Call) and grandparent.func is parent
        allowed_first_attribute = {
            "os": "environ",
            "os-environ": "get",
            "importlib": "resources",
            "importlib-resources": "files",
        }[kind]
        if (
            not isinstance(parent, ast.Attribute)
            or parent.value is not node
            or parent.attr != allowed_first_attribute
        ):
            return False
        if kind in {"os-environ", "importlib-resources"}:
            grandparent = parents.get(parent)
            return isinstance(grandparent, ast.Call) and grandparent.func is parent
        grandparent = parents.get(parent)
        final_attribute = "get" if kind == "os" else "files"
        if (
            not isinstance(grandparent, ast.Attribute)
            or grandparent.value is not parent
            or grandparent.attr != final_attribute
        ):
            return False
        great_grandparent = parents.get(grandparent)
        return isinstance(great_grandparent, ast.Call) and great_grandparent.func is grandparent

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".", 1)[0] in forbidden_imports for alias in node.names):
                raise EvidenceError("shell, -c, eval, and opaque trainer indirection are forbidden")
            if any(
                alias.name == "importlib"
                or (
                    alias.name.startswith("importlib.")
                    and not alias.name.startswith("importlib.resources")
                )
                for alias in node.names
            ):
                raise EvidenceError("shell, -c, eval, and opaque trainer indirection are forbidden")
        if isinstance(node, ast.ImportFrom):
            root_module = (node.module or "").split(".", 1)[0]
            forbidden_importlib = root_module == "importlib" and not (
                node.module == "importlib.resources"
                or (
                    node.module == "importlib"
                    and all(alias.name == "resources" for alias in node.names)
                )
            )
            if (
                root_module in forbidden_imports
                or forbidden_importlib
                or (root_module == "os" and any(alias.name != "environ" for alias in node.names))
                or any(alias.name in forbidden_attributes for alias in node.names)
            ):
                raise EvidenceError("shell, -c, eval, and opaque trainer indirection are forbidden")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in {"globals", "locals", "__builtins__"}:
                raise EvidenceError("shell, -c, eval, and opaque trainer indirection are forbidden")
            if node.id in restricted_names and not permitted_restricted_name(node):
                raise EvidenceError("shell, -c, eval, and opaque trainer indirection are forbidden")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "vars" and not node.args:
                raise EvidenceError("shell, -c, eval, and opaque trainer indirection are forbidden")
            forbidden_call = (
                isinstance(node.func, ast.Name) and node.func.id in forbidden_builtins
            ) or (isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_attributes)
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
            ):
                attribute = static_string(node.args[1])
                dangerous_target = isinstance(node.args[0], ast.Name) and (
                    node.args[0].id in restricted_names
                )
                parent = parents.get(node)
                dynamic_callable = attribute is None and (
                    (isinstance(parent, ast.Call) and parent.func is node)
                    or isinstance(parent, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
                )
                if (
                    attribute in forbidden_attributes
                    or (dangerous_target and attribute is None)
                    or dynamic_callable
                ):
                    forbidden_call = True
            if forbidden_call:
                raise EvidenceError("shell, -c, eval, and opaque trainer indirection are forbidden")
    return tree


def _executes_during_module_import(
    node: ast.AST,
    *,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    parent = parents.get(node)
    while parent is not None:
        if isinstance(
            parent,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            return False
        parent = parents.get(parent)
    return True


def _inventory_code_root_paths(code_root: Path) -> list[Path]:
    try:
        root_metadata = code_root.lstat()
    except OSError as error:
        raise EvidenceError(
            f"benchmark trainer code_root cannot be inventoried: {code_root}"
        ) from error
    if stat.S_ISLNK(root_metadata.st_mode):
        raise EvidenceError(
            f"benchmark trainer code_root contains a symlink alias: {code_root}"
        )
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise EvidenceError(f"benchmark trainer code_root is not a directory: {code_root}")

    paths: list[Path] = []

    def inventory_error(error: OSError) -> None:
        raise error

    try:
        for directory, names, filenames in os.walk(code_root, onerror=inventory_error):
            directory_path = Path(directory)
            names.sort()
            filenames.sort()
            for name in names:
                child = directory_path / name
                try:
                    metadata = child.lstat()
                except OSError as error:
                    raise EvidenceError(
                        f"benchmark trainer code_root entry cannot be inventoried: {child}"
                    ) from error
                if stat.S_ISLNK(metadata.st_mode):
                    raise EvidenceError(
                        f"benchmark trainer code_root contains a symlink alias: {child}"
                    )
                if not stat.S_ISDIR(metadata.st_mode):
                    raise EvidenceError(
                        f"benchmark trainer code_root contains a non-regular entry: {child}"
                    )
            for filename in filenames:
                candidate = directory_path / filename
                try:
                    metadata = candidate.lstat()
                except OSError as error:
                    raise EvidenceError(
                        f"benchmark trainer code_root entry cannot be inventoried: {candidate}"
                    ) from error
                if stat.S_ISLNK(metadata.st_mode):
                    raise EvidenceError(
                        f"benchmark trainer code_root contains a symlink alias: {candidate}"
                    )
                if not stat.S_ISREG(metadata.st_mode):
                    raise EvidenceError(
                        f"benchmark trainer code_root contains a non-regular entry: {candidate}"
                    )
                paths.append(candidate)
    except OSError as error:
        raise EvidenceError(
            f"benchmark trainer code_root cannot be inventoried: {code_root}"
        ) from error
    return paths


class _TrainerFileIdentityMarkers:
    def __init__(self) -> None:
        self.streams: dict[str, BinaryIO] = {}
        self.archive_descriptor: int | None = None
        self.archive_writer_stream: BinaryIO | None = None
        self.archive_writer: zipfile.ZipFile | None = None
        self.sealed_archive: BinaryIO | None = None
        self.sealed_archive_sha256: str | None = None
        self.code_root: Path | None = None
        self.entry_relative: Path | None = None
        self.executable_argv0: str | None = None

    def close(self) -> None:
        streams, self.streams = self.streams, {}
        for stream in streams.values():
            stream.close()
        writer, self.archive_writer = self.archive_writer, None
        if writer is not None:
            writer.close()
        writer_stream, self.archive_writer_stream = self.archive_writer_stream, None
        if writer_stream is not None:
            writer_stream.close()
        descriptor, self.archive_descriptor = self.archive_descriptor, None
        if descriptor is not None:
            os.close(descriptor)
        archive, self.sealed_archive = self.sealed_archive, None
        if archive is not None:
            archive.close()

    def __del__(self) -> None:
        with suppress(OSError):
            self.close()


def _identity_marker_capacity() -> int:
    soft_limit, _hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit == resource.RLIM_INFINITY:
        return _MAX_CODE_ROOT_IDENTITY_MARKERS
    descriptor_directories = ("/proc/self/fd", "/dev/fd")
    for descriptor_directory in descriptor_directories:
        try:
            # listdir may transiently count its own directory descriptor, which makes this
            # fail-closed calculation conservatively smaller by at most one marker.
            open_descriptors = len(os.listdir(descriptor_directory))
            break
        except OSError:
            continue
    else:
        return 0
    available = max(0, int(soft_limit) - open_descriptors - _OPEN_DESCRIPTOR_RESERVE)
    return min(_MAX_CODE_ROOT_IDENTITY_MARKERS, available)


def _snapshot_regular_code_file(
    path: Path,
    *,
    code_root: Path,
    identity_markers: _TrainerFileIdentityMarkers | None = None,
) -> dict[str, Any]:
    stream: BinaryIO | None = None
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise EvidenceError(
                f"benchmark trainer code_root entry is no longer a regular file: {path}"
            )
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            stream = os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
        try:
            opened = os.fstat(descriptor)
            payload = stream.read()
            after_read = os.fstat(descriptor)
        except BaseException:
            stream.close()
            stream = None
            raise
        after_path = path.lstat()
    except EvidenceError:
        if stream is not None:
            stream.close()
        raise
    except OSError as error:
        if stream is not None:
            stream.close()
        raise EvidenceError(
            f"benchmark trainer code_root file changed while being inventoried: {path}"
        ) from error

    def stable_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
    if not (
        stable_identity(before)
        == stable_identity(opened)
        == stable_identity(after_read)
        == stable_identity(after_path)
    ):
        assert stream is not None
        stream.close()
        raise EvidenceError(
            f"benchmark trainer code_root file changed while being inventoried: {path}"
        )
    relative_path = str(path.relative_to(code_root))
    snapshot = {
        "role": "code-root-file",
        "path": str(path),
        "relative_path": relative_path,
        "sha256": _sha256_bytes(payload),
        "device": before.st_dev,
        "inode": before.st_ino,
        "mode": stat.S_IMODE(before.st_mode),
        "size": before.st_size,
    }
    if identity_markers is None:
        assert stream is not None
        stream.close()
    else:
        assert stream is not None
        identity_markers.streams[relative_path] = stream
        _append_trainer_execution_payload(identity_markers, relative_path, payload)
    return snapshot


def _new_sealable_memory_file() -> int:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        memfd_create = libc.memfd_create
        memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
        memfd_create.restype = ctypes.c_int
        descriptor = memfd_create(b"gradoom-sealed-trainer", _MFD_ALLOW_SEALING | _MFD_CLOEXEC)
    except (AttributeError, OSError) as error:
        raise EvidenceError(
            "benchmark trainer requires kernel-sealed in-memory execution binding support"
        ) from error
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise EvidenceError(
            "benchmark trainer requires kernel-sealed in-memory execution binding support"
        ) from OSError(error_number, os.strerror(error_number))
    return descriptor


def _append_trainer_execution_payload(
    identity_markers: _TrainerFileIdentityMarkers,
    relative_path: str,
    payload: bytes,
) -> None:
    if identity_markers.archive_descriptor is None:
        descriptor = _new_sealable_memory_file()
        try:
            writer_stream = os.fdopen(os.dup(descriptor), "w+b")
            writer = zipfile.ZipFile(writer_stream, mode="w", compression=zipfile.ZIP_STORED)
        except BaseException:
            os.close(descriptor)
            raise
        identity_markers.archive_descriptor = descriptor
        identity_markers.archive_writer_stream = writer_stream
        identity_markers.archive_writer = writer
    writer = identity_markers.archive_writer
    assert writer is not None
    info = zipfile.ZipInfo(relative_path, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100444 << 16
    writer.writestr(info, payload)


def _seal_trainer_execution_archive(identity_markers: _TrainerFileIdentityMarkers) -> None:
    descriptor = identity_markers.archive_descriptor
    writer = identity_markers.archive_writer
    archive_stream = identity_markers.archive_writer_stream
    if descriptor is None or writer is None or archive_stream is None:
        raise EvidenceError("benchmark trainer execution archive has no source payloads")
    try:
        writer.close()
        identity_markers.archive_writer = None
        archive_stream.flush()
        os.fsync(archive_stream.fileno())
        archive_stream.close()
        identity_markers.archive_writer_stream = None
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        with os.fdopen(os.dup(descriptor), "rb") as archive_reader:
            while chunk := archive_reader.read(1024 * 1024):
                digest.update(chunk)
        seals = _F_SEAL_SHRINK | _F_SEAL_GROW | _F_SEAL_WRITE | _F_SEAL_SEAL
        fcntl.fcntl(descriptor, _F_ADD_SEALS, seals)
        if fcntl.fcntl(descriptor, _F_GET_SEALS) & seals != seals:
            raise EvidenceError("benchmark trainer execution archive could not be kernel-sealed")
        identity_markers.sealed_archive = os.fdopen(descriptor, "rb")
        identity_markers.archive_descriptor = None
        identity_markers.sealed_archive_sha256 = digest.hexdigest()
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise EvidenceError("benchmark trainer execution archive could not be sealed") from error


def _bound_code_root_files(
    script_path: Path,
    code_root: Path,
    *,
    identity_markers: _TrainerFileIdentityMarkers | None = None,
) -> list[dict[str, Any]]:
    paths = _python_source_closure(script_path, code_root)
    if identity_markers is not None and len(paths) > _identity_marker_capacity():
        raise EvidenceError(
            "benchmark trainer code_root exceeds the safe open-file descriptor budget "
            "for replacement detection"
        )
    try:
        bound = [
            _snapshot_regular_code_file(
                path,
                code_root=code_root,
                identity_markers=identity_markers,
            )
            if identity_markers is not None
            else _snapshot_regular_code_file(path, code_root=code_root)
            for path in paths
        ]
        membership_after_hashing = _inventory_code_root_paths(code_root)
        if {path.relative_to(code_root) for path in paths} != {
            path.relative_to(code_root) for path in membership_after_hashing
        }:
            raise EvidenceError(
                "benchmark trainer code-root membership changed while being inventoried"
            )
        if identity_markers is not None:
            _seal_trainer_execution_archive(identity_markers)
        return bound
    except BaseException:
        if identity_markers is not None:
            identity_markers.close()
        raise


def _python_source_closure(
    script_path: Path,
    code_root: Path,
) -> list[Path]:
    # Reject aliases and establish the whole-byte membership boundary before resolving
    # static imports.  In particular, a symlink to a mutable output directory must never
    # disappear through target-based exclusion logic.
    local_sources = set(_inventory_code_root_paths(code_root))
    pending = [script_path]
    closure: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in closure:
            continue
        closure.add(path)
        tree = _validated_python_tree(path)
        parents = {
            child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }
        modules: set[str] = set()
        relative_modules: list[tuple[int, str]] = []
        relative_members: list[tuple[int, str, str]] = []
        imported_members: list[tuple[str, str]] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and _executes_during_module_import(
                node, parents=parents
            ):
                modules.update(alias.name for alias in node.names)
            elif (
                isinstance(node, ast.ImportFrom)
                and _executes_during_module_import(node, parents=parents)
                and node.level == 0
                and node.module
            ):
                modules.add(node.module)
                imported_members.extend((node.module, alias.name) for alias in node.names)
            elif (
                isinstance(node, ast.ImportFrom)
                and _executes_during_module_import(node, parents=parents)
                and node.level > 0
            ):
                relative_modules.extend(
                    (node.level, module)
                    for module in (
                        [node.module] if node.module else [alias.name for alias in node.names]
                    )
                )
                if node.module:
                    relative_members.extend(
                        (node.level, node.module, alias.name)
                        for alias in node.names
                        if alias.name != "*"
                    )
        for module in modules:
            parts = module.split(".")
            candidates = (
                code_root.joinpath(*parts).with_suffix(".py"),
                code_root.joinpath(*parts, "__init__.py"),
                code_root.joinpath("src", *parts).with_suffix(".py"),
                code_root.joinpath("src", *parts, "__init__.py"),
                path.parent.joinpath(*parts).with_suffix(".py"),
            )
            resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
            if resolved is None:
                continue
            pending.append(resolved)
            relative = resolved.relative_to(code_root)
            for parent in relative.parents:
                if parent == Path("."):
                    continue
                initializer = code_root / parent / "__init__.py"
                if initializer.is_file():
                    pending.append(initializer)
        for module, member in imported_members:
            parts = [*module.split("."), *member.split(".")]
            candidates = (
                code_root.joinpath(*parts).with_suffix(".py"),
                code_root.joinpath(*parts, "__init__.py"),
                code_root.joinpath("src", *parts).with_suffix(".py"),
                code_root.joinpath("src", *parts, "__init__.py"),
                path.parent.joinpath(*parts).with_suffix(".py"),
            )
            resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
            if resolved is not None:
                pending.append(resolved)
                relative = resolved.relative_to(code_root)
                for parent in relative.parents:
                    if parent == Path("."):
                        continue
                    initializer = code_root / parent / "__init__.py"
                    if initializer.is_file():
                        pending.append(initializer)
        for level, module in relative_modules:
            base = path.parent
            for _ in range(level - 1):
                base = base.parent
            parts = module.split(".")
            candidates = (
                base.joinpath(*parts).with_suffix(".py"),
                base.joinpath(*parts, "__init__.py"),
            )
            resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
            if resolved is None:
                continue
            pending.append(resolved)
            relative = resolved.relative_to(code_root)
            for parent in relative.parents:
                if parent == Path("."):
                    continue
                initializer = code_root / parent / "__init__.py"
                if initializer.is_file():
                    pending.append(initializer)
        for level, module, member in relative_members:
            base = path.parent
            for _ in range(level - 1):
                base = base.parent
            parts = [*module.split("."), *member.split(".")]
            candidates = (
                base.joinpath(*parts).with_suffix(".py"),
                base.joinpath(*parts, "__init__.py"),
            )
            resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
            if resolved is None:
                continue
            pending.append(resolved)
            relative = resolved.relative_to(code_root)
            for parent in relative.parents:
                if parent == Path("."):
                    continue
                initializer = code_root / parent / "__init__.py"
                if initializer.is_file():
                    pending.append(initializer)
    # Static import closure rejects known opaque launch primitives, while the complete
    # inventory binds arbitrary bytes regardless of suffix, encoding, or parseability.
    return sorted(closure | local_sources)


def _bind_trainer_files(
    trainer: dict[str, Any],
    *,
    base_directory: Path,
    artifacts_root: Path,
) -> dict[str, Any]:
    command = trainer["command"]
    raw_executable = Path(command[0])
    if raw_executable.is_absolute() or raw_executable.parent != Path("."):
        executable_argv0 = (
            raw_executable if raw_executable.is_absolute() else base_directory / raw_executable
        ).absolute()
        executable_path = _resolve_evidence_path(raw_executable, base_directory=base_directory)
    else:
        executable = shutil.which(command[0])
        executable_argv0 = (
            (base_directory / raw_executable).absolute()
            if executable is None
            else Path(executable).absolute()
        )
        executable_path = (
            _resolve_evidence_path(raw_executable, base_directory=base_directory)
            if executable is None
            else Path(executable).resolve()
        )
    executable_name = executable_path.name.lower()
    if executable_name in {"sh", "bash", "dash", "zsh", "fish", "cmd", "powershell", "pwsh"}:
        raise EvidenceError("shell, -c, eval, and opaque trainer indirection are forbidden")
    is_python = executable_name.startswith(("python", "pypy"))
    if is_python and (len(command) != 2 or command[1].startswith("-")):
        raise EvidenceError("shell, -c, eval, and opaque trainer indirection are forbidden")
    if not is_python:
        raise EvidenceError(
            "non-Python trainer wrappers are ineligible because their executed-code closure "
            "cannot be proven"
        )
    bound_files = [
        {
            "role": "executable",
            "path": str(executable_path),
            "sha256": (
                _sha256_bytes(executable_path.read_bytes()) if executable_path.is_file() else None
            ),
        }
    ]
    if is_python:
        script_path = _resolve_evidence_path(Path(command[1]), base_directory=base_directory)
        if not script_path.is_file() or script_path.suffix != ".py":
            raise EvidenceError("benchmark trainer Python command must name one source file")
        raw_root = trainer.get("code_root")
        declared_root = Path(raw_root) if raw_root is not None else script_path.parent
        if not declared_root.is_absolute():
            declared_root = base_directory / declared_root
        if declared_root.is_symlink():
            raise EvidenceError(
                f"benchmark trainer code_root contains a symlink alias: {declared_root}"
            )
        code_root = _resolve_evidence_path(
            declared_root,
            base_directory=base_directory,
        )
        if not code_root.is_dir() or not script_path.is_relative_to(code_root):
            raise EvidenceError("benchmark trainer script must be inside its code_root")
        if (
            artifacts_root == code_root
            or artifacts_root.is_relative_to(code_root)
            or code_root.is_relative_to(artifacts_root)
        ):
            raise EvidenceError("benchmark artifacts must be outside trainer code_root")
        identity_markers = _TrainerFileIdentityMarkers()
        identity_markers.code_root = code_root
        identity_markers.entry_relative = script_path.relative_to(code_root)
        identity_markers.executable_argv0 = str(executable_argv0)
        bound_files.extend(
            _bound_code_root_files(
                script_path,
                code_root,
                identity_markers=identity_markers,
            )
        )
        assert identity_markers.sealed_archive_sha256 is not None
        execution_binding = {
            "contract": _SEALED_EXECUTION_CONTRACT,
            "archive_sha256": identity_markers.sealed_archive_sha256,
            "bootstrap_sha256": _sha256_bytes(_SEALED_PYTHON_BOOTSTRAP.encode()),
        }
        resolved_root: str | None = str(code_root)
    else:
        resolved_root = None
    bound_command = [str(executable_path)]
    if is_python:
        bound_command.append(str(script_path))
    return {
        **trainer,
        "command": bound_command,
        "code_root": resolved_root,
        "bound_files": bound_files,
        "execution_binding": execution_binding,
        "_identity_markers": identity_markers,
    }


def _reverify_trainer_files(trainer: dict[str, Any]) -> None:
    identity_markers = trainer["_identity_markers"]
    try:
        executable = trainer["bound_files"][0]
        for bound in (executable,):
            path = Path(bound["path"])
            if bound["sha256"] is None and not path.exists():
                continue
            try:
                digest = _sha256_bytes(path.read_bytes())
            except OSError as error:
                raise EvidenceError(
                    f"bound trainer file changed during the cohort: {path}"
                ) from error
            if digest != bound["sha256"]:
                raise EvidenceError(f"bound trainer file changed during the cohort: {path}")
        code_root = Path(trainer["code_root"])
        script_path = Path(trainer["command"][1])
        current_files = _bound_code_root_files(script_path, code_root)
        expected_by_relative = {
            item["relative_path"]: item
            for item in trainer["bound_files"]
            if item["role"] == "code-root-file"
        }
        current_by_relative = {item["relative_path"]: item for item in current_files}
        if set(expected_by_relative) != set(current_by_relative):
            raise EvidenceError("benchmark trainer code-root membership changed during the cohort")
        if set(identity_markers.streams) != set(expected_by_relative):
            raise EvidenceError("benchmark trainer identity markers are incomplete")
        for relative_path, expected in expected_by_relative.items():
            current = current_by_relative[relative_path]
            path = current["path"]
            if current["sha256"] != expected["sha256"]:
                raise EvidenceError(f"bound trainer file changed during the cohort: {path}")
            if any(
                current[field] != expected[field]
                for field in ("device", "inode", "mode", "size")
            ):
                raise EvidenceError(
                    f"bound trainer file identity changed during the cohort: {path}"
                )
            try:
                current_descriptor = os.open(
                    path,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    same_file = os.path.sameopenfile(
                        identity_markers.streams[relative_path].fileno(),
                        current_descriptor,
                    )
                finally:
                    os.close(current_descriptor)
            except OSError as error:
                raise EvidenceError(
                    f"bound trainer file identity changed during the cohort: {path}"
                ) from error
            if not same_file:
                raise EvidenceError(
                    f"bound trainer file identity changed during the cohort: {path}"
                )
    finally:
        identity_markers.close()


def _validate_certificate(value: object) -> dict[str, Any]:
    if value is None:
        return {
            "available": False,
            "reason": "No current parity certificate was declared.",
        }
    certificate = _required_mapping(value, "benchmark.parity_certificate")
    if type(certificate.get("available")) is not bool:
        raise EvidenceError("benchmark.parity_certificate.available must be a boolean")
    if not certificate["available"]:
        reason = certificate.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise EvidenceError(
                "benchmark.parity_certificate.reason must explain an unavailable certificate"
            )
    return certificate


def _decode_base64(value: object, field: str, *, length: int) -> bytes:
    if not isinstance(value, str):
        raise EvidenceError(f"{field} must be base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise EvidenceError(f"{field} must be valid base64") from error
    if len(decoded) != length:
        raise EvidenceError(f"{field} has the wrong byte length")
    return decoded


def _validate_elapsed_time_anchors(
    value: object,
    *,
    training_seeds: list[int],
    fixture: bool,
) -> list[dict[str, Any]]:
    if value is None:
        raise EvidenceError(
            "benchmark.elapsed_time_anchors are required for every externally anchored outcome"
        )
    if not isinstance(value, list):
        raise EvidenceError("benchmark.elapsed_time_anchors must be an array")
    anchors: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        field = f"benchmark.elapsed_time_anchors[{index}]"
        anchor = _required_mapping(raw, field)
        if set(anchor) != {"payload", "public_key", "signature"}:
            raise EvidenceError(f"{field} has an unsupported contract")
        payload = _required_mapping(anchor.get("payload"), f"{field}.payload")
        if set(payload) != {"schema_version", "authority", "seed", "started_unix_ns"}:
            raise EvidenceError(f"{field}.payload has an unsupported contract")
        if payload.get("schema_version") != 1:
            raise EvidenceError(f"{field}.payload schema is unsupported")
        authority = payload.get("authority")
        seed = payload.get("seed")
        started_unix_ns = payload.get("started_unix_ns")
        if not isinstance(authority, str) or not authority.strip():
            raise EvidenceError(f"{field}.payload.authority must identify an external signer")
        if type(seed) is not int or seed not in training_seeds:
            raise EvidenceError(f"{field}.payload.seed is not a declared training seed")
        if type(started_unix_ns) is not int or started_unix_ns <= 0:
            raise EvidenceError(f"{field}.payload.started_unix_ns must be positive")
        public_key_text = anchor.get("public_key")
        if fixture:
            expected_authority = _FIXTURE_ELAPSED_ANCHOR_AUTHORITY
            expected_public_key = _FIXTURE_ELAPSED_ANCHOR_PUBLIC_KEY
        else:
            identity = _formal_time_authority().identity
            expected_authority = identity["authority"]
            expected_public_key = identity["public_key"]
        if authority != expected_authority or public_key_text != expected_public_key:
            raise EvidenceError(f"{field} is not rooted in the pinned public authority")
        public_key = _decode_base64(public_key_text, f"{field}.public_key", length=32)
        signature = _decode_base64(anchor.get("signature"), f"{field}.signature", length=64)
        signed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        try:
            Ed25519PublicKey.from_public_bytes(public_key).verify(signature, signed)
        except InvalidSignature as error:
            raise EvidenceError(f"{field} signature is invalid") from error
        anchors.append(
            {
                "payload": dict(payload),
                "public_key": public_key_text,
                "signature": anchor["signature"],
            }
        )
    if [item["payload"]["seed"] for item in anchors] != training_seeds:
        raise EvidenceError("benchmark.elapsed_time_anchors must cover each training seed in order")
    return anchors


def _attestation_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _verify_generation_attestation(
    attestation: object,
    *,
    anchor: dict[str, Any],
    expected_payload: dict[str, Any],
    recover_same_head: bool = False,
) -> dict[str, Any]:
    value = _required_mapping(attestation, "attempt journal authority_attestation")
    if set(value) != {"payload", "signature"} or value.get("payload") != expected_payload:
        raise EvidenceError("attempt journal authority attestation payload mismatch")
    public_key = _decode_base64(anchor["public_key"], "elapsed anchor public key", length=32)
    signature = _decode_base64(value.get("signature"), "authority attestation signature", length=64)
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature, _attestation_bytes(expected_payload)
        )
    except InvalidSignature as error:
        raise EvidenceError("attempt journal authority attestation signature is invalid") from error
    if anchor["payload"]["authority"] != _FIXTURE_ELAPSED_ANCHOR_AUTHORITY:
        authority = _formal_time_authority()
        try:
            authority.verify_latest_journal_head(value)
        except TimeAuthorityError as error:
            if recover_same_head:
                try:
                    return authority.recover_latest_journal_head(value)
                except TimeAuthorityError:
                    pass
            raise EvidenceError(
                f"reusable-time authority rejected the journal head: {error}"
            ) from error
    return dict(value)


def _sign_generation_attestation(
    payload: dict[str, Any],
    *,
    anchor: dict[str, Any],
    prior_elapsed: float,
    minimum_elapsed: float,
    started: float,
    clock: Callable[[], float],
) -> dict[str, Any]:
    authority = anchor["payload"]["authority"]
    if authority == _FIXTURE_ELAPSED_ANCHOR_AUTHORITY:
        payload = {
            **payload,
            "reusable_elapsed_seconds": max(
                minimum_elapsed,
                prior_elapsed + max(0.0, clock() - started),
            ),
        }
        private_key = Ed25519PrivateKey.from_private_bytes(b"\x19" * 32)
        signature = private_key.sign(_attestation_bytes(payload))
    else:
        request = {
            **payload,
            "prior_reusable_elapsed_seconds": prior_elapsed,
            "minimum_reusable_elapsed_seconds": minimum_elapsed,
        }
        try:
            attestation = _formal_time_authority().sign_journal_head(request)
        except TimeAuthorityError as error:
            raise EvidenceError(f"reusable-time authority refused journal seal: {error}") from error
        sealed_payload = attestation.get("payload")
        if not isinstance(sealed_payload, dict) or any(
            sealed_payload.get(key) != value for key, value in payload.items()
        ):
            raise EvidenceError("external authority journal seal changed the requested head")
        elapsed = sealed_payload.get("reusable_elapsed_seconds")
        if type(elapsed) not in (int, float) or float(elapsed) < minimum_elapsed:
            raise EvidenceError("external authority journal seal has an invalid elapsed time")
        payload = sealed_payload
        signature = _decode_base64(attestation.get("signature"), "authority signature", length=64)
    attestation = {
        "payload": payload,
        "signature": base64.b64encode(signature).decode(),
    }
    _verify_generation_attestation(attestation, anchor=anchor, expected_payload=payload)
    return attestation


def _journal_attestation_payload(
    journal_payload: dict[str, Any],
    *,
    anchor: dict[str, Any],
    journal_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "authority": anchor["payload"]["authority"],
        "seed": journal_payload["seed"],
        "started_unix_ns": anchor["payload"]["started_unix_ns"],
        "generation": journal_payload["generation"],
        "previous_journal_sha256": journal_payload["previous_journal_sha256"],
        "journal_sha256": journal_sha256,
        "status": journal_payload["status"],
    }


_BOOTSTRAP_CONTRACT = "gradoom-declarative-bootstrap-v1"
_BOOTSTRAP_PROTOCOL = "canonical-declared-input-binding-v1"


def _validate_bootstrap_artifacts(value: object, *, fixture: bool) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise EvidenceError("benchmark.bootstrap_artifacts must be an array")
    artifacts: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw_artifact in enumerate(value):
        field = f"benchmark.bootstrap_artifacts[{index}]"
        artifact = _required_mapping(raw_artifact, field)
        expected_fields = {
            "name",
            "path",
            "sha256",
            "creation_elapsed_seconds",
            "contract",
            "immutable_inputs",
            "creation_protocol",
            "reuse_conditions",
            "persistent",
            "run_independent",
            "reused_unchanged",
            "eligibility_attestation",
        }
        undeclared_fields = sorted(set(artifact) - expected_fields)
        if undeclared_fields:
            formatted = ", ".join(repr(name) for name in undeclared_fields)
            raise EvidenceError(f"{field} has undeclared fields: {formatted}")
        name = artifact.get("name")
        if not isinstance(name, str) or not name.strip():
            raise EvidenceError(f"{field}.name must be a non-whitespace string")
        if name in names:
            raise EvidenceError(f"{field}.name {name!r} is duplicated")
        names.add(name)
        path = artifact.get("path")
        if not isinstance(path, str) or not path.strip():
            raise EvidenceError(f"{field}.path must be a non-whitespace path")
        sha256 = _validate_sha256(artifact.get("sha256"), f"{field}.sha256")
        creation_elapsed = artifact.get("creation_elapsed_seconds")
        if (
            type(creation_elapsed) not in (int, float)
            or not math.isfinite(float(creation_elapsed))
            or float(creation_elapsed) < 0.0
        ):
            raise EvidenceError(f"{field}.creation_elapsed_seconds must be finite and non-negative")
        creation_protocol = artifact.get("creation_protocol")
        if artifact.get("contract") != _BOOTSTRAP_CONTRACT:
            raise EvidenceError(f"{field}.contract must use {_BOOTSTRAP_CONTRACT}")
        if creation_protocol != _BOOTSTRAP_PROTOCOL:
            raise EvidenceError(f"{field} must use the canonical declarative bootstrap protocol")
        immutable_inputs = artifact.get("immutable_inputs")
        if not isinstance(immutable_inputs, list) or not immutable_inputs:
            raise EvidenceError(f"{field}.immutable_inputs must contain at least one input")
        normalized_inputs: list[dict[str, str]] = []
        input_names: set[str] = set()
        for input_index, item in enumerate(immutable_inputs):
            if not isinstance(item, dict) or set(item) != {"name", "sha256"}:
                raise EvidenceError(f"{field}.immutable_inputs[{input_index}] is malformed")
            input_name = item.get("name")
            if not isinstance(input_name, str) or not input_name or input_name in input_names:
                raise EvidenceError(f"{field}.immutable_inputs names must be unique strings")
            input_names.add(input_name)
            normalized_inputs.append(
                {
                    "name": input_name,
                    "sha256": _validate_sha256(
                        item.get("sha256"), f"{field}.immutable_inputs[{input_index}].sha256"
                    ),
                }
            )
        if normalized_inputs != sorted(normalized_inputs, key=lambda item: item["name"]):
            raise EvidenceError(f"{field}.immutable_inputs must be sorted by name")
        reuse_conditions = _string_array(
            artifact.get("reuse_conditions"),
            f"{field}.reuse_conditions",
            non_empty=True,
        )
        for required_true in ("persistent", "run_independent", "reused_unchanged"):
            if artifact.get(required_true) is not True:
                raise EvidenceError(f"{field}.{required_true} must be true for excluded work")
        if normalized_inputs != [
            {"name": "compiler-target", "sha256": normalized_inputs[0]["sha256"]}
        ]:
            raise EvidenceError(
                f"{field}.immutable_inputs must use the constrained compiler-target protocol"
            )
        canonical_reuse = [
            "exact compiler and target identity",
            "read-only bytes reused without transformation",
        ]
        if reuse_conditions != canonical_reuse:
            raise EvidenceError(f"{field}.reuse_conditions are not the canonical protocol")
        attestation = _required_mapping(
            artifact.get("eligibility_attestation"), f"{field}.eligibility_attestation"
        )
        if set(attestation) != {"payload", "public_key", "signature"}:
            raise EvidenceError(f"{field}.eligibility_attestation has an unsupported contract")
        if fixture:
            expected_authority = _FIXTURE_ELAPSED_ANCHOR_AUTHORITY
            expected_public_key = _FIXTURE_ELAPSED_ANCHOR_PUBLIC_KEY
        else:
            identity = _formal_time_authority().identity
            expected_authority = identity["authority"]
            expected_public_key = identity["public_key"]
        expected_attestation_payload = {
            "schema_version": 1,
            "authority": expected_authority,
            "artifact_name": name,
            "artifact_sha256": sha256,
            "creation_elapsed_seconds": float(creation_elapsed),
            "creation_protocol": creation_protocol,
            "immutable_inputs": normalized_inputs,
            "reuse_conditions": reuse_conditions,
            "artifact_identity": attestation.get("payload", {}).get("artifact_identity")
            if isinstance(attestation.get("payload"), dict)
            else None,
            "creation_event": attestation.get("payload", {}).get("creation_event")
            if isinstance(attestation.get("payload"), dict)
            else None,
            "prior_reuse_event": attestation.get("payload", {}).get("prior_reuse_event")
            if isinstance(attestation.get("payload"), dict)
            else None,
        }
        artifact_identity = expected_attestation_payload["artifact_identity"]
        creation_event = expected_attestation_payload["creation_event"]
        prior_reuse_event = expected_attestation_payload["prior_reuse_event"]
        if not isinstance(artifact_identity, dict) or set(artifact_identity) != {
            "resolved_path",
            "device",
            "inode",
        }:
            raise EvidenceError(f"{field}.eligibility_attestation has no artifact object identity")
        if not isinstance(creation_event, dict) or creation_event.get("artifact_sha256") != sha256:
            raise EvidenceError(f"{field}.eligibility_attestation has no valid creation event")
        if not isinstance(prior_reuse_event, dict):
            raise EvidenceError(
                f"{field}.eligibility_attestation has no prior unchanged reuse event"
            )
        if (
            prior_reuse_event.get("artifact_sha256") != sha256
            or prior_reuse_event.get("artifact_identity") != artifact_identity
            or not isinstance(prior_reuse_event.get("event_id"), str)
            or not prior_reuse_event["event_id"]
            or prior_reuse_event.get("event_id") == creation_event.get("event_id")
        ):
            raise EvidenceError(
                f"{field}.eligibility_attestation prior unchanged reuse event is invalid"
            )
        if (
            attestation.get("payload") != expected_attestation_payload
            or attestation.get("public_key") != expected_public_key
        ):
            raise EvidenceError(f"{field}.eligibility_attestation does not bind canonical claims")
        public_key = _decode_base64(
            attestation["public_key"], f"{field}.eligibility_attestation.public_key", length=32
        )
        signature = _decode_base64(
            attestation.get("signature"),
            f"{field}.eligibility_attestation.signature",
            length=64,
        )
        try:
            Ed25519PublicKey.from_public_bytes(public_key).verify(
                signature, _attestation_bytes(expected_attestation_payload)
            )
        except InvalidSignature as error:
            raise EvidenceError(f"{field}.eligibility_attestation signature is invalid") from error
        artifacts.append(
            {
                "name": name,
                "path": path,
                "sha256": sha256,
                "creation_elapsed_seconds": float(creation_elapsed),
                "contract": _BOOTSTRAP_CONTRACT,
                "immutable_inputs": normalized_inputs,
                "creation_protocol": creation_protocol,
                "reuse_conditions": reuse_conditions,
                "persistent": True,
                "run_independent": True,
                "reused_unchanged": True,
                "eligibility_attestation": dict(attestation),
            }
        )
    return artifacts


def _validate_bootstrap_files(
    declarations: list[dict[str, Any]],
    *,
    base_directory: Path,
    artifacts_root: Path,
    declared_inputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    declared_by_name = {item["name"]: item["sha256"] for item in declared_inputs}
    declared_paths = {item["name"]: item["path"] for item in declared_inputs}
    for declaration in declarations:
        path = _resolve_evidence_path(Path(declaration["path"]), base_directory=base_directory)
        if path == artifacts_root or path.is_relative_to(artifacts_root):
            raise EvidenceError(
                f"bootstrap artifact {declaration['name']!r} must persist outside "
                "benchmark artifacts"
            )
        try:
            metadata = path.lstat()
            payload = path.read_bytes()
        except OSError as error:
            raise EvidenceError(
                f"bootstrap artifact {declaration['name']!r} is missing or unreadable: {path}"
            ) from error
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise EvidenceError(
                f"bootstrap artifact {declaration['name']!r} must be a regular file"
            )
        if metadata.st_nlink != 1 or metadata.st_mode & 0o222:
            raise EvidenceError(f"bootstrap artifact {declaration['name']!r} is mutable")
        attestation = declaration["eligibility_attestation"]
        signed_identity = attestation["payload"]["artifact_identity"]
        actual_identity = {
            "resolved_path": str(path),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
        }
        if signed_identity != actual_identity:
            raise EvidenceError(
                f"bootstrap artifact {declaration['name']!r} does not match its externally "
                "attested artifact object identity"
            )
        if attestation["payload"]["prior_reuse_event"]["artifact_identity"] != actual_identity:
            raise EvidenceError(
                f"bootstrap artifact {declaration['name']!r} has no prior unchanged reuse of "
                "this artifact object"
            )
        if attestation["payload"]["authority"] != _FIXTURE_ELAPSED_ANCHOR_AUTHORITY:
            try:
                _formal_time_authority().verify_bootstrap_reuse(attestation)
            except TimeAuthorityError as error:
                raise EvidenceError(
                    f"reusable-time authority rejected bootstrap reuse history: {error}"
                ) from error
        actual_sha256 = _sha256_bytes(payload)
        if actual_sha256 != declaration["sha256"]:
            raise EvidenceError(
                f"bootstrap artifact {declaration['name']!r} SHA-256 mismatch: "
                f"expected {declaration['sha256']}, got {actual_sha256}"
            )
        for immutable_input in declaration["immutable_inputs"]:
            if declared_by_name.get(immutable_input["name"]) != immutable_input["sha256"]:
                raise EvidenceError(
                    f"bootstrap artifact {declaration['name']!r} references an undeclared or "
                    "mismatched immutable input"
                )
            input_path = _resolve_evidence_path(
                Path(declared_paths[immutable_input["name"]]), base_directory=base_directory
            )
            try:
                compiler_target = _parse_json_document(
                    input_path.read_bytes(), document="canonical bootstrap compiler target"
                )
            except OSError as error:
                raise EvidenceError("canonical bootstrap compiler target is unreadable") from error
            if (
                not isinstance(compiler_target, dict)
                or not set(compiler_target)
                <= {
                    "target",
                    "compiler",
                    "compiler_version",
                    "flags",
                    "source_sha256",
                }
                or "target" not in compiler_target
            ):
                raise EvidenceError(
                    "canonical bootstrap compiler-target input has unsupported fields"
                )
            serialized_target = json.dumps(compiler_target, sort_keys=True).lower()
            if any(
                forbidden in serialized_target
                for forbidden in ("seed", "candidate", "policy", "optimizer", "rollout", "learned")
            ):
                raise EvidenceError(
                    "canonical bootstrap compiler-target contains run-specific or learned state"
                )
        expected_payload = (
            json.dumps(
                {
                    "contract": _BOOTSTRAP_CONTRACT,
                    "immutable_inputs": declaration["immutable_inputs"],
                    "protocol": _BOOTSTRAP_PROTOCOL,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        if payload != expected_payload:
            raise EvidenceError(
                f"bootstrap artifact {declaration['name']!r} does not match the canonical "
                "declarative bootstrap contract; opaque payloads are ineligible"
            )
        validated.append(
            {
                **declaration,
                "path": str(path),
                "eligibility_evidence": {
                    "contract": _BOOTSTRAP_CONTRACT,
                    "derivation": _BOOTSTRAP_PROTOCOL,
                    "opaque_payload_allowed": False,
                },
                "validated_before_cohort": True,
                "reverified_unchanged_after_cohort": False,
            }
        )
    return validated


def _reverify_bootstrap_files(artifacts: list[dict[str, Any]]) -> None:
    for artifact in artifacts:
        path = Path(artifact["path"])
        try:
            metadata = path.lstat()
            actual_sha256 = _sha256_bytes(path.read_bytes())
        except OSError as error:
            raise EvidenceError(
                f"bootstrap artifact {artifact['name']!r} changed during the cohort"
            ) from error
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o222
            or actual_sha256 != artifact["sha256"]
        ):
            raise EvidenceError(
                f"bootstrap artifact {artifact['name']!r} changed during the cohort"
            )
        artifact["reverified_unchanged_after_cohort"] = True


def _load_benchmark_continuation(
    path: Path,
    *,
    run_identity: str,
    protocol: dict[str, Any],
    code_provenance: dict[str, Any],
    declared_inputs: list[dict[str, Any]],
    wad_profile: dict[str, Any] | None,
    initial_evidence_entries: list[dict[str, str]],
    manifest_directory: Path,
) -> dict[str, Any]:
    try:
        report = _parse_json_document(path.read_bytes(), document="benchmark continuation report")
    except OSError as error:
        raise EvidenceError(f"cannot read benchmark continuation report: {path}") from error
    if not isinstance(report, dict):
        raise EvidenceError("benchmark continuation report must be a JSON object")
    exact_fields = {
        "schema_version": 1,
        "workflow": _WORKFLOW,
        "evidence_level": "development",
        "run_identity": run_identity,
        "code_provenance": code_provenance,
        "declared_inputs": declared_inputs,
        "benchmark_protocol": protocol,
        "wad_profile": wad_profile,
    }
    for field, expected in exact_fields.items():
        if report.get(field) != expected:
            raise EvidenceError(f"cannot continue benchmark with unlike {field.replace('_', ' ')}")
    if report.get("claim_eligible") is not False or report.get("authoritative") is not False:
        raise EvidenceError("benchmark continuation report changed development evidence status")
    evidence_index = _required_mapping(
        report.get("evidence_index"),
        "benchmark continuation report evidence_index",
    )
    entries = evidence_index.get("entries")
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise EvidenceError("benchmark continuation report evidence_index.entries must be an array")
    if evidence_index.get("sha256") != _canonical_sha256(
        entries, document="benchmark continuation"
    ):
        raise EvidenceError("benchmark continuation report evidence index SHA-256 mismatch")
    entries_by_name = {entry.get("name"): entry for entry in entries}
    if len(entries_by_name) != len(entries):
        raise EvidenceError("benchmark continuation report evidence index has duplicate names")
    for current in initial_evidence_entries:
        if entries_by_name.get(current["name"]) != current:
            raise EvidenceError(
                f"benchmark continuation evidence {current['name']!r} does not match current inputs"
            )
    generated = report.get("generated_artifacts")
    if not isinstance(generated, list):
        raise EvidenceError("benchmark continuation report generated_artifacts must be an array")
    paths_by_name = {
        item.get("name"): item.get("path")
        for item in generated
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and isinstance(item.get("path"), str)
    }
    initial_names = {entry["name"] for entry in initial_evidence_entries}
    for name, entry in entries_by_name.items():
        if name in initial_names:
            continue
        raw_path = paths_by_name.get(name)
        if raw_path is None:
            raise EvidenceError(f"benchmark continuation evidence {name!r} has no artifact path")
        artifact_path = _resolve_evidence_path(
            Path(raw_path),
            base_directory=manifest_directory,
        )
        if _fsync_file(
            artifact_path, field=f"benchmark continuation evidence {name!r}"
        ) != entry.get("sha256"):
            raise EvidenceError(f"benchmark continuation evidence {name!r} SHA-256 mismatch")
    attempts = report.get("attempts")
    if not isinstance(attempts, list):
        raise EvidenceError("benchmark continuation report attempts must be an array")
    expected_seeds = protocol["training_seeds"]
    if [attempt.get("seed") for attempt in attempts if isinstance(attempt, dict)] != expected_seeds:
        raise EvidenceError("benchmark continuation cannot replace or reorder training seeds")
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise EvidenceError("benchmark continuation attempt must be an object")
        expected_attempt_identity = _canonical_sha256(
            {"run_identity": run_identity, "seed": attempt["seed"]},
            document="benchmark attempt",
        )
        if attempt.get("attempt_identity") != expected_attempt_identity:
            raise EvidenceError("benchmark continuation attempt identity mismatch")
        if attempt.get("status") not in {
            "succeeded",
            "exhausted",
            "crashed",
            "evaluation_failed",
            "evidence_failed",
            "interrupted",
        }:
            raise EvidenceError("benchmark continuation attempt has invalid status")
        attempt_journal = attempt.get("attempt_journal")
        if not isinstance(attempt_journal, dict):
            raise EvidenceError("benchmark continuation attempt has no attempt journal")
        attempt_journal_path = _resolve_evidence_path(
            Path(attempt_journal.get("path", "")),
            base_directory=manifest_directory,
        )
        try:
            attempt_journal_bytes = attempt_journal_path.read_bytes()
            stored_attempt = _parse_json_document(
                attempt_journal_bytes,
                document="benchmark attempt journal",
            )
        except OSError as error:
            raise EvidenceError("benchmark attempt journal is missing") from error
        if not isinstance(stored_attempt, dict):
            raise EvidenceError("benchmark attempt journal must be an object")
        raw_generation = stored_attempt.get("generation")
        if type(raw_generation) is not int or raw_generation < 0:
            raise EvidenceError("benchmark attempt journal generation is invalid")
        latest_generation = (
            _next_append_only_generation(
                attempt_journal_path.parent, "attempt-state-*.json", "attempt-state-"
            )
            - 1
        )
        if raw_generation != latest_generation:
            raise EvidenceError("stale attempt journal generation cannot be replayed")
        previous_sha256 = stored_attempt.get("previous_journal_sha256")
        if raw_generation == 0:
            if previous_sha256 is not None:
                raise EvidenceError("initial attempt journal has an invalid predecessor")
        else:
            previous_path = attempt_journal_path.parent / f"attempt-state-{raw_generation - 1}.json"
            if (
                _fsync_file(previous_path, field="previous benchmark attempt journal")
                != previous_sha256
            ):
                raise EvidenceError("attempt journal generation continuity is broken")
        expected_stored = _attempt_journal_payload(
            attempt,
            run_identity=run_identity,
            generation=raw_generation,
            previous_journal_sha256=previous_sha256,
        )
        anchor = next(
            (
                item
                for item in protocol["elapsed_time_anchors"]
                if item["payload"]["seed"] == attempt["seed"]
            ),
            None,
        )
        if anchor is None:
            raise EvidenceError("benchmark attempt outcome has no external time authority")
        attestation = attempt_journal.get("authority_attestation")
        attestation_payload = _journal_attestation_payload(
            expected_stored,
            anchor=anchor,
            journal_sha256=_sha256_bytes(attempt_journal_bytes),
        )
        attestation_payload["reusable_elapsed_seconds"] = attempt.get("reusable_elapsed_seconds")
        verified_attestation = _verify_generation_attestation(
            attestation,
            anchor=anchor,
            expected_payload=attestation_payload,
            recover_same_head=True,
        )
        recovered_payload = verified_attestation["payload"]
        if recovered_payload != attestation_payload:
            recovered_elapsed = recovered_payload.get("reusable_elapsed_seconds")
            if (
                type(recovered_elapsed) not in (int, float)
                or not math.isfinite(float(recovered_elapsed))
                or float(recovered_elapsed) < float(attempt["reusable_elapsed_seconds"])
            ):
                raise EvidenceError("recovered authority attestation has invalid elapsed time")
            attempt["reusable_elapsed_seconds"] = float(recovered_elapsed)
            recovery = attempt.get("recovery")
            if isinstance(recovery, dict):
                recovery["accumulated_reusable_elapsed_seconds"] = float(recovered_elapsed)
            attempt_journal["authority_attestation"] = verified_attestation
            attempt["_recovered_terminal_timing"] = {
                "prior_elapsed": float(recovered_elapsed),
                "anchor": anchor,
                "journal_payload": expected_stored,
                "journal_sha256": _sha256_bytes(attempt_journal_bytes),
            }
        if stored_attempt != expected_stored or attempt_journal.get("sha256") != _sha256_bytes(
            attempt_journal_bytes
        ):
            raise EvidenceError("attempt journal does not match completed unit")
        if attempt["status"] == "interrupted":
            recovery = attempt.get("recovery")
            if not isinstance(recovery, dict):
                raise EvidenceError("interrupted benchmark attempt has no recovery checkpoint")
            if (
                recovery.get("run_identity") != run_identity
                or recovery.get("attempt_identity") != expected_attempt_identity
                or recovery.get("restorable_state") != _RESTORABLE_STATE
                or recovery.get("accumulated_reusable_elapsed_seconds")
                != attempt.get("reusable_elapsed_seconds")
            ):
                raise EvidenceError(
                    "interrupted benchmark recovery identity or state is incomplete"
                )
            journal = attempt.get("recovery_journal")
            if not isinstance(journal, dict):
                raise EvidenceError("interrupted benchmark attempt has no recovery journal")
            journal_path = _resolve_evidence_path(
                Path(journal.get("path", "")),
                base_directory=manifest_directory,
            )
            try:
                journal_payload = _parse_json_document(
                    journal_path.read_bytes(),
                    document="recovery journal",
                )
            except OSError as error:
                raise EvidenceError("recovery journal is missing") from error
            recovery_journal_elapsed = (
                journal_payload.get("accumulated_reusable_elapsed_seconds")
                if isinstance(journal_payload, dict)
                else None
            )
            if (
                type(recovery_journal_elapsed) not in (int, float)
                or float(recovery_journal_elapsed) < 0.0
                or float(recovery_journal_elapsed) > float(attempt["reusable_elapsed_seconds"])
            ):
                raise EvidenceError("recovery journal has invalid pre-terminal elapsed time")
            expected_journal = {
                "schema_version": 1,
                "run_identity": run_identity,
                "attempt_identity": expected_attempt_identity,
                "seed": attempt["seed"],
                "checkpoint": recovery["checkpoint"],
                "checkpoint_sha256": recovery["checkpoint_sha256"],
                "progress_step": recovery["progress_step"],
                "restorable_state": _RESTORABLE_STATE,
                "accumulated_reusable_elapsed_seconds": recovery_journal_elapsed,
            }
            if journal_payload != expected_journal or journal.get("sha256") != _sha256_bytes(
                journal_path.read_bytes()
            ):
                raise EvidenceError("recovery journal does not match interrupted attempt")
    return report


def _validate_benchmark(manifest: dict[str, Any]) -> dict[str, Any]:
    _validate_schema_version(manifest.get("schema_version"), document="manifest")
    if manifest.get("workflow") != _WORKFLOW:
        raise EvidenceError(f"this command path requires workflow {_WORKFLOW}")
    if manifest.get("evidence_level") != "development":
        raise EvidenceError("development training benchmark requires development evidence")
    if type(manifest.get("fixture")) is not bool:
        raise EvidenceError("fixture is required and must be a boolean")
    code_provenance = _validate_code_provenance(manifest.get("code_provenance"))
    benchmark = _required_mapping(manifest.get("benchmark"), "benchmark")
    training_seeds = _seed_list(
        benchmark.get("training_seeds", [_DEFAULT_TRAINING_SEED]),
        "benchmark.training_seeds",
    )
    failure_budget = _positive_integer(
        benchmark.get("failure_budget_steps"),
        "benchmark.failure_budget_steps",
    )
    checkpoint_steps = benchmark.get("checkpoint_steps")
    if not isinstance(checkpoint_steps, list) or not checkpoint_steps:
        raise EvidenceError("benchmark.checkpoint_steps must be a non-empty array")
    validated_steps = [
        _positive_integer(step, f"benchmark.checkpoint_steps[{index}]")
        for index, step in enumerate(checkpoint_steps)
    ]
    if validated_steps != sorted(set(validated_steps)):
        raise EvidenceError("benchmark.checkpoint_steps must be unique and strictly increasing")
    if validated_steps[-1] != failure_budget:
        raise EvidenceError("benchmark.checkpoint_steps must end at benchmark.failure_budget_steps")
    evaluation_seeds = _seed_list(
        benchmark.get("evaluation_episode_seeds"),
        "benchmark.evaluation_episode_seeds",
        exact_count=_EVALUATION_EPISODES,
    )
    evaluation_action_seed = benchmark.get("evaluation_action_seed", _DEFAULT_TRAINING_SEED)
    if type(evaluation_action_seed) is not int or not 0 <= evaluation_action_seed <= _UINT32_MAX:
        raise EvidenceError(
            f"benchmark.evaluation_action_seed must be an integer in [0, {_UINT32_MAX}]"
        )
    trainer = _validate_trainer(benchmark.get("trainer"))
    artifacts_directory = benchmark.get("artifacts_directory")
    if not isinstance(artifacts_directory, str) or not artifacts_directory.strip():
        raise EvidenceError("benchmark.artifacts_directory must be a non-whitespace path")
    certificate = _validate_certificate(benchmark.get("parity_certificate"))
    bootstrap_artifacts = _validate_bootstrap_artifacts(
        benchmark.get("bootstrap_artifacts"), fixture=manifest["fixture"]
    )
    elapsed_time_anchors = _validate_elapsed_time_anchors(
        benchmark.get("elapsed_time_anchors"),
        training_seeds=training_seeds,
        fixture=manifest["fixture"],
    )
    cuda_residency_acceptance = benchmark.get("cuda_residency_acceptance", False)
    if type(cuda_residency_acceptance) is not bool:
        raise EvidenceError("benchmark.cuda_residency_acceptance must be a boolean")
    return {
        "code_provenance": code_provenance,
        "training_seeds": training_seeds,
        "failure_budget_steps": failure_budget,
        "checkpoint_steps": validated_steps,
        "evaluation_episode_seeds": evaluation_seeds,
        "evaluation_action_seed": evaluation_action_seed,
        "trainer": trainer,
        "artifacts_directory": artifacts_directory,
        "parity_certificate": certificate,
        "bootstrap_artifacts": bootstrap_artifacts,
        "elapsed_time_anchors": elapsed_time_anchors,
        "cuda_residency_acceptance": cuda_residency_acceptance,
    }


def _read_jsonl(path: Path, *, phase: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_bytes().splitlines()
    except OSError as error:
        raise EvidenceError(f"{phase} process did not write metrics JSONL: {path}") from error
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        parsed = _parse_json_document(line, document=f"{phase} metrics line {index}")
        if not isinstance(parsed, dict):
            raise EvidenceError(f"{phase} metrics line {index} must be a JSON object")
        records.append(parsed)
    if not records:
        raise EvidenceError(f"{phase} process wrote no metrics records")
    return records


def _record(records: list[dict[str, Any]], record_type: str, *, phase: str) -> dict[str, Any]:
    matches = [record for record in records if record.get("type") == record_type]
    if len(matches) != 1:
        raise EvidenceError(
            f"{phase} process must emit exactly one {record_type!r} record, got {len(matches)}"
        )
    return matches[0]


def _resolved_record_path(value: object, *, field: str, base_directory: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{field} must be a non-whitespace path")
    return _resolve_evidence_path(Path(value), base_directory=base_directory)


def _validate_training_records(
    records: list[dict[str, Any]],
    *,
    checkpoint: Path,
    requested_step: int,
    previous_checkpoint: dict[str, Any] | None,
    manifest_directory: Path,
    wad_profile: dict[str, Any] | None,
    run_identity: str,
    attempt_identity: str,
    interrupted: bool = False,
    cuda_residency_acceptance: bool,
    fixture: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    config = _record(records, "config", phase="training")
    if config.get("contract") != _TRAINER_CONTRACT or config.get("operation") != "train":
        raise EvidenceError("training process did not use the standalone GPU-resident trainer")
    expected_binding = {
        "run_identity": run_identity,
        "attempt_identity": attempt_identity,
    }
    if config.get("evidence_binding") != expected_binding:
        raise EvidenceError("training process did not bind the exact run and attempt identity")
    configured_acceptance = config.get("cuda_residency_acceptance")
    if cuda_residency_acceptance:
        if (
            not isinstance(configured_acceptance, dict)
            or configured_acceptance.get("contract") != CUDA_RESIDENCY_CONTRACT
            or configured_acceptance.get("enabled") is not True
        ):
            raise EvidenceError("training process did not enable CUDA residency acceptance")
    elif configured_acceptance is not None:
        raise EvidenceError("training process enabled unrequested CUDA residency acceptance")
    initialization = config.get("initialization")
    if not isinstance(initialization, dict) or initialization.get("mode") != "random":
        raise EvidenceError("training process must declare random policy initialization")
    if initialization.get("checkpoint") is not None:
        raise EvidenceError("training process initialized policy from learned state")
    state_initialization = config.get("state_initialization")
    expected_state_initialization = (
        {"policy_state": "fresh_random", "optimizer_state": "fresh"}
        if previous_checkpoint is None
        else {"policy_state": "resumed", "optimizer_state": "resumed"}
    )
    if state_initialization != expected_state_initialization:
        raise EvidenceError(
            "training process did not declare the required fresh or continuous policy and "
            "optimizer state"
        )
    resumed = [record for record in records if record.get("event") == "resumed"]
    if previous_checkpoint is None and resumed:
        raise EvidenceError("cold-start training process unexpectedly resumed learned state")
    if previous_checkpoint is not None:
        if len(resumed) != 1:
            raise EvidenceError("continued training process must emit exactly one resumed event")
        resumed_path = _resolved_record_path(
            resumed[0].get("checkpoint"),
            field="training resumed checkpoint",
            base_directory=manifest_directory,
        )
        if resumed_path != previous_checkpoint["path"]:
            raise EvidenceError("continued training process resumed the wrong checkpoint")
        if resumed[0].get("train/global_step") != previous_checkpoint["progress_step"]:
            raise EvidenceError("continued training process resumed the wrong progress step")
        if resumed[0].get("restored_state") != _RESTORABLE_STATE:
            raise EvidenceError(
                "continued training process did not restore policy, optimizer, RNG, and progress"
            )
        if resumed[0].get("evidence_binding") != expected_binding:
            raise EvidenceError("continued training process resumed unlike run or attempt identity")
    summary = _record(records, "summary", phase="training")
    expected_status = "interrupted" if interrupted else "completed"
    if summary.get("status") != expected_status:
        raise EvidenceError(f"training process reported status {summary.get('status')!r}")
    step = summary.get("train/global_step")
    if (
        type(step) is not int
        or (interrupted and not 0 < step < requested_step)
        or (not interrupted and step != requested_step)
    ):
        boundary = (
            "a recoverable intermediate step" if interrupted else "the predeclared checkpoint step"
        )
        raise EvidenceError(f"training process stopped outside {boundary}")
    if config.get("requested_timesteps") != requested_step:
        raise EvidenceError("training config did not bind the predeclared checkpoint step")
    if config.get("execution_timesteps") != requested_step:
        raise EvidenceError("training config would execute outside the predeclared checkpoint step")
    if summary.get("requested_timesteps") != requested_step:
        raise EvidenceError("training summary did not bind the predeclared checkpoint step")
    if summary.get("execution_timesteps") != requested_step:
        raise EvidenceError("training summary executed outside the predeclared checkpoint step")
    recorded_checkpoint = _resolved_record_path(
        summary.get("checkpoint"),
        field="training summary checkpoint",
        base_directory=manifest_directory,
    )
    if recorded_checkpoint != checkpoint:
        raise EvidenceError("training process reported a different checkpoint path")
    _validate_runtime_assets(summary, phase="training", wad_profile=wad_profile)
    acceptance_records = [
        record for record in records if record.get("type") == "cuda_residency_acceptance"
    ]
    resumed_record = None if previous_checkpoint is None else resumed[0]
    if not cuda_residency_acceptance:
        if acceptance_records:
            raise EvidenceError(
                "training process emitted unrequested CUDA residency acceptance evidence"
            )
        return summary, resumed_record, None
    if len(acceptance_records) != 1:
        raise EvidenceError(
            "training process must emit exactly one 'cuda_residency_acceptance' record, "
            f"got {len(acceptance_records)}"
        )
    acceptance = acceptance_records[0]
    _validate_cuda_residency_record(acceptance, fixture=fixture)
    return summary, resumed_record, acceptance


def _validate_cuda_residency_record(record: dict[str, Any], *, fixture: bool) -> None:
    if record.get("contract") != CUDA_RESIDENCY_CONTRACT or record.get("status") != "passed":
        raise EvidenceError("CUDA residency acceptance record did not pass the required contract")
    expected_kind = "fixture_contract" if fixture else "cuda_hardware"
    if record.get("evidence_kind") != expected_kind:
        raise EvidenceError(
            f"CUDA residency acceptance evidence_kind must be {expected_kind!r}"
        )
    if record.get("checked_categories") != list(CUDA_RESIDENCY_CATEGORIES):
        raise EvidenceError("CUDA residency acceptance did not check every required category")
    devices = record.get("devices")
    if not isinstance(devices, dict) or set(devices) != set(CUDA_RESIDENCY_CATEGORIES):
        raise EvidenceError("CUDA residency acceptance device evidence is incomplete")
    concrete_devices: set[str] = set()
    for category in CUDA_RESIDENCY_CATEGORIES:
        values = devices[category]
        if not isinstance(values, list) or not values:
            raise EvidenceError(f"CUDA residency acceptance {category} devices must be non-empty")
        for value in values:
            if not isinstance(value, str) or not value.startswith("cuda:"):
                raise EvidenceError(
                    f"CUDA residency acceptance {category} used a non-CUDA device"
                )
            concrete_devices.add(value)
    if len(concrete_devices) != 1:
        raise EvidenceError("CUDA residency acceptance used inconsistent CUDA devices")
    guard = record.get("host_transition_guard")
    if (
        not isinstance(guard, dict)
        or guard.get("status") != "passed"
        or type(guard.get("guarded_scopes")) is not int
        or guard["guarded_scopes"] <= 0
        or guard.get("detected_transfers") != 0
    ):
        raise EvidenceError("CUDA residency acceptance host-transition guard did not pass")
    for field in ("checked_rollouts", "checked_steps"):
        if type(record.get(field)) is not int or record[field] <= 0:
            raise EvidenceError(f"CUDA residency acceptance {field} must be positive")
    workload = record.get("workload")
    if (
        not isinstance(workload, dict)
        or workload.get("trainer_contract") != _TRAINER_CONTRACT
    ):
        raise EvidenceError("CUDA residency acceptance workload did not use the benchmark trainer")
    if (
        workload.get("checked_rollouts") != record["checked_rollouts"]
        or workload.get("checked_transitions") != record["checked_steps"]
    ):
        raise EvidenceError("CUDA residency acceptance workload counts are inconsistent")
    hardware = record.get("hardware")
    if not isinstance(hardware, dict):
        raise EvidenceError("CUDA residency acceptance hardware is required")
    for field in ("gpu_model", "device", "compute_capability"):
        if not isinstance(hardware.get(field), str) or not hardware[field]:
            raise EvidenceError(f"CUDA residency acceptance hardware.{field} is required")
    if hardware["device"] != next(iter(concrete_devices)):
        raise EvidenceError(
            "CUDA residency acceptance hardware device does not match checked tensor devices"
        )
    minimum_memory = 0 if fixture else 1
    if (
        type(hardware.get("total_memory_bytes")) is not int
        or hardware["total_memory_bytes"] < minimum_memory
    ):
        raise EvidenceError("CUDA residency acceptance hardware.total_memory_bytes is invalid")
    software = record.get("software")
    if not isinstance(software, dict):
        raise EvidenceError("CUDA residency acceptance software is required")
    required_software = {"python", "gradoom", "torch", "cuda", "cudnn", "numpy"}
    if set(software) != required_software or any(
        not isinstance(software[field], str) or not software[field]
        for field in required_software
    ):
        raise EvidenceError("CUDA residency acceptance software versions are incomplete")


def _validate_runtime_assets(
    record: dict[str, Any],
    *,
    phase: str,
    wad_profile: dict[str, Any] | None,
) -> None:
    if wad_profile is None:
        return
    binding = wad_profile["binding_identity"]
    providers = binding["providers"]
    gradoom_provider = next(provider for provider in providers if provider["id"] == "gradoom")
    if record.get("iwad_sha256") != gradoom_provider["iwad_sha256"]:
        raise EvidenceError(f"{phase} process used an IWAD outside the declared WAD profile")
    if record.get("scenario_sha256") != gradoom_provider["pwad_sha256"]:
        raise EvidenceError(f"{phase} process used a PWAD outside the declared WAD profile")


def _validate_evaluation_records(
    records: list[dict[str, Any]],
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    episode_seeds: list[int],
    evaluation_action_seed: int,
    manifest_directory: Path,
    wad_profile: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], float, float | None]:
    config = _record(records, "config", phase="evaluation")
    if config.get("contract") != _TRAINER_CONTRACT or config.get("operation") != "evaluate":
        raise EvidenceError("evaluation process did not use the standalone trainer evaluation path")
    evaluation_config = config.get("evaluation")
    if not isinstance(evaluation_config, dict):
        raise EvidenceError("evaluation config is missing")
    if evaluation_config.get("episodes") != _EVALUATION_EPISODES:
        raise EvidenceError("evaluation process did not declare exactly 100 episodes")
    if evaluation_config.get("stochastic_actions") is not True:
        raise EvidenceError("evaluation process did not declare stochastic actions")
    if evaluation_config.get("seed") != evaluation_action_seed:
        raise EvidenceError("evaluation process did not use the predeclared stochastic action seed")
    if evaluation_config.get("kills_signal") != "player_killcount":
        raise EvidenceError("evaluation process did not declare player_killcount quality")
    evaluation = _record(records, "evaluation", phase="evaluation")
    if evaluation.get("status") != "completed":
        raise EvidenceError(f"evaluation process reported status {evaluation.get('status')!r}")
    if evaluation.get("checkpoint_sha256") != checkpoint_sha256:
        raise EvidenceError("evaluation checkpoint SHA-256 does not match the durable checkpoint")
    recorded_checkpoint = _resolved_record_path(
        evaluation.get("checkpoint"),
        field="evaluation checkpoint",
        base_directory=manifest_directory,
    )
    if recorded_checkpoint != checkpoint:
        raise EvidenceError("evaluation process reported a different checkpoint path")
    _validate_runtime_assets(evaluation, phase="evaluation", wad_profile=wad_profile)
    if evaluation.get("deterministic_actions") is not False:
        raise EvidenceError("evaluation process did not execute stochastic policy actions")
    if evaluation.get("evaluation/episode/count") != _EVALUATION_EPISODES:
        raise EvidenceError("evaluation process did not complete exactly 100 episodes")
    if evaluation.get("evaluation/kills/signal") != "player_killcount":
        raise EvidenceError("evaluation result did not use player_killcount quality")
    raw_episodes = evaluation.get("episodes")
    if not isinstance(raw_episodes, list) or len(raw_episodes) != _EVALUATION_EPISODES:
        raise EvidenceError("evaluation episodes must contain exactly 100 outcomes")
    episodes: list[dict[str, Any]] = []
    player_killcounts: list[float] = []
    compatibility_killcounts: list[float] = []
    for index, raw_episode in enumerate(raw_episodes):
        if not isinstance(raw_episode, dict):
            raise EvidenceError(f"evaluation episodes[{index}] must be an object")
        if raw_episode.get("index") != index:
            raise EvidenceError("evaluation episode indices do not match their declared order")
        if raw_episode.get("game_seed") != episode_seeds[index]:
            raise EvidenceError("evaluation episode seeds do not match the predeclared seed grid")
        player_value = raw_episode.get("player_killcount")
        if type(player_value) not in (int, float) or not math.isfinite(float(player_value)):
            raise EvidenceError(f"evaluation episodes[{index}].player_killcount must be finite")
        player_killcounts.append(float(player_value))
        compatibility_value = raw_episode.get(
            "compatibility_killcount",
            raw_episode.get("killcount", raw_episode.get("vizdoom_killcount")),
        )
        if compatibility_value is not None:
            if type(compatibility_value) not in (int, float) or not math.isfinite(
                float(compatibility_value)
            ):
                raise EvidenceError(
                    f"evaluation episodes[{index}].compatibility_killcount must be finite"
                )
            compatibility_killcounts.append(float(compatibility_value))
        episodes.append(raw_episode)
    mean_player = statistics.fmean(player_killcounts)
    mean_compatibility = (
        statistics.fmean(compatibility_killcounts)
        if len(compatibility_killcounts) == _EVALUATION_EPISODES
        else None
    )
    return evaluation, episodes, mean_player, mean_compatibility


def _run_process(
    command: list[str],
    *,
    cwd: Path,
    execution_binding: _TrainerFileIdentityMarkers,
    heartbeat: Callable[[], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    archive = execution_binding.sealed_archive
    if archive is None or archive.closed:
        raise EvidenceError("sealed trainer execution binding is unavailable")
    if len(command) < 2:
        raise EvidenceError("sealed trainer execution binding has no Python entry point")
    code_root = execution_binding.code_root
    entry_relative = execution_binding.entry_relative
    executable_argv0 = execution_binding.executable_argv0
    if code_root is None or entry_relative is None or executable_argv0 is None:
        raise EvidenceError("sealed trainer execution binding has incomplete source metadata")
    archive_descriptor = archive.fileno()
    launched_command = [
        executable_argv0,
        "-P",
        "-s",
        "-c",
        _SEALED_PYTHON_BOOTSTRAP,
        f"/proc/self/fd/{archive_descriptor}",
        str(code_root),
        str(entry_relative),
        *command[2:],
    ]
    try:
        process = subprocess.Popen(
            launched_command,
            executable=command[0],
            cwd=cwd,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            pass_fds=(archive_descriptor,),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        return subprocess.CompletedProcess(
            command,
            127,
            stdout="",
            stderr=f"cannot execute benchmark process {command[0]!r}: {error}",
        )
    previous_handlers: dict[int, Any] = {}
    forwarded_signal: int | None = None

    def persist_before_termination(signum: int, _frame: Any) -> None:
        nonlocal forwarded_signal
        if forwarded_signal is not None:
            return
        forwarded_signal = signum
        if heartbeat is not None:
            heartbeat()
        process.send_signal(signum)

    try:
        if heartbeat is not None:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.signal(signum, persist_before_termination)
        stdout, stderr = process.communicate()
        if heartbeat is not None:
            heartbeat()
        completed = subprocess.CompletedProcess(
            command,
            process.returncode,
            stdout=stdout,
            stderr=stderr,
        )
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if forwarded_signal is not None:
        signal.signal(forwarded_signal, signal.SIG_DFL)
        os.kill(os.getpid(), forwarded_signal)
        raise AssertionError("termination signal did not terminate evidence parent")
    if completed.returncode == _SEALED_EXECUTION_EXIT:
        raise EvidenceError("sealed trainer execution binding rejected a code-root mutation")
    return completed


def _fsync_file(path: Path, *, field: str) -> str:
    try:
        with path.open("rb") as stream:
            payload = stream.read()
            os.fsync(stream.fileno())
    except OSError as error:
        raise EvidenceError(f"{field} is not a durable readable file: {path}") from error
    try:
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as error:
        raise EvidenceError(f"{field} directory cannot be made durable: {path.parent}") from error
    return _sha256_bytes(payload)


def _write_durable_json(path: Path, payload: dict[str, Any], *, field: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _fsync_file(path, field=field)


def _load_durable_attempt_seals(
    run_directory: Path,
    *,
    run_identity: str,
    protocol: dict[str, Any],
    initial_evidence_entries: list[dict[str, str]],
    manifest_directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    """Recover authority-signed attempts whose public report was not yet durable."""
    attempts: list[dict[str, Any]] = []
    evidence_entries = [dict(entry) for entry in initial_evidence_entries]
    generated_artifacts: list[dict[str, str]] = []
    generated_paths: dict[str, str] = {}
    missing_seal_seen = False
    anchors_by_seed = {
        anchor["payload"]["seed"]: anchor for anchor in protocol["elapsed_time_anchors"]
    }
    for seed in protocol["training_seeds"]:
        attempt_directory = run_directory / f"seed-{seed}"
        seal_paths = sorted(attempt_directory.glob("attempt-seal-*.json"))
        if not seal_paths:
            missing_seal_seen = True
            continue
        if missing_seal_seen:
            raise EvidenceError("durable attempt seals cannot skip or replace a training seed")
        generations: list[int] = []
        for seal_path in seal_paths:
            raw_generation = seal_path.stem.removeprefix("attempt-seal-")
            if not raw_generation.isdigit():
                raise EvidenceError("durable attempt seal has an invalid generation")
            generations.append(int(raw_generation))
        if generations != list(range(generations[-1] + 1)):
            raise EvidenceError("durable attempt seal generation continuity is broken")
        seal_path = seal_paths[-1]
        generation = generations[-1]
        try:
            seal_bytes = seal_path.read_bytes()
            seal = _parse_json_document(seal_bytes, document="durable benchmark attempt seal")
        except OSError as error:
            raise EvidenceError("durable benchmark attempt seal is unreadable") from error
        if not isinstance(seal, dict) or set(seal) != {
            "schema_version",
            "run_identity",
            "continuation_identity",
            "attempt",
            "evidence_entries",
            "generated_artifacts",
        }:
            raise EvidenceError("durable benchmark attempt seal has an unsupported contract")
        if (
            seal.get("schema_version") != 1
            or seal.get("run_identity") != run_identity
            or seal.get("continuation_identity") != protocol["continuation_identity"]
        ):
            raise EvidenceError("durable benchmark attempt seal has unlike provenance")
        attempt = seal.get("attempt")
        if not isinstance(attempt, dict) or attempt.get("seed") != seed:
            raise EvidenceError("durable benchmark attempt seal has an invalid seed")
        expected_attempt_identity = _canonical_sha256(
            {"run_identity": run_identity, "seed": seed},
            document="benchmark attempt",
        )
        if attempt.get("attempt_identity") != expected_attempt_identity:
            raise EvidenceError("durable benchmark attempt seal has unlike attempt identity")
        journal = attempt.get("attempt_journal")
        if not isinstance(journal, dict):
            raise EvidenceError("durable benchmark attempt seal has no attempt journal")
        journal_path = _resolve_evidence_path(
            Path(journal.get("path", "")), base_directory=manifest_directory
        )
        try:
            journal_bytes = journal_path.read_bytes()
            journal_payload = _parse_json_document(
                journal_bytes, document="sealed benchmark attempt journal"
            )
        except OSError as error:
            raise EvidenceError("sealed benchmark attempt journal is unreadable") from error
        if not isinstance(journal_payload, dict) or journal_payload.get("generation") != generation:
            raise EvidenceError("durable benchmark attempt seal generation is invalid")
        if (
            _next_append_only_generation(
                attempt_directory, "attempt-state-*.json", "attempt-state-"
            )
            != generation + 1
        ):
            raise EvidenceError("durable benchmark attempt seal is stale")
        previous_journal_sha256 = journal_payload.get("previous_journal_sha256")
        expected_journal = _attempt_journal_payload(
            attempt,
            run_identity=run_identity,
            generation=generation,
            previous_journal_sha256=previous_journal_sha256,
        )
        journal_sha256 = _sha256_bytes(journal_bytes)
        if journal_payload != expected_journal or journal.get("sha256") != journal_sha256:
            raise EvidenceError("durable benchmark attempt seal does not match its journal")
        anchor = anchors_by_seed.get(seed)
        if anchor is None:
            raise EvidenceError("durable benchmark attempt seal has no elapsed-time anchor")
        expected_attestation = _journal_attestation_payload(
            expected_journal,
            anchor=anchor,
            journal_sha256=journal_sha256,
        )
        expected_attestation["reusable_elapsed_seconds"] = attempt.get(
            "reusable_elapsed_seconds"
        )
        verified_attestation = _verify_generation_attestation(
            journal.get("authority_attestation"),
            anchor=anchor,
            expected_payload=expected_attestation,
            recover_same_head=True,
        )
        recovered_elapsed = float(verified_attestation["payload"]["reusable_elapsed_seconds"])
        if recovered_elapsed < float(attempt["reusable_elapsed_seconds"]):
            raise EvidenceError("durable benchmark attempt seal elapsed time moved backwards")
        attempt["reusable_elapsed_seconds"] = recovered_elapsed
        recovery = attempt.get("recovery")
        if isinstance(recovery, dict):
            recovery["accumulated_reusable_elapsed_seconds"] = recovered_elapsed
        journal["authority_attestation"] = verified_attestation
        attempt["_recovered_terminal_timing"] = {
            "prior_elapsed": recovered_elapsed,
            "anchor": anchor,
            "journal_payload": expected_journal,
            "journal_sha256": journal_sha256,
        }
        stored_entries = seal.get("evidence_entries")
        stored_generated = seal.get("generated_artifacts")
        if (
            not isinstance(stored_entries, list)
            or not all(isinstance(entry, dict) for entry in stored_entries)
            or not isinstance(stored_generated, list)
            or not all(isinstance(item, dict) for item in stored_generated)
            or stored_entries[: len(evidence_entries)] != evidence_entries
        ):
            raise EvidenceError("durable benchmark attempt seal changed prior evidence")
        for artifact in stored_generated:
            name = artifact.get("name")
            path = artifact.get("path")
            if not isinstance(name, str) or not isinstance(path, str):
                raise EvidenceError("durable benchmark attempt seal has invalid artifact paths")
            if name in generated_paths and generated_paths[name] != path:
                raise EvidenceError("durable benchmark attempt seal replaced an artifact")
            generated_paths[name] = path
        for entry in stored_entries[len(evidence_entries) :]:
            name = entry.get("name")
            path = generated_paths.get(name)
            if not isinstance(name, str) or path is None:
                raise EvidenceError("durable benchmark attempt seal has unbound evidence")
            artifact_path = _resolve_evidence_path(
                Path(path), base_directory=manifest_directory
            )
            if _fsync_file(artifact_path, field=f"sealed attempt evidence {name!r}") != entry.get(
                "sha256"
            ):
                raise EvidenceError("durable benchmark attempt seal evidence changed")
        evidence_entries = [dict(entry) for entry in stored_entries]
        attempt_generated_artifacts = [dict(item) for item in stored_generated]
        generated_artifacts = list(
            {
                (item["name"], item["path"]): item
                for item in [*generated_artifacts, *attempt_generated_artifacts]
            }.values()
        )
        seal_name = f"seed-{seed}-attempt-seal-{generation}"
        seal_sha256 = _sha256_bytes(seal_bytes)
        evidence_entries.append({"name": seal_name, "sha256": seal_sha256})
        seal_artifact = {"name": seal_name, "path": str(seal_path)}
        generated_artifacts.append(seal_artifact)
        generated_paths[seal_name] = str(seal_path)
        attempt["sealed_attempt"] = {"path": str(seal_path), "sha256": seal_sha256}
        attempt["generated_artifacts"] = [*attempt_generated_artifacts, seal_artifact]
        attempts.append(attempt)
    return attempts, evidence_entries, generated_artifacts


def _write_seed_file(path: Path, seeds: list[int]) -> str:
    path.write_text(json.dumps(seeds, separators=(",", ":")) + "\n", encoding="utf-8")
    return _fsync_file(path, field="evaluation seed file")


def _next_append_only_generation(directory: Path, pattern: str, prefix: str) -> int:
    generations: list[int] = []
    for path in directory.glob(pattern):
        suffix = path.stem.removeprefix(prefix)
        if suffix.isdigit():
            generations.append(int(suffix))
    return max(generations, default=-1) + 1


def _failure(
    *,
    seed: int,
    phase: str,
    checkpoint_step: int,
    process: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    return {
        "seed": seed,
        "phase": phase,
        "checkpoint_step": checkpoint_step,
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def _execute_evaluation(
    *,
    base_command: list[str],
    protocol: dict[str, Any],
    seed: int,
    checkpoint: Path,
    checkpoint_sha256: str,
    checkpoint_step: int,
    seed_file: Path,
    evaluation_metrics: Path,
    manifest_directory: Path,
    wad_profile: dict[str, Any] | None,
    execution_binding: _TrainerFileIdentityMarkers,
    heartbeat: Callable[[], None] | None,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None, str | None]:
    evaluation_command = [
        *base_command,
        "--evaluate-checkpoint",
        str(checkpoint),
        "--evaluation-episodes",
        str(_EVALUATION_EPISODES),
        "--evaluation-seeds-file",
        str(seed_file),
        "--evaluation-seed",
        str(protocol["evaluation_action_seed"]),
        "--evaluation-stochastic",
        "--metrics-jsonl",
        str(evaluation_metrics),
    ]
    evaluation_process = _run_process(
        evaluation_command,
        cwd=manifest_directory,
        execution_binding=execution_binding,
        heartbeat=heartbeat,
    )
    if evaluation_process.returncode == 130:
        return "interrupted", None, None, None
    if evaluation_process.returncode != 0:
        return (
            "evaluation_failed",
            None,
            _failure(
                seed=seed,
                phase="evaluation",
                checkpoint_step=checkpoint_step,
                process=evaluation_process,
            ),
            None,
        )
    try:
        evaluation_records = _read_jsonl(evaluation_metrics, phase="evaluation")
        evaluation, episodes, mean_player, mean_compatibility = _validate_evaluation_records(
            evaluation_records,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            episode_seeds=protocol["evaluation_episode_seeds"],
            evaluation_action_seed=protocol["evaluation_action_seed"],
            manifest_directory=manifest_directory,
            wad_profile=wad_profile,
        )
        evaluation_metrics_sha256 = _fsync_file(
            evaluation_metrics,
            field="evaluation metrics",
        )
    except EvidenceError as error:
        return (
            "evidence_failed",
            None,
            {
                "seed": seed,
                "phase": "evaluation_evidence",
                "checkpoint_step": checkpoint_step,
                "message": str(error),
            },
            None,
        )
    passed = mean_player >= _QUALITY_THRESHOLD
    outcome = {
        "checkpoint_step": checkpoint_step,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "mean_player_killcount": mean_player,
        "mean_killcount": mean_compatibility,
        "passed": passed,
        "evaluation": evaluation,
        "episodes": episodes,
    }
    return ("succeeded" if passed else "exhausted"), outcome, None, evaluation_metrics_sha256


def _load_live_interrupted_attempt(
    attempt_directory: Path,
    *,
    seed: int,
    run_identity: str,
    attempt_identity: str,
    protocol: dict[str, Any],
    manifest_directory: Path,
    evidence_entries: list[dict[str, str]],
    wad_profile: dict[str, Any] | None,
    elapsed_time_anchor: dict[str, Any] | None,
) -> dict[str, Any] | None:
    journals = sorted(
        attempt_directory.glob("attempt-live-*.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for journal_path in journals:
        try:
            journal_bytes = journal_path.read_bytes()
            journal = _parse_json_document(journal_bytes, document="live benchmark attempt journal")
        except OSError:
            continue
        if not isinstance(journal, dict) or journal.get("status") != "running":
            continue
        journal_payload_sha256 = journal.pop("payload_sha256", None)
        if journal_payload_sha256 != _canonical_sha256(
            journal,
            document="live benchmark attempt journal",
        ):
            raise EvidenceError(f"seed {seed} live attempt journal checksum mismatch")
        expected_identity = protocol["continuation_identity"]
        if (
            journal.get("schema_version") != 1
            or journal.get("run_identity") != run_identity
            or journal.get("attempt_identity") != attempt_identity
            or journal.get("seed") != seed
            or journal.get("continuation_identity") != expected_identity
        ):
            raise EvidenceError(f"seed {seed} live attempt journal has unlike identity")
        phase = journal.get("phase", "training")
        if phase not in {"training", "evaluation"}:
            raise EvidenceError(f"seed {seed} live attempt journal has an invalid phase")
        checkpoint = _resolve_evidence_path(
            Path(journal.get("checkpoint", "")), base_directory=manifest_directory
        )
        training_metrics = _resolve_evidence_path(
            Path(journal.get("training_metrics", "")), base_directory=manifest_directory
        )
        if not checkpoint.is_file() or not training_metrics.is_file():
            raise EvidenceError(
                f"seed {seed} interrupted before producing a recoverable checkpoint"
            )
        previous_checkpoint = journal.get("previous_checkpoint")
        if previous_checkpoint is not None:
            previous_checkpoint = _required_mapping(
                previous_checkpoint, "live benchmark previous checkpoint"
            )
            previous_checkpoint = {
                **previous_checkpoint,
                "path": _resolve_evidence_path(
                    Path(previous_checkpoint["path"]),
                    base_directory=manifest_directory,
                ),
            }
        requested_step = journal.get("checkpoint_step")
        if type(requested_step) is not int:
            raise EvidenceError("live benchmark attempt has invalid checkpoint step")
        records = _read_jsonl(training_metrics, phase="interrupted training")
        summary, _resumed, _cuda_residency = _validate_training_records(
            records,
            checkpoint=checkpoint,
            requested_step=requested_step,
            previous_checkpoint=previous_checkpoint,
            manifest_directory=manifest_directory,
            wad_profile=wad_profile,
            run_identity=run_identity,
            attempt_identity=attempt_identity,
            interrupted=phase == "training",
            cuda_residency_acceptance=protocol["cuda_residency_acceptance"]["enabled"],
            fixture=protocol["fixture"],
        )
        checkpoint_sha256 = _fsync_file(checkpoint, field="live recovery checkpoint")
        metrics_sha256 = _fsync_file(training_metrics, field="live recovery metrics")
        journal_sha256 = _sha256_bytes(journal_bytes)
        indexed = (
            (journal["checkpoint_evidence_name"], checkpoint_sha256, checkpoint),
            (journal["metrics_evidence_name"], metrics_sha256, training_metrics),
            (journal["journal_evidence_name"], journal_sha256, journal_path),
        )
        existing_names = {entry["name"] for entry in evidence_entries}
        generated_artifacts = []
        for name, sha256, path in indexed:
            if name not in existing_names:
                evidence_entries.append({"name": name, "sha256": sha256})
                existing_names.add(name)
            generated_artifacts.append({"name": name, "path": str(path)})
        elapsed = journal.get("reusable_elapsed_seconds")
        if type(elapsed) not in (int, float) or not math.isfinite(float(elapsed)) or elapsed < 0:
            raise EvidenceError("live benchmark attempt has invalid accumulated elapsed time")
        launch_elapsed = journal.get("reusable_elapsed_seconds_at_launch")
        started_unix_ns = journal.get("started_unix_ns")
        if (
            type(launch_elapsed) not in (int, float)
            or not math.isfinite(float(launch_elapsed))
            or float(launch_elapsed) < 0
            or type(started_unix_ns) is not int
            or started_unix_ns <= 0
        ):
            raise EvidenceError("live benchmark attempt has invalid launch timing")
        elapsed_to_checkpoint = max(
            0.0,
            (checkpoint.stat().st_mtime_ns - started_unix_ns) / 1_000_000_000,
        )
        elapsed = max(
            float(elapsed),
            float(launch_elapsed) + elapsed_to_checkpoint,
        )
        if elapsed_time_anchor is None:
            raise EvidenceError("live benchmark recovery is missing a signed elapsed-time floor")
        return {
            "seed": seed,
            "attempt_identity": attempt_identity,
            "cold_start": journal["cold_start"],
            "status": "interrupted",
            "reusable_elapsed_seconds": float(elapsed),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "outcomes": journal["outcomes"],
            "failures": journal["failures"],
            "recovery": {
                "phase": phase,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "progress_step": summary["train/global_step"],
                "restorable_state": dict(_RESTORABLE_STATE),
                "run_identity": run_identity,
                "attempt_identity": attempt_identity,
                "accumulated_reusable_elapsed_seconds": float(elapsed),
                "checkpoint_step": requested_step,
                "training_metrics": str(training_metrics),
                "checkpoint_evidence_name": journal["checkpoint_evidence_name"],
                "metrics_evidence_name": journal["metrics_evidence_name"],
                "previous_checkpoint": journal.get("previous_checkpoint"),
            },
            "recovery_history": journal["recovery_history"],
            "recovery_journal": None,
            "generated_artifacts": generated_artifacts,
        }
    return None


def _run_attempt(
    *,
    seed: int,
    protocol: dict[str, Any],
    run_identity: str,
    run_directory: Path,
    manifest_directory: Path,
    evidence_entries: list[dict[str, str]],
    wad_profile: dict[str, Any] | None,
    execution_binding: _TrainerFileIdentityMarkers,
    elapsed_time_anchor: dict[str, Any] | None,
    existing_attempt: dict[str, Any] | None = None,
    prior_generated_artifacts: list[dict[str, str]] | None = None,
    started: float | None = None,
    recurring_setup_elapsed: float = 0.0,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    if started is None:
        started = clock()
    attempt_directory = run_directory / f"seed-{seed}"
    attempt_identity = _canonical_sha256(
        {"run_identity": run_identity, "seed": seed},
        document="benchmark attempt",
    )
    if existing_attempt is None:
        if attempt_directory.is_dir():
            existing_attempt = _load_live_interrupted_attempt(
                attempt_directory,
                seed=seed,
                run_identity=run_identity,
                attempt_identity=attempt_identity,
                protocol=protocol,
                manifest_directory=manifest_directory,
                evidence_entries=evidence_entries,
                wad_profile=wad_profile,
                elapsed_time_anchor=elapsed_time_anchor,
            )
            if existing_attempt is None:
                raise EvidenceError(
                    f"seed {seed} artifact directory exists without a recoverable live attempt"
                )
        else:
            attempt_directory.mkdir()
    elif existing_attempt.get("attempt_identity") != attempt_identity:
        raise EvidenceError(f"seed {seed} continuation has unlike attempt identity")
    elif existing_attempt.get("status") != "interrupted":
        return existing_attempt
    seed_file = attempt_directory / "evaluation-seeds.json"
    if seed_file.exists():
        seed_file_sha256 = _fsync_file(seed_file, field="evaluation seed file")
        expected_seed_sha256 = _sha256_bytes(
            (
                json.dumps(protocol["evaluation_episode_seeds"], separators=(",", ":")) + "\n"
            ).encode()
        )
        if seed_file_sha256 != expected_seed_sha256:
            raise EvidenceError(f"seed {seed} continuation changed its evaluation seed grid")
    else:
        seed_file_sha256 = _write_seed_file(seed_file, protocol["evaluation_episode_seeds"])
        evidence_entries.append(
            {"name": f"seed-{seed}-evaluation-seeds", "sha256": seed_file_sha256}
        )
    base_command = [
        *protocol["trainer"]["command"],
        *protocol["trainer"]["arguments"],
    ]
    outcomes: list[dict[str, Any]] = list((existing_attempt or {}).get("outcomes", []))
    failures: list[dict[str, Any]] = list((existing_attempt or {}).get("failures", []))
    recovery_history: list[dict[str, Any]] = list(
        (existing_attempt or {}).get("recovery_history", [])
    )
    prior_elapsed = float((existing_attempt or {}).get("reusable_elapsed_seconds", 0.0))
    recovery_journal: dict[str, Any] | None = (existing_attempt or {}).get("recovery_journal")
    existing_recovery = (existing_attempt or {}).get("recovery")
    previous_checkpoint: dict[str, Any] | None = None
    if isinstance(existing_recovery, dict):
        recovery_path = _resolve_evidence_path(
            Path(existing_recovery["checkpoint"]),
            base_directory=manifest_directory,
        )
        if _fsync_file(recovery_path, field="recovery checkpoint") != existing_recovery.get(
            "checkpoint_sha256"
        ):
            raise EvidenceError(f"seed {seed} recovery checkpoint SHA-256 mismatch")
        if elapsed_time_anchor is None:
            raise EvidenceError("benchmark recovery is missing a signed elapsed-time floor")
        previous_checkpoint = {
            "path": recovery_path,
            "progress_step": existing_recovery["progress_step"],
            "kind": "recovery",
            "checkpoint_sha256": existing_recovery["checkpoint_sha256"],
            "prior_reusable_elapsed_seconds": prior_elapsed,
        }
    elif outcomes:
        previous_checkpoint = {
            "path": _resolve_evidence_path(
                Path(outcomes[-1]["checkpoint"]),
                base_directory=manifest_directory,
            ),
            "progress_step": outcomes[-1]["checkpoint_step"],
            "kind": "checkpoint",
            "checkpoint_sha256": outcomes[-1]["checkpoint_sha256"],
        }
    final_checkpoint = None if not outcomes else Path(outcomes[-1]["checkpoint"])
    final_checkpoint_sha256 = None if not outcomes else outcomes[-1]["checkpoint_sha256"]
    recovery: dict[str, Any] | None = None
    generated_artifacts: list[dict[str, str]] = list(prior_generated_artifacts or [])
    generated_artifacts.extend((existing_attempt or {}).get("generated_artifacts", []))
    generated_artifacts = list(
        {
            (artifact["name"], artifact["path"]): artifact
            for artifact in generated_artifacts
        }.values()
    )
    evaluation_seed_artifact = {
        "name": f"seed-{seed}-evaluation-seeds",
        "path": str(seed_file),
    }
    if evaluation_seed_artifact not in generated_artifacts:
        generated_artifacts.append(evaluation_seed_artifact)
    status = "exhausted"
    cold_start = (existing_attempt or {}).get(
        "cold_start",
        {
            "policy_state": "fresh_random",
            "optimizer_state": "fresh",
            "learned_initialization": False,
        },
    )

    def evaluation_heartbeat(
        *,
        checkpoint_step: int,
        checkpoint: Path,
        training_metrics: Path,
        previous_checkpoint_payload: dict[str, Any] | None,
        checkpoint_evidence_name: str,
        metrics_evidence_name: str,
        evaluation_live_path: Path,
        evaluation_journal_name: str,
    ) -> Callable[[], None]:
        reusable_elapsed_at_launch = prior_elapsed + clock() - started
        live_started_unix_ns = time.time_ns()

        def persist() -> None:
            live_payload = {
                "schema_version": 1,
                "status": "running",
                "phase": "evaluation",
                "run_identity": run_identity,
                "attempt_identity": attempt_identity,
                "seed": seed,
                "continuation_identity": protocol["continuation_identity"],
                "checkpoint_step": checkpoint_step,
                "checkpoint": str(checkpoint),
                "training_metrics": str(training_metrics),
                "previous_checkpoint": previous_checkpoint_payload,
                "cold_start": cold_start,
                "outcomes": outcomes,
                "failures": failures,
                "recovery_history": recovery_history,
                "reusable_elapsed_seconds": prior_elapsed + clock() - started,
                "reusable_elapsed_seconds_at_launch": reusable_elapsed_at_launch,
                "started_unix_ns": live_started_unix_ns,
                "checkpoint_evidence_name": checkpoint_evidence_name,
                "metrics_evidence_name": metrics_evidence_name,
                "journal_evidence_name": evaluation_journal_name,
            }
            live_payload["payload_sha256"] = _canonical_sha256(
                live_payload,
                document="live benchmark attempt journal",
            )
            _write_durable_json(
                evaluation_live_path,
                live_payload,
                field="live benchmark evaluation journal",
            )

        persist()
        return persist

    if isinstance(existing_recovery, dict) and existing_recovery.get("phase") == "evaluation":
        checkpoint_step = existing_recovery["checkpoint_step"]
        checkpoint = _resolve_evidence_path(
            Path(existing_recovery["checkpoint"]), base_directory=manifest_directory
        )
        training_metrics = _resolve_evidence_path(
            Path(existing_recovery["training_metrics"]), base_directory=manifest_directory
        )
        training_records = _read_jsonl(training_metrics, phase="recovered training")
        serialized_previous = existing_recovery.get("previous_checkpoint")
        validation_previous = serialized_previous
        if isinstance(validation_previous, dict):
            validation_previous = {
                **validation_previous,
                "path": _resolve_evidence_path(
                    Path(validation_previous["path"]), base_directory=manifest_directory
                ),
            }
        training_summary, _resumed, cuda_residency = _validate_training_records(
            training_records,
            checkpoint=checkpoint,
            requested_step=checkpoint_step,
            previous_checkpoint=validation_previous,
            manifest_directory=manifest_directory,
            wad_profile=wad_profile,
            run_identity=run_identity,
            attempt_identity=attempt_identity,
            cuda_residency_acceptance=protocol["cuda_residency_acceptance"]["enabled"],
            fixture=protocol["fixture"],
        )
        checkpoint_sha256 = _fsync_file(checkpoint, field="recovered evaluation checkpoint")
        if checkpoint_sha256 != existing_recovery["checkpoint_sha256"]:
            raise EvidenceError("recovered evaluation checkpoint SHA-256 mismatch")
        recovery_history.append(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "progress_step": checkpoint_step,
                "restored_state": dict(_RESTORABLE_STATE),
                "resumed_phase": "evaluation",
                "prior_reusable_elapsed_seconds": prior_elapsed,
            }
        )
        evaluation_metrics = attempt_directory / f"evaluation-step-{checkpoint_step}.jsonl"
        evaluation_live_path = attempt_directory / (
            f"attempt-live-evaluation-step-{checkpoint_step}-recovery.json"
        )
        evaluation_journal_name = f"seed-{seed}-step-{checkpoint_step}-evaluation-live-recovery"
        heartbeat = evaluation_heartbeat(
            checkpoint_step=checkpoint_step,
            checkpoint=checkpoint,
            training_metrics=training_metrics,
            previous_checkpoint_payload=serialized_previous,
            checkpoint_evidence_name=existing_recovery["checkpoint_evidence_name"],
            metrics_evidence_name=existing_recovery["metrics_evidence_name"],
            evaluation_live_path=evaluation_live_path,
            evaluation_journal_name=evaluation_journal_name,
        )
        evaluation_status, outcome, failure, evaluation_sha256 = _execute_evaluation(
            base_command=base_command,
            protocol=protocol,
            seed=seed,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_step=checkpoint_step,
            seed_file=seed_file,
            evaluation_metrics=evaluation_metrics,
            manifest_directory=manifest_directory,
            wad_profile=wad_profile,
            execution_binding=execution_binding,
            heartbeat=heartbeat,
        )
        evaluation_live_sha256 = _fsync_file(
            evaluation_live_path, field="live benchmark evaluation recovery journal"
        )
        if evaluation_journal_name not in {entry["name"] for entry in evidence_entries}:
            evidence_entries.append(
                {"name": evaluation_journal_name, "sha256": evaluation_live_sha256}
            )
        generated_artifacts.append(
            {"name": evaluation_journal_name, "path": str(evaluation_live_path)}
        )
        if failure is not None:
            failures.append(failure)
        if outcome is not None:
            outcome["training"] = training_summary
            outcome["cuda_residency_acceptance"] = cuda_residency
            outcomes.append(outcome)
            assert evaluation_sha256 is not None
            evaluation_name = f"seed-{seed}-step-{checkpoint_step}-evaluation-metrics"
            if evaluation_name not in {entry["name"] for entry in evidence_entries}:
                evidence_entries.append({"name": evaluation_name, "sha256": evaluation_sha256})
            generated_artifacts.append({"name": evaluation_name, "path": str(evaluation_metrics)})
        status = evaluation_status
        previous_checkpoint = {
            "path": checkpoint,
            "progress_step": checkpoint_step,
            "kind": "checkpoint",
            "checkpoint_sha256": checkpoint_sha256,
        }
        final_checkpoint = checkpoint
        final_checkpoint_sha256 = checkpoint_sha256
        recovery = None
    for checkpoint_step in protocol["checkpoint_steps"]:
        if status in {"succeeded", "evaluation_failed", "evidence_failed", "interrupted"}:
            break
        if any(outcome["checkpoint_step"] == checkpoint_step for outcome in outcomes):
            continue
        generation = 0
        checkpoint = attempt_directory / f"checkpoint-step-{checkpoint_step}.pt"
        while checkpoint.exists():
            generation += 1
            checkpoint = (
                attempt_directory / f"checkpoint-step-{checkpoint_step}-recovery-{generation}.pt"
            )
        suffix = "" if generation == 0 else f"-recovery-{generation}"
        training_metrics = attempt_directory / f"training-step-{checkpoint_step}{suffix}.jsonl"
        training_command = [
            *base_command,
            "--evidence-run-identity",
            run_identity,
            "--evidence-attempt-identity",
            attempt_identity,
            "--seed",
            str(seed),
            "--timesteps",
            str(checkpoint_step),
            "--checkpoint",
            str(checkpoint),
            "--metrics-jsonl",
            str(training_metrics),
        ]
        if protocol["cuda_residency_acceptance"]["enabled"]:
            training_command.append("--cuda-residency-acceptance")
        if previous_checkpoint is not None:
            training_command.extend(("--resume", str(previous_checkpoint["path"])))
        checkpoint_evidence_name = f"seed-{seed}-step-{checkpoint_step}{suffix}-checkpoint"
        metrics_evidence_name = f"seed-{seed}-step-{checkpoint_step}{suffix}-training-metrics"
        live_journal_path = attempt_directory / f"attempt-live-step-{checkpoint_step}{suffix}.json"
        journal_evidence_name = f"seed-{seed}-step-{checkpoint_step}{suffix}-live-attempt"
        live_started_unix_ns = time.time_ns()
        reusable_elapsed_at_launch = prior_elapsed + clock() - started
        serialized_previous = None
        if previous_checkpoint is not None:
            serialized_previous = {
                **previous_checkpoint,
                "path": str(previous_checkpoint["path"]),
            }

        def persist_live_attempt(
            *,
            checkpoint_step: int = checkpoint_step,
            checkpoint: Path = checkpoint,
            training_metrics: Path = training_metrics,
            serialized_previous: dict[str, Any] | None = serialized_previous,
            cold_start: dict[str, Any] = cold_start,
            checkpoint_evidence_name: str = checkpoint_evidence_name,
            metrics_evidence_name: str = metrics_evidence_name,
            journal_evidence_name: str = journal_evidence_name,
            live_journal_path: Path = live_journal_path,
            reusable_elapsed_at_launch: float = reusable_elapsed_at_launch,
            live_started_unix_ns: int = live_started_unix_ns,
        ) -> None:
            live_payload = {
                "schema_version": 1,
                "status": "running",
                "run_identity": run_identity,
                "attempt_identity": attempt_identity,
                "seed": seed,
                "continuation_identity": protocol["continuation_identity"],
                "checkpoint_step": checkpoint_step,
                "checkpoint": str(checkpoint),
                "training_metrics": str(training_metrics),
                "previous_checkpoint": serialized_previous,
                "cold_start": cold_start,
                "outcomes": outcomes,
                "failures": failures,
                "recovery_history": recovery_history,
                "reusable_elapsed_seconds": prior_elapsed + clock() - started,
                "reusable_elapsed_seconds_at_launch": reusable_elapsed_at_launch,
                "started_unix_ns": live_started_unix_ns,
                "checkpoint_evidence_name": checkpoint_evidence_name,
                "metrics_evidence_name": metrics_evidence_name,
                "journal_evidence_name": journal_evidence_name,
            }
            live_payload["payload_sha256"] = _canonical_sha256(
                live_payload,
                document="live benchmark attempt journal",
            )
            _write_durable_json(
                live_journal_path,
                live_payload,
                field="live benchmark attempt journal",
            )

        persist_live_attempt()
        training_process = _run_process(
            training_command,
            cwd=manifest_directory,
            execution_binding=execution_binding,
            heartbeat=persist_live_attempt,
        )
        live_journal_sha256 = _fsync_file(
            live_journal_path,
            field="live benchmark attempt journal",
        )
        if journal_evidence_name not in {entry["name"] for entry in evidence_entries}:
            evidence_entries.append({"name": journal_evidence_name, "sha256": live_journal_sha256})
        generated_artifacts.append({"name": journal_evidence_name, "path": str(live_journal_path)})
        crash_left_recovery_evidence = (
            training_process.returncode not in {0, 130}
            and checkpoint.is_file()
            and training_metrics.is_file()
        )
        if training_process.returncode not in {0, 130} and not crash_left_recovery_evidence:
            failures.append(
                _failure(
                    seed=seed,
                    phase="training",
                    checkpoint_step=checkpoint_step,
                    process=training_process,
                )
            )
            status = "crashed"
            break
        was_interrupted = training_process.returncode == 130 or crash_left_recovery_evidence
        try:
            training_records = _read_jsonl(training_metrics, phase="training")
            training_summary, resumed_record, cuda_residency = _validate_training_records(
                training_records,
                checkpoint=checkpoint,
                requested_step=checkpoint_step,
                previous_checkpoint=previous_checkpoint,
                manifest_directory=manifest_directory,
                wad_profile=wad_profile,
                run_identity=run_identity,
                attempt_identity=attempt_identity,
                interrupted=was_interrupted,
                cuda_residency_acceptance=protocol["cuda_residency_acceptance"]["enabled"],
                fixture=protocol["fixture"],
            )
            checkpoint_sha256 = _fsync_file(checkpoint, field="training checkpoint")
            training_metrics_sha256 = _fsync_file(
                training_metrics,
                field="training metrics",
            )
        except EvidenceError as error:
            failures.append(
                {
                    "seed": seed,
                    "phase": "training_evidence",
                    "checkpoint_step": checkpoint_step,
                    "message": str(error),
                }
            )
            status = "evidence_failed"
            break
        evidence_entries.extend(
            (
                {
                    "name": checkpoint_evidence_name,
                    "sha256": checkpoint_sha256,
                },
                {
                    "name": metrics_evidence_name,
                    "sha256": training_metrics_sha256,
                },
            )
        )
        generated_artifacts.extend(
            (
                {"name": evidence_entries[-2]["name"], "path": str(checkpoint)},
                {"name": evidence_entries[-1]["name"], "path": str(training_metrics)},
            )
        )
        if previous_checkpoint is not None and previous_checkpoint.get("kind") == "recovery":
            assert resumed_record is not None
            recovery_history.append(
                {
                    "checkpoint": str(previous_checkpoint["path"]),
                    "checkpoint_sha256": previous_checkpoint["checkpoint_sha256"],
                    "progress_step": previous_checkpoint["progress_step"],
                    "restored_state": resumed_record["restored_state"],
                    "prior_reusable_elapsed_seconds": previous_checkpoint[
                        "prior_reusable_elapsed_seconds"
                    ],
                }
            )
        if was_interrupted:
            final_checkpoint = checkpoint
            final_checkpoint_sha256 = checkpoint_sha256
            if elapsed_time_anchor is None:
                failures.append(
                    {
                        "seed": seed,
                        "phase": "recovery_trust",
                        "checkpoint_step": checkpoint_step,
                        "message": "interruption has no pre-attempt signed elapsed-time floor",
                    }
                )
                status = "evidence_failed"
                break
            recovery = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "progress_step": training_summary["train/global_step"],
                "restorable_state": dict(_RESTORABLE_STATE),
                "run_identity": run_identity,
                "attempt_identity": attempt_identity,
                "accumulated_reusable_elapsed_seconds": 0.0,
            }
            status = "interrupted"
            break
        evaluation_metrics = attempt_directory / f"evaluation-step-{checkpoint_step}.jsonl"
        evaluation_live_path = attempt_directory / (
            f"attempt-live-evaluation-step-{checkpoint_step}.json"
        )
        evaluation_journal_name = f"seed-{seed}-step-{checkpoint_step}-evaluation-live"
        heartbeat = evaluation_heartbeat(
            checkpoint_step=checkpoint_step,
            checkpoint=checkpoint,
            training_metrics=training_metrics,
            previous_checkpoint_payload=serialized_previous,
            checkpoint_evidence_name=checkpoint_evidence_name,
            metrics_evidence_name=metrics_evidence_name,
            evaluation_live_path=evaluation_live_path,
            evaluation_journal_name=evaluation_journal_name,
        )
        evaluation_status, outcome, failure, evaluation_metrics_sha256 = _execute_evaluation(
            base_command=base_command,
            protocol=protocol,
            seed=seed,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_step=checkpoint_step,
            seed_file=seed_file,
            evaluation_metrics=evaluation_metrics,
            manifest_directory=manifest_directory,
            wad_profile=wad_profile,
            execution_binding=execution_binding,
            heartbeat=heartbeat,
        )
        evaluation_live_sha256 = _fsync_file(
            evaluation_live_path, field="live benchmark evaluation journal"
        )
        if evaluation_journal_name not in {entry["name"] for entry in evidence_entries}:
            evidence_entries.append(
                {"name": evaluation_journal_name, "sha256": evaluation_live_sha256}
            )
        generated_artifacts.append(
            {"name": evaluation_journal_name, "path": str(evaluation_live_path)}
        )
        if failure is not None:
            failures.append(failure)
        if outcome is not None:
            outcome["training"] = training_summary
            outcome["cuda_residency_acceptance"] = cuda_residency
            outcomes.append(outcome)
            assert evaluation_metrics_sha256 is not None
            evidence_entries.append(
                {
                    "name": f"seed-{seed}-step-{checkpoint_step}-evaluation-metrics",
                    "sha256": evaluation_metrics_sha256,
                }
            )
            generated_artifacts.append(
                {"name": evidence_entries[-1]["name"], "path": str(evaluation_metrics)}
            )
        if evaluation_status == "interrupted":
            recovery = {
                "phase": "evaluation",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "progress_step": checkpoint_step,
                "checkpoint_step": checkpoint_step,
                "training_metrics": str(training_metrics),
                "checkpoint_evidence_name": checkpoint_evidence_name,
                "metrics_evidence_name": metrics_evidence_name,
                "previous_checkpoint": serialized_previous,
                "restorable_state": dict(_RESTORABLE_STATE),
                "run_identity": run_identity,
                "attempt_identity": attempt_identity,
                "accumulated_reusable_elapsed_seconds": 0.0,
            }
        status = evaluation_status
        previous_checkpoint = {
            "path": checkpoint,
            "progress_step": checkpoint_step,
            "kind": "checkpoint",
            "checkpoint_sha256": checkpoint_sha256,
        }
        final_checkpoint = checkpoint
        final_checkpoint_sha256 = checkpoint_sha256
        if status != "exhausted":
            break
    elapsed = prior_elapsed + recurring_setup_elapsed + clock() - started
    if recovery is not None:
        recovery["accumulated_reusable_elapsed_seconds"] = elapsed
        recovery_generation = _next_append_only_generation(
            attempt_directory,
            f"recovery-step-{recovery['progress_step']}-*.json",
            f"recovery-step-{recovery['progress_step']}-",
        )
        recovery_journal_path = attempt_directory / (
            f"recovery-step-{recovery['progress_step']}-{recovery_generation}.json"
        )
        recovery_journal_payload = {
            "schema_version": 1,
            "run_identity": run_identity,
            "attempt_identity": attempt_identity,
            "seed": seed,
            "checkpoint": recovery["checkpoint"],
            "checkpoint_sha256": recovery["checkpoint_sha256"],
            "progress_step": recovery["progress_step"],
            "restorable_state": _RESTORABLE_STATE,
            "accumulated_reusable_elapsed_seconds": elapsed,
        }
        recovery_journal_path.write_text(
            json.dumps(recovery_journal_payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        recovery_journal_sha256 = _fsync_file(
            recovery_journal_path,
            field="recovery journal",
        )
        recovery_journal = {
            "path": str(recovery_journal_path),
            "sha256": recovery_journal_sha256,
        }
        recovery_evidence_name = (
            f"seed-{seed}-recovery-step-{recovery['progress_step']}-{recovery_generation}"
        )
        evidence_entries.append({"name": recovery_evidence_name, "sha256": recovery_journal_sha256})
        generated_artifacts.append(
            {"name": recovery_evidence_name, "path": str(recovery_journal_path)}
        )
    attempt = {
        "seed": seed,
        "attempt_identity": attempt_identity,
        "cold_start": {
            "policy_state": "fresh_random",
            "optimizer_state": "fresh",
            "learned_initialization": False,
        },
        "status": status,
        "reusable_elapsed_seconds": elapsed,
        "checkpoint": None if final_checkpoint is None else str(final_checkpoint),
        "checkpoint_sha256": final_checkpoint_sha256,
        "outcomes": outcomes,
        "failures": failures,
        "recovery": recovery,
        "recovery_history": recovery_history,
        "recovery_journal": recovery_journal,
        "generated_artifacts": generated_artifacts,
    }
    attempt_journal_generation = _next_append_only_generation(
        attempt_directory,
        "attempt-state-*.json",
        "attempt-state-",
    )
    attempt_journal_path = attempt_directory / (f"attempt-state-{attempt_journal_generation}.json")
    previous_journal_sha256 = None
    if attempt_journal_generation:
        previous_journal_sha256 = _fsync_file(
            attempt_directory / f"attempt-state-{attempt_journal_generation - 1}.json",
            field="previous benchmark attempt journal",
        )
    attempt_journal_payload = _attempt_journal_payload(
        attempt,
        run_identity=run_identity,
        generation=attempt_journal_generation,
        previous_journal_sha256=previous_journal_sha256,
    )
    if elapsed_time_anchor is None:
        raise EvidenceError("benchmark attempt outcome has no external time authority")
    _write_durable_json(
        attempt_journal_path,
        attempt_journal_payload,
        field="benchmark attempt journal",
    )
    attempt_journal_sha256 = _fsync_file(
        attempt_journal_path,
        field="benchmark attempt journal",
    )
    authority_attestation = _sign_generation_attestation(
        _journal_attestation_payload(
            attempt_journal_payload,
            anchor=elapsed_time_anchor,
            journal_sha256=attempt_journal_sha256,
        ),
        anchor=elapsed_time_anchor,
        prior_elapsed=prior_elapsed,
        minimum_elapsed=elapsed,
        started=started,
        clock=clock,
    )
    elapsed = float(authority_attestation["payload"]["reusable_elapsed_seconds"])
    attempt["reusable_elapsed_seconds"] = elapsed
    if recovery is not None:
        recovery["accumulated_reusable_elapsed_seconds"] = elapsed
    attempt_journal_name = f"seed-{seed}-attempt-state-{attempt_journal_generation}"
    evidence_entries.append({"name": attempt_journal_name, "sha256": attempt_journal_sha256})
    generated_artifacts.append({"name": attempt_journal_name, "path": str(attempt_journal_path)})
    attempt["attempt_journal"] = {
        "path": str(attempt_journal_path),
        "sha256": attempt_journal_sha256,
        "authority_attestation": authority_attestation,
    }
    attempt_seal_name = f"seed-{seed}-attempt-seal-{attempt_journal_generation}"
    attempt_seal_path = attempt_directory / f"attempt-seal-{attempt_journal_generation}.json"
    attempt_seal_sha256 = _write_durable_json(
        attempt_seal_path,
        {
            "schema_version": 1,
            "run_identity": run_identity,
            "continuation_identity": protocol["continuation_identity"],
            "attempt": attempt,
            "evidence_entries": evidence_entries,
            "generated_artifacts": generated_artifacts,
        },
        field="signed durable benchmark attempt seal",
    )
    attempt["sealed_attempt"] = {
        "path": str(attempt_seal_path),
        "sha256": attempt_seal_sha256,
    }
    evidence_entries.append({"name": attempt_seal_name, "sha256": attempt_seal_sha256})
    generated_artifacts.append({"name": attempt_seal_name, "path": str(attempt_seal_path)})
    attempt["_terminal_timing"] = {
        "prior_elapsed": prior_elapsed,
        "terminal_elapsed": elapsed,
        "anchor": elapsed_time_anchor,
        "journal_payload": attempt_journal_payload,
        "journal_sha256": attempt_journal_sha256,
    }
    return attempt


def _build_development_benchmark_report(
    manifest_path: Path,
    *,
    identity_marker_registry: list[_TrainerFileIdentityMarkers],
    merge_path: Path | None = None,
    invocation_started: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
    report_writer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if invocation_started is None:
        invocation_started = clock()
    manifest, manifest_payload = _load_manifest(manifest_path)
    validated = _validate_benchmark(manifest)
    declared_inputs = _validate_declared_inputs(
        manifest.get("declared_inputs"),
        base_directory=manifest_path.parent,
    )
    evidence_entries = [{"name": "manifest", "sha256": _sha256_bytes(manifest_payload)}]
    artifacts_root = _resolve_evidence_path(
        Path(validated["artifacts_directory"]),
        base_directory=manifest_path.parent,
    )
    validated["trainer"] = _bind_trainer_files(
        validated["trainer"],
        base_directory=manifest_path.parent,
        artifacts_root=artifacts_root,
    )
    identity_marker_registry.append(validated["trainer"]["_identity_markers"])
    if not manifest["fixture"]:
        authority = _formal_time_authority()
        for authority_root, label in (
            (authority.state_directory, "state"),
            (authority.witness_directory, "monotonic witness"),
        ):
            if (
                authority_root == artifacts_root
                or authority_root.is_relative_to(artifacts_root)
                or artifacts_root.is_relative_to(authority_root)
            ):
                raise EvidenceError(
                    f"formal reusable-time authority {label} must not overlap benchmark artifacts"
                )
    bootstrap_exclusions = _validate_bootstrap_files(
        validated["bootstrap_artifacts"],
        base_directory=manifest_path.parent,
        artifacts_root=artifacts_root,
        declared_inputs=declared_inputs,
    )
    evidence_entries.extend(
        {"name": f"bootstrap-{item['name']}", "sha256": item["sha256"]}
        for item in bootstrap_exclusions
    )
    wad_profile = None
    if "wad_profile" in manifest:
        wad_profile, wad_entries = validate_wad_profile(
            manifest["wad_profile"],
            base_directory=manifest_path.parent,
        )
        evidence_entries.extend(wad_entries)
        if wad_profile["status"] != "matched":
            raise EvidenceError("development benchmark WAD profile did not match")
    elif not manifest["fixture"]:
        raise EvidenceError("non-fixture development benchmark requires wad_profile")
    reserved_evidence_names = {entry["name"] for entry in evidence_entries}
    reserved_evidence_names.update(
        f"seed-{seed}-evaluation-seeds" for seed in validated["training_seeds"]
    )
    reserved_evidence_names.update(
        name
        for seed in validated["training_seeds"]
        for step in validated["checkpoint_steps"]
        for name in (
            f"seed-{seed}-step-{step}-checkpoint",
            f"seed-{seed}-step-{step}-training-metrics",
            f"seed-{seed}-step-{step}-evaluation-metrics",
        )
    )
    for declared_input in declared_inputs:
        if declared_input["name"] in reserved_evidence_names:
            raise EvidenceError(
                f"declared input name {declared_input['name']!r} is reserved by the benchmark"
            )
        input_path = _resolve_evidence_path(
            Path(declared_input["path"]),
            base_directory=manifest_path.parent,
        )
        try:
            actual_sha256 = _sha256_bytes(input_path.read_bytes())
        except OSError as error:
            raise EvidenceError(
                f"cannot read declared input {declared_input['name']!r}: {error}"
            ) from error
        if actual_sha256 != declared_input["sha256"]:
            raise EvidenceError(
                f"declared input {declared_input['name']!r} SHA-256 mismatch: "
                f"expected {declared_input['sha256']}, got {actual_sha256}"
            )
        evidence_entries.append({"name": declared_input["name"], "sha256": actual_sha256})
    protocol = {
        "training_seeds": validated["training_seeds"],
        "failure_budget_steps": validated["failure_budget_steps"],
        "checkpoint_steps": validated["checkpoint_steps"],
        "evaluation_episode_seeds": validated["evaluation_episode_seeds"],
        "evaluation_action_seed": validated["evaluation_action_seed"],
        "quality_gate": {
            "episodes": _EVALUATION_EPISODES,
            "mean_at_least": _QUALITY_THRESHOLD,
            "signal": "player_killcount",
            "stochastic_actions": True,
        },
        "cold_start": {
            "policy_state": "fresh_random",
            "optimizer_state": "fresh",
            "learned_initialization_allowed": False,
        },
        "timer_includes": [
            "command_parsing",
            "manifest_and_configuration_validation",
            "identity_and_input_hashing",
            "artifact_directory_setup",
            "continuation_and_recovery_verification",
            "recurring_initialization",
            "per_process_or_uncached_compilation",
            "graph_capture",
            "warmup",
            "training",
            "checkpoint_evaluation",
            "durable_checkpoint_write",
            "terminal_evidence_verification",
            "report_validation_serialization_replacement_and_fsync",
            "durable_authority_elapsed_seal",
        ],
        "timer_boundaries": {
            "start": (
                "before_command_parsing_manifest_validation_identity_hashing_artifact_setup_"
                "and_continuation_verification"
            ),
            "resume": "add_prior_hashed_recovery_elapsed_before_recurring_recovery_work",
            "stop": "after_final_durable_report_contains_a_conservative_authority_elapsed_seal",
        },
        "trainer": {
            key: value
            for key, value in validated["trainer"].items()
            if key != "_identity_markers"
        },
        "fixture": manifest["fixture"],
        "cuda_residency_acceptance": {
            "contract": CUDA_RESIDENCY_CONTRACT,
            "enabled": validated["cuda_residency_acceptance"],
        },
        "parity_certificate": validated["parity_certificate"],
        "wad_profile_binding_sha256": (
            None if wad_profile is None else wad_profile["binding_sha256"]
        ),
        "bootstrap_artifacts": [
            {
                key: value
                for key, value in artifact.items()
                if key not in {"validated_before_cohort", "reverified_unchanged_after_cohort"}
            }
            for artifact in bootstrap_exclusions
        ],
        "elapsed_time_anchors": validated["elapsed_time_anchors"],
        "time_authority": (
            {
                "authority": _FIXTURE_ELAPSED_ANCHOR_AUTHORITY,
                "public_key": _FIXTURE_ELAPSED_ANCHOR_PUBLIC_KEY,
                "monotonic_witness": None,
                "claim_eligible": False,
            }
            if manifest["fixture"]
            else dict(_formal_time_authority().identity)
        ),
    }
    protocol["continuation_identity"] = {
        "schema_sha256": _canonical_sha256(
            {
                "schema_version": 1,
                "workflow": _WORKFLOW,
                "evidence_level": "development",
                "trainer_contract": _TRAINER_CONTRACT,
            },
            document="benchmark schema identity",
        ),
        "recipe_sha256": _canonical_sha256(
            protocol["trainer"],
            document="benchmark recipe identity",
        ),
        "asset_sha256": _canonical_sha256(
            {
                "wad_profile_binding_sha256": protocol["wad_profile_binding_sha256"],
                "declared_inputs": sorted(
                    ({"name": item["name"], "sha256": item["sha256"]} for item in declared_inputs),
                    key=lambda item: item["name"],
                ),
                "bootstrap_artifacts": [
                    {"name": item["name"], "sha256": item["sha256"]}
                    for item in protocol["bootstrap_artifacts"]
                ],
            },
            document="benchmark asset identity",
        ),
        "seed_sha256": _canonical_sha256(
            {
                "training_seeds": protocol["training_seeds"],
                "evaluation_episode_seeds": protocol["evaluation_episode_seeds"],
                "evaluation_action_seed": protocol["evaluation_action_seed"],
            },
            document="benchmark seed identity",
        ),
        "timer_sha256": _canonical_sha256(
            {
                "includes": protocol["timer_includes"],
                "boundaries": protocol["timer_boundaries"],
            },
            document="benchmark timer identity",
        ),
    }
    identity_payload = {
        "schema_version": 1,
        "workflow": _WORKFLOW,
        "evidence_level": "development",
        "fixture": manifest["fixture"],
        "code_provenance": validated["code_provenance"],
        "declared_inputs": sorted(
            ({"name": item["name"], "sha256": item["sha256"]} for item in declared_inputs),
            key=lambda item: item["name"],
        ),
        "benchmark_protocol": protocol,
    }
    run_identity = _canonical_sha256(identity_payload, document="manifest")
    continuation = None
    if merge_path is not None:
        continuation = _load_benchmark_continuation(
            merge_path,
            run_identity=run_identity,
            protocol=protocol,
            code_provenance=validated["code_provenance"],
            declared_inputs=declared_inputs,
            wad_profile=wad_profile,
            initial_evidence_entries=evidence_entries,
            manifest_directory=manifest_path.parent,
        )
        evidence_entries = [dict(entry) for entry in continuation["evidence_index"]["entries"]]
    run_directory = artifacts_root / run_identity
    run_directory_existed = run_directory.exists()
    if continuation is None and not run_directory_existed:
        try:
            run_directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise EvidenceError(
                "benchmark artifact directory already exists; refusing to overwrite: "
                f"{run_directory}"
            ) from error
    elif not run_directory.is_dir():
        raise EvidenceError("benchmark continuation artifact directory is missing")
    durable_attempts: list[dict[str, Any]] = []
    durable_generated_artifacts: list[dict[str, str]] = []
    if continuation is None and run_directory_existed:
        durable_attempts, evidence_entries, durable_generated_artifacts = (
            _load_durable_attempt_seals(
                run_directory,
                run_identity=run_identity,
                protocol=protocol,
                initial_evidence_entries=evidence_entries,
                manifest_directory=manifest_path.parent,
            )
        )
    prior_attempts = (
        continuation.get("attempts", []) if continuation is not None else durable_attempts
    )
    existing_by_seed = {
        attempt["seed"]: attempt for attempt in prior_attempts
    }
    attempts = []
    actual_generated_artifacts: list[dict[str, str]] = list(
        (continuation or {}).get("generated_artifacts", durable_generated_artifacts)
    )
    recurring_setup_elapsed = max(0.0, clock() - invocation_started)
    anchors_by_seed = {
        anchor["payload"]["seed"]: anchor for anchor in validated["elapsed_time_anchors"]
    }
    for seed in validated["training_seeds"]:
        existing_attempt = existing_by_seed.get(seed)
        active_attempt = existing_attempt is None or existing_attempt.get("status") == "interrupted"
        attempt = _run_attempt(
            seed=seed,
            protocol=protocol,
            run_identity=run_identity,
            run_directory=run_directory,
            manifest_directory=manifest_path.parent,
            evidence_entries=evidence_entries,
            wad_profile=wad_profile,
            execution_binding=validated["trainer"]["_identity_markers"],
            elapsed_time_anchor=anchors_by_seed.get(seed),
            existing_attempt=existing_attempt,
            prior_generated_artifacts=actual_generated_artifacts,
            started=(clock() if active_attempt else None),
            recurring_setup_elapsed=(recurring_setup_elapsed if active_attempt else 0.0),
            clock=clock,
        )
        recovered_terminal_timing = attempt.pop("_recovered_terminal_timing", None)
        if isinstance(recovered_terminal_timing, dict):
            attempt["_terminal_timing"] = {
                **recovered_terminal_timing,
                "terminal_elapsed": attempt["reusable_elapsed_seconds"],
            }
        actual_generated_artifacts.extend(attempt.pop("generated_artifacts", []))
        attempts.append(attempt)
    generated_artifacts: list[dict[str, str]] = list(
        (continuation or {}).get("generated_artifacts", [])
    )
    for seed in validated["training_seeds"]:
        attempt_directory = run_directory / f"seed-{seed}"
        generated_artifacts.append(
            {
                "name": f"seed-{seed}-evaluation-seeds",
                "path": str(attempt_directory / "evaluation-seeds.json"),
            }
        )
        for step in validated["checkpoint_steps"]:
            for kind, path in (
                ("checkpoint", attempt_directory / f"checkpoint-step-{step}.pt"),
                ("training-metrics", attempt_directory / f"training-step-{step}.jsonl"),
                ("evaluation-metrics", attempt_directory / f"evaluation-step-{step}.jsonl"),
            ):
                generated_artifacts.append(
                    {
                        "name": f"seed-{seed}-step-{step}-{kind}",
                        "path": str(path),
                    }
                )
    generated_artifacts.extend(actual_generated_artifacts)
    generated_artifacts = list(
        {
            (artifact["name"], artifact["path"]): artifact for artifact in generated_artifacts
        }.values()
    )
    failures = [failure for attempt in attempts for failure in attempt["failures"]]
    evidence_names = [entry["name"] for entry in evidence_entries]
    if len(evidence_names) != len(set(evidence_names)):
        raise EvidenceError("benchmark evidence index contains duplicate entry names")
    all_succeeded = all(attempt["status"] == "succeeded" for attempt in attempts)
    claim_reasons: list[dict[str, Any]] = [
        {
            "code": "development_evidence",
            "message": "Development evidence is non-authoritative and cannot support claims.",
        }
    ]
    if manifest["fixture"]:
        claim_reasons.append(
            {
                "code": "fixture_evidence",
                "message": "Fixture evidence cannot support public claims.",
            }
        )
    certificate = validated["parity_certificate"]
    if not certificate["available"]:
        claim_reasons.append(
            {
                "code": "missing_current_parity_certificate",
                "message": certificate["reason"],
            }
        )
    report = {
        "schema_version": 1,
        "workflow": _WORKFLOW,
        "evidence_level": "development",
        "fixture": manifest["fixture"],
        "authoritative": False,
        "status": "passed" if all_succeeded else "failed",
        "claim_eligible": False,
        "claim_reasons": claim_reasons,
        "run_identity": run_identity,
        "code_provenance": validated["code_provenance"],
        "declared_inputs": declared_inputs,
        "benchmark_protocol": protocol,
        "bootstrap_exclusions": bootstrap_exclusions,
        "wad_profile": wad_profile,
        "attempts": attempts,
        "failures": failures,
        "generated_artifacts": generated_artifacts,
        "evidence_index": {
            "algorithm": "sha256",
            "entries": evidence_entries,
            "sha256": _canonical_sha256(evidence_entries, document="manifest"),
        },
    }
    terminal_timing: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for attempt in attempts:
        timing = attempt.pop("_terminal_timing", None)
        if isinstance(timing, dict):
            terminal_timing.append((attempt, timing))
    finalization_started = clock()
    _reverify_bootstrap_files(bootstrap_exclusions)
    _reverify_trainer_files(validated["trainer"])
    shared_finalization_elapsed = max(0.0, clock() - finalization_started)
    if report_writer is not None:
        write_started = clock()
        report_writer(report)
        write_finished = clock()
        shared_finalization_elapsed += max(0.0, write_finished - write_started)
        if not terminal_timing:
            return report
        allowance = max(0.001, 2.0 * max(0.0, write_finished - write_started))
        signing_elapsed = {id(attempt): 0.0 for attempt, _timing in terminal_timing}
        for _iteration in range(8):
            for attempt, timing in terminal_timing:
                sign_started = clock()
                minimum_elapsed = max(
                    float(attempt["reusable_elapsed_seconds"]),
                    float(timing["terminal_elapsed"])
                    + shared_finalization_elapsed
                    + signing_elapsed[id(attempt)]
                    + allowance,
                )
                attestation = _sign_generation_attestation(
                    _journal_attestation_payload(
                        timing["journal_payload"],
                        anchor=timing["anchor"],
                        journal_sha256=timing["journal_sha256"],
                    ),
                    anchor=timing["anchor"],
                    prior_elapsed=(
                        float(timing["prior_elapsed"])
                        if timing["anchor"]["payload"]["authority"]
                        == _FIXTURE_ELAPSED_ANCHOR_AUTHORITY
                        else float(attempt["reusable_elapsed_seconds"])
                    ),
                    minimum_elapsed=minimum_elapsed,
                    started=sign_started,
                    clock=clock,
                )
                signing_elapsed[id(attempt)] += max(0.0, clock() - sign_started)
                elapsed = float(attestation["payload"]["reusable_elapsed_seconds"])
                attempt["reusable_elapsed_seconds"] = elapsed
                recovery = attempt.get("recovery")
                if isinstance(recovery, dict):
                    recovery["accumulated_reusable_elapsed_seconds"] = elapsed
                attempt["attempt_journal"]["authority_attestation"] = attestation
            write_started = clock()
            report_writer(report)
            write_finished = clock()
            shared_finalization_elapsed += max(0.0, write_finished - write_started)
            if all(
                float(attempt["reusable_elapsed_seconds"])
                >= float(timing["terminal_elapsed"])
                + shared_finalization_elapsed
                + signing_elapsed[id(attempt)]
                for attempt, timing in terminal_timing
            ):
                break
            allowance = max(
                allowance * 2.0,
                2.0 * max(0.0, write_finished - write_started),
            )
        else:
            raise EvidenceError(
                "could not conservatively seal reusable elapsed time through durable report write"
            )
    return report


def build_development_benchmark_report(
    manifest_path: Path,
    *,
    merge_path: Path | None = None,
    invocation_started: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
    report_writer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    identity_marker_registry: list[_TrainerFileIdentityMarkers] = []
    try:
        return _build_development_benchmark_report(
            manifest_path,
            identity_marker_registry=identity_marker_registry,
            merge_path=merge_path,
            invocation_started=invocation_started,
            clock=clock,
            report_writer=report_writer,
        )
    finally:
        for identity_markers in identity_marker_registry:
            identity_markers.close()
