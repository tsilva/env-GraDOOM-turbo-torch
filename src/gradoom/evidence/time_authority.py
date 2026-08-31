from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


class TimeAuthorityError(ValueError):
    pass


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: object, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(raw_temporary)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, field: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TimeAuthorityError(f"{field} is missing or invalid") from error


def _signature(private_key: Ed25519PrivateKey, payload: dict[str, Any]) -> str:
    return base64.b64encode(private_key.sign(_canonical_bytes(payload))).decode()


def _verify_signature(public_key: Ed25519PublicKey, signature: object, payload: object) -> None:
    if not isinstance(signature, str):
        raise TimeAuthorityError("authority signature is missing")
    try:
        raw = base64.b64decode(signature, validate=True)
        public_key.verify(raw, _canonical_bytes(payload))
    except (ValueError, InvalidSignature) as error:
        raise TimeAuthorityError("authority signature is invalid") from error


class ReusableTimeAuthority:
    """Repository-owned persistent authority for reusable-time and bootstrap history."""

    def __init__(self, state_directory: Path, witness_directory: Path | None = None) -> None:
        self.state_directory = state_directory.resolve()
        self.witness_directory = (
            witness_directory.resolve()
            if witness_directory is not None
            else self.default_witness_directory(self.state_directory)
        )
        if (
            self.witness_directory == self.state_directory
            or self.witness_directory.is_relative_to(self.state_directory)
            or self.state_directory.is_relative_to(self.witness_directory)
        ):
            raise TimeAuthorityError(
                "authority witness must use an independent directory outside authority state"
            )
        try:
            state_mode = self.state_directory.stat().st_mode
            private_mode = (self.state_directory / "private-key.json").stat().st_mode
            witness_mode = self.witness_directory.stat().st_mode
            witness_private_mode = (self.witness_directory / "private-key.json").stat().st_mode
        except OSError as error:
            raise TimeAuthorityError("authority state permissions cannot be inspected") from error
        if (
            state_mode & 0o022
            or private_mode & 0o077
            or witness_mode & 0o022
            or witness_private_mode & 0o077
        ):
            raise TimeAuthorityError("authority state or private key permissions are unsafe")
        identity = _read_json(self.state_directory / "identity.json", "authority identity")
        private_document = _read_json(
            self.state_directory / "private-key.json", "authority private key"
        )
        if not isinstance(identity, dict) or set(identity) != {
            "schema_version",
            "authority",
            "public_key",
            "created_unix_ns",
            "witness",
            "witness_public_key",
            "witness_directory",
        }:
            raise TimeAuthorityError("authority identity has an unsupported schema")
        if identity.get("schema_version") != 1:
            raise TimeAuthorityError("authority identity has an unsupported schema")
        try:
            public_bytes = base64.b64decode(identity["public_key"], validate=True)
            private_bytes = base64.b64decode(private_document["private_key"], validate=True)
            self.public_key = Ed25519PublicKey.from_public_bytes(public_bytes)
            self.private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)
        except (KeyError, TypeError, ValueError) as error:
            raise TimeAuthorityError("authority key material is invalid") from error
        derived_public = self.private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        if derived_public != public_bytes:
            raise TimeAuthorityError("authority public and private identities do not match")
        expected_name = f"gradoom-reusable-time-authority-v1:{_sha256(public_bytes)[:32]}"
        if identity["authority"] != expected_name:
            raise TimeAuthorityError("authority identity does not match its public key")
        if identity["witness_directory"] != str(self.witness_directory):
            raise TimeAuthorityError("authority witness directory does not match its identity")
        witness_identity = _read_json(
            self.witness_directory / "identity.json", "authority monotonic witness identity"
        )
        witness_private_document = _read_json(
            self.witness_directory / "private-key.json",
            "authority monotonic witness private key",
        )
        if not isinstance(witness_identity, dict) or set(witness_identity) != {
            "schema_version",
            "witness",
            "public_key",
            "created_unix_ns",
        }:
            raise TimeAuthorityError("authority monotonic witness has an unsupported schema")
        try:
            witness_public_bytes = base64.b64decode(witness_identity["public_key"], validate=True)
            witness_private_bytes = base64.b64decode(
                witness_private_document["private_key"], validate=True
            )
            self.witness_public_key = Ed25519PublicKey.from_public_bytes(witness_public_bytes)
            self.witness_private_key = Ed25519PrivateKey.from_private_bytes(witness_private_bytes)
        except (KeyError, TypeError, ValueError) as error:
            raise TimeAuthorityError(
                "authority monotonic witness key material is invalid"
            ) from error
        derived_witness_public = self.witness_private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        expected_witness = f"gradoom-monotonic-witness-v1:{_sha256(witness_public_bytes)[:32]}"
        if (
            derived_witness_public != witness_public_bytes
            or witness_identity["witness"] != expected_witness
            or identity["witness"] != expected_witness
            or identity["witness_public_key"] != witness_identity["public_key"]
        ):
            raise TimeAuthorityError("authority monotonic witness identity is invalid")
        self.identity = identity
        self.witness_identity = witness_identity
        self.ledger = self._validated_ledger()

    @staticmethod
    def default_witness_directory(state_directory: Path) -> Path:
        state_directory = state_directory.resolve()
        return state_directory.parent / f".{state_directory.name}.monotonic-witness"

    @classmethod
    def initialize(
        cls, state_directory: Path, witness_directory: Path | None = None
    ) -> ReusableTimeAuthority:
        state_directory = state_directory.resolve()
        witness_directory = (
            witness_directory.resolve()
            if witness_directory is not None
            else cls.default_witness_directory(state_directory)
        )
        if (
            witness_directory == state_directory
            or witness_directory.is_relative_to(state_directory)
            or state_directory.is_relative_to(witness_directory)
        ):
            raise TimeAuthorityError(
                "authority witness must use an independent directory outside authority state"
            )
        if state_directory.exists() and any(state_directory.iterdir()):
            raise TimeAuthorityError("authority state directory is not empty")
        if witness_directory.exists() and any(witness_directory.iterdir()):
            raise TimeAuthorityError("authority monotonic witness directory is not empty")
        state_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        state_directory.chmod(0o700)
        witness_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        witness_directory.chmod(0o700)
        private_key = Ed25519PrivateKey.generate()
        witness_private_key = Ed25519PrivateKey.generate()
        private_bytes = private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        public_bytes = private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        witness_private_bytes = witness_private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        witness_public_bytes = witness_private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        witness_identity = {
            "schema_version": 1,
            "witness": f"gradoom-monotonic-witness-v1:{_sha256(witness_public_bytes)[:32]}",
            "public_key": base64.b64encode(witness_public_bytes).decode(),
            "created_unix_ns": time.time_ns(),
        }
        identity = {
            "schema_version": 1,
            "authority": f"gradoom-reusable-time-authority-v1:{_sha256(public_bytes)[:32]}",
            "public_key": base64.b64encode(public_bytes).decode(),
            "created_unix_ns": time.time_ns(),
            "witness": witness_identity["witness"],
            "witness_public_key": witness_identity["public_key"],
            "witness_directory": str(witness_directory),
        }
        _atomic_json(
            witness_directory / "private-key.json",
            {
                "schema_version": 1,
                "private_key": base64.b64encode(witness_private_bytes).decode(),
            },
        )
        _atomic_json(witness_directory / "identity.json", witness_identity)
        _atomic_json(
            state_directory / "private-key.json",
            {
                "schema_version": 1,
                "private_key": base64.b64encode(private_bytes).decode(),
            },
        )
        _atomic_json(state_directory / "identity.json", identity)
        _atomic_json(state_directory / "ledger.json", [])
        empty_head = {
            "schema_version": 1,
            "authority": identity["authority"],
            "sequence": 0,
            "event_sha256": None,
        }
        _atomic_json(
            state_directory / "head.json",
            {"payload": empty_head, "signature": _signature(private_key, empty_head)},
        )
        empty_witness_head = {
            "schema_version": 1,
            "witness": witness_identity["witness"],
            "authority": identity["authority"],
            "sequence": 0,
            "event_sha256": None,
        }
        _atomic_json(
            witness_directory / "head.json",
            {
                "payload": empty_witness_head,
                "signature": _signature(witness_private_key, empty_witness_head),
            },
        )
        return cls(state_directory, witness_directory)

    def _validated_ledger(self) -> list[dict[str, Any]]:
        ledger = _read_json(self.state_directory / "ledger.json", "authority ledger")
        if not isinstance(ledger, list):
            raise TimeAuthorityError("authority ledger has an unsupported schema")
        previous_sha256: str | None = None
        for index, envelope in enumerate(ledger, start=1):
            if not isinstance(envelope, dict) or set(envelope) != {"payload", "signature"}:
                raise TimeAuthorityError("authority ledger event is malformed")
            payload = envelope.get("payload")
            if (
                not isinstance(payload, dict)
                or payload.get("authority") != self.identity["authority"]
                or payload.get("sequence") != index
                or payload.get("previous_event_sha256") != previous_sha256
            ):
                raise TimeAuthorityError("authority ledger chain is invalid")
            _verify_signature(self.public_key, envelope.get("signature"), payload)
            previous_sha256 = _sha256(_canonical_bytes(envelope))
        event_hashes = [None, *(_sha256(_canonical_bytes(item)) for item in ledger)]
        head = _read_json(self.state_directory / "head.json", "authority durable head")
        if not isinstance(head, dict) or set(head) != {"payload", "signature"}:
            raise TimeAuthorityError("authority durable head is invalid")
        _verify_signature(self.public_key, head.get("signature"), head.get("payload"))
        head_payload = head.get("payload")
        if not isinstance(head_payload, dict):
            raise TimeAuthorityError("authority durable head is invalid")
        head_sequence = head_payload.get("sequence")
        if type(head_sequence) is not int or not 0 <= head_sequence <= len(ledger):
            raise TimeAuthorityError("authority ledger rollback or incomplete commit detected")
        expected_head = {
            "schema_version": 1,
            "authority": self.identity["authority"],
            "sequence": head_sequence,
            "event_sha256": event_hashes[head_sequence],
        }
        if head_payload != expected_head:
            raise TimeAuthorityError("authority ledger rollback or incomplete commit detected")

        witness_head = _read_json(
            self.witness_directory / "head.json", "authority monotonic witness head"
        )
        if not isinstance(witness_head, dict) or set(witness_head) != {"payload", "signature"}:
            raise TimeAuthorityError("authority monotonic witness head is invalid")
        _verify_signature(
            self.witness_public_key,
            witness_head.get("signature"),
            witness_head.get("payload"),
        )
        witness_payload = witness_head.get("payload")
        if not isinstance(witness_payload, dict):
            raise TimeAuthorityError("authority monotonic witness head is invalid")
        witness_sequence = witness_payload.get("sequence")
        if type(witness_sequence) is not int or not 0 <= witness_sequence <= len(ledger):
            raise TimeAuthorityError("authority monotonic witness detected ledger rollback")
        expected_witness = {
            "schema_version": 1,
            "witness": self.witness_identity["witness"],
            "authority": self.identity["authority"],
            "sequence": witness_sequence,
            "event_sha256": event_hashes[witness_sequence],
        }
        if witness_payload != expected_witness:
            raise TimeAuthorityError("authority monotonic witness detected ledger rollback")

        ledger_sequence = len(ledger)
        if head_sequence == ledger_sequence and witness_sequence == ledger_sequence:
            return ledger
        if head_sequence != ledger_sequence - 1 or witness_sequence not in {
            ledger_sequence - 1,
            ledger_sequence,
        }:
            raise TimeAuthorityError("authority ledger rollback or incomplete commit detected")
        # A signed final ledger event is the transaction intent. Complete either durable
        # boundary left behind by an interrupted append instead of stranding valid state.
        if witness_sequence == ledger_sequence - 1:
            self._write_witness_head(ledger_sequence, event_hashes[ledger_sequence])
        self._write_authority_head(ledger_sequence, event_hashes[ledger_sequence])
        return ledger

    def _write_authority_head(self, sequence: int, event_sha256: str | None) -> None:
        payload = {
            "schema_version": 1,
            "authority": self.identity["authority"],
            "sequence": sequence,
            "event_sha256": event_sha256,
        }
        _atomic_json(
            self.state_directory / "head.json",
            {"payload": payload, "signature": _signature(self.private_key, payload)},
        )

    def _write_witness_head(self, sequence: int, event_sha256: str | None) -> None:
        payload = {
            "schema_version": 1,
            "witness": self.witness_identity["witness"],
            "authority": self.identity["authority"],
            "sequence": sequence,
            "event_sha256": event_sha256,
        }
        _atomic_json(
            self.witness_directory / "head.json",
            {
                "payload": payload,
                "signature": _signature(self.witness_private_key, payload),
            },
        )

    @staticmethod
    def _test_interrupt_after(step: str) -> None:
        if os.environ.get("GRADOOM_TIME_AUTHORITY_TEST_INTERRUPT_AFTER") == step:
            os._exit(91)

    def _append(self, event_type: str, fields: dict[str, Any]) -> dict[str, Any]:
        previous = None if not self.ledger else _sha256(_canonical_bytes(self.ledger[-1]))
        payload = {
            "schema_version": 1,
            "authority": self.identity["authority"],
            "sequence": len(self.ledger) + 1,
            "previous_event_sha256": previous,
            "event_id": os.urandom(16).hex(),
            "occurred_unix_ns": time.time_ns(),
            "event_type": event_type,
            **fields,
        }
        envelope = {"payload": payload, "signature": _signature(self.private_key, payload)}
        updated = [*self.ledger, envelope]
        _atomic_json(self.state_directory / "ledger.json", updated)
        self._test_interrupt_after("ledger")
        event_sha256 = _sha256(_canonical_bytes(envelope))
        self._write_witness_head(payload["sequence"], event_sha256)
        self._test_interrupt_after("witness")
        self._write_authority_head(payload["sequence"], event_sha256)
        self._test_interrupt_after("head")
        self.ledger = updated
        return payload

    def start_attempt(self, seed: int) -> dict[str, Any]:
        if type(seed) is not int or seed < 0:
            raise TimeAuthorityError("seed must be a non-negative integer")
        started_unix_ns = time.time_ns()
        payload = {
            "schema_version": 1,
            "authority": self.identity["authority"],
            "seed": seed,
            "started_unix_ns": started_unix_ns,
        }
        self._append(
            "attempt_started",
            {"anchor_sha256": _sha256(_canonical_bytes(payload)), **payload},
        )
        return {
            "payload": payload,
            "public_key": self.identity["public_key"],
            "signature": _signature(self.private_key, payload),
        }

    def _start_event(self, seed: int, started_unix_ns: int) -> dict[str, Any]:
        expected = _sha256(
            _canonical_bytes(
                {
                    "schema_version": 1,
                    "authority": self.identity["authority"],
                    "seed": seed,
                    "started_unix_ns": started_unix_ns,
                }
            )
        )
        matches = [
            item["payload"]
            for item in self.ledger
            if item["payload"].get("event_type") == "attempt_started"
            and item["payload"].get("anchor_sha256") == expected
        ]
        if len(matches) != 1:
            raise TimeAuthorityError("attempt anchor is not registered in the persistent ledger")
        return matches[0]

    def sign_journal_head(self, request: dict[str, Any]) -> dict[str, Any]:
        required = {
            "schema_version",
            "authority",
            "seed",
            "started_unix_ns",
            "generation",
            "previous_journal_sha256",
            "journal_sha256",
            "status",
            "prior_reusable_elapsed_seconds",
            "minimum_reusable_elapsed_seconds",
        }
        if set(request) != required or request.get("authority") != self.identity["authority"]:
            raise TimeAuthorityError("journal-head request has an unsupported authority contract")
        self._start_event(request["seed"], request["started_unix_ns"])
        for field in ("prior_reusable_elapsed_seconds", "minimum_reusable_elapsed_seconds"):
            value = request[field]
            if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
                raise TimeAuthorityError(f"{field} must be finite and non-negative")
        prior_seals = [
            item["payload"]
            for item in self.ledger
            if item["payload"].get("event_type") == "journal_sealed"
            and item["payload"].get("seed") == request["seed"]
            and item["payload"].get("started_unix_ns") == request["started_unix_ns"]
        ]
        if prior_seals:
            latest = prior_seals[-1]["attestation_payload"]
            same_head = (
                request["generation"] == latest["generation"]
                and request["journal_sha256"] == latest["journal_sha256"]
                and request["previous_journal_sha256"] == latest["previous_journal_sha256"]
                and request["status"] == latest["status"]
            )
            next_head = (
                request["generation"] == latest["generation"] + 1
                and request["previous_journal_sha256"] == latest["journal_sha256"]
            )
            if not (same_head or next_head):
                raise TimeAuthorityError("journal generation is stale or breaks continuity")
            if float(request["prior_reusable_elapsed_seconds"]) < float(
                latest["reusable_elapsed_seconds"]
            ):
                raise TimeAuthorityError("journal elapsed time attempts to move backwards")
        elif request["generation"] != 0 or request["previous_journal_sha256"] is not None:
            raise TimeAuthorityError("initial journal generation is invalid")
        elapsed = max(
            float(request["prior_reusable_elapsed_seconds"]),
            float(request["minimum_reusable_elapsed_seconds"]),
        )
        attestation_payload = {
            key: request[key]
            for key in (
                "schema_version",
                "authority",
                "seed",
                "started_unix_ns",
                "generation",
                "previous_journal_sha256",
                "journal_sha256",
                "status",
            )
        }
        attestation_payload["reusable_elapsed_seconds"] = elapsed
        self._append(
            "journal_sealed",
            {
                "attestation_payload": attestation_payload,
                **{"seed": request["seed"], "started_unix_ns": request["started_unix_ns"]},
            },
        )
        return {
            "payload": attestation_payload,
            "signature": _signature(self.private_key, attestation_payload),
        }

    def verify_latest_journal_head(self, attestation: dict[str, Any]) -> None:
        payload = attestation.get("payload")
        if not isinstance(payload, dict) or payload.get("authority") != self.identity["authority"]:
            raise TimeAuthorityError("journal attestation has the wrong authority identity")
        _verify_signature(self.public_key, attestation.get("signature"), payload)
        matches = [
            item["payload"]["attestation_payload"]
            for item in self.ledger
            if item["payload"].get("event_type") == "journal_sealed"
            and item["payload"].get("seed") == payload.get("seed")
            and item["payload"].get("started_unix_ns") == payload.get("started_unix_ns")
        ]
        if not matches or matches[-1] != payload:
            raise TimeAuthorityError("stale journal head is not the latest durable seal")

    def recover_latest_journal_head(self, attestation: dict[str, Any]) -> dict[str, Any]:
        """Upgrade a durable report only when resealing preserved its exact journal head."""
        payload = attestation.get("payload")
        if not isinstance(payload, dict) or payload.get("authority") != self.identity["authority"]:
            raise TimeAuthorityError("journal attestation has the wrong authority identity")
        _verify_signature(self.public_key, attestation.get("signature"), payload)
        matches = [
            item["payload"]["attestation_payload"]
            for item in self.ledger
            if item["payload"].get("event_type") == "journal_sealed"
            and item["payload"].get("seed") == payload.get("seed")
            and item["payload"].get("started_unix_ns") == payload.get("started_unix_ns")
        ]
        if payload not in matches:
            raise TimeAuthorityError("journal attestation is not durably recorded")
        latest = matches[-1]
        head_fields = {
            "schema_version",
            "authority",
            "seed",
            "started_unix_ns",
            "generation",
            "previous_journal_sha256",
            "journal_sha256",
            "status",
        }
        if any(payload.get(field) != latest.get(field) for field in head_fields):
            raise TimeAuthorityError("stale journal head cannot recover a different generation")
        return {
            "payload": latest,
            "signature": _signature(self.private_key, latest),
        }

    @staticmethod
    def _artifact_state(path_value: object) -> tuple[Path, dict[str, Any], str]:
        if not isinstance(path_value, str) or not path_value.strip():
            raise TimeAuthorityError("artifact_path must be a non-empty path")
        path = Path(path_value).resolve()
        try:
            metadata = path.lstat()
            payload = path.read_bytes()
        except OSError as error:
            raise TimeAuthorityError("bootstrap artifact is missing or unreadable") from error
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o222
        ):
            raise TimeAuthorityError("bootstrap artifact must be one immutable regular object")
        identity = {
            "resolved_path": str(path),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
        }
        return path, identity, _sha256(payload)

    @staticmethod
    def _bootstrap_claims(request: dict[str, Any]) -> dict[str, Any]:
        required = {
            "artifact_name",
            "artifact_path",
            "creation_elapsed_seconds",
            "creation_protocol",
            "immutable_inputs",
            "reuse_conditions",
        }
        if set(request) != required:
            raise TimeAuthorityError("bootstrap request has an unsupported contract")
        if not isinstance(request["artifact_name"], str) or not request["artifact_name"]:
            raise TimeAuthorityError("artifact_name must be a non-empty string")
        elapsed = request["creation_elapsed_seconds"]
        if type(elapsed) not in (int, float) or not math.isfinite(float(elapsed)) or elapsed < 0:
            raise TimeAuthorityError("creation_elapsed_seconds must be finite and non-negative")
        return {
            "artifact_name": request["artifact_name"],
            "creation_elapsed_seconds": float(elapsed),
            "creation_protocol": request["creation_protocol"],
            "immutable_inputs": request["immutable_inputs"],
            "reuse_conditions": request["reuse_conditions"],
        }

    def create_bootstrap(self, request: dict[str, Any]) -> dict[str, Any]:
        required = {
            "artifact_name",
            "artifact_path",
            "creation_protocol",
            "immutable_inputs",
            "reuse_conditions",
        }
        if set(request) != required:
            raise TimeAuthorityError("bootstrap creation request has an unsupported contract")
        if request["creation_protocol"] != "canonical-declared-input-binding-v1":
            raise TimeAuthorityError("bootstrap creation protocol is not repository-owned")
        if request["reuse_conditions"] != [
            "exact compiler and target identity",
            "read-only bytes reused without transformation",
        ]:
            raise TimeAuthorityError("bootstrap reuse conditions are not canonical")
        if not isinstance(request["immutable_inputs"], list) or not request["immutable_inputs"]:
            raise TimeAuthorityError("bootstrap immutable_inputs must be a non-empty array")
        artifact_path = Path(request["artifact_path"]).resolve()
        payload = (
            json.dumps(
                {
                    "contract": "gradoom-declarative-bootstrap-v1",
                    "immutable_inputs": request["immutable_inputs"],
                    "protocol": request["creation_protocol"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        started = time.perf_counter()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(artifact_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise TimeAuthorityError(
                "bootstrap creation refuses to replace an existing path"
            ) from error
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            artifact_path.chmod(0o444)
            with artifact_path.open("rb") as stream:
                os.fsync(stream.fileno())
            directory_descriptor = os.open(artifact_path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except BaseException:
            artifact_path.unlink(missing_ok=True)
            raise
        creation_elapsed = time.perf_counter() - started
        claims = self._bootstrap_claims({**request, "creation_elapsed_seconds": creation_elapsed})
        _path, identity, digest = self._artifact_state(str(artifact_path))
        if any(
            item["payload"].get("event_type") == "bootstrap_created"
            and item["payload"].get("artifact_name") == claims["artifact_name"]
            for item in self.ledger
        ):
            raise TimeAuthorityError("bootstrap creation is already recorded")
        self._append(
            "bootstrap_created",
            {**claims, "artifact_identity": identity, "artifact_sha256": digest},
        )
        return {**claims, "artifact_path": str(artifact_path)}

    def _bootstrap_creation(self, artifact_name: str) -> dict[str, Any]:
        matches = [
            item["payload"]
            for item in self.ledger
            if item["payload"].get("event_type") == "bootstrap_created"
            and item["payload"].get("artifact_name") == artifact_name
        ]
        if len(matches) != 1:
            raise TimeAuthorityError("bootstrap creation is not uniquely recorded")
        return matches[0]

    def record_bootstrap_reuse(self, request: dict[str, Any]) -> dict[str, Any]:
        claims = self._bootstrap_claims(request)
        creation = self._bootstrap_creation(claims["artifact_name"])
        _path, identity, digest = self._artifact_state(request["artifact_path"])
        if identity != creation["artifact_identity"]:
            raise TimeAuthorityError("bootstrap artifact object identity changed after creation")
        if digest != creation["artifact_sha256"] or any(
            claims[key] != creation[key] for key in claims
        ):
            raise TimeAuthorityError("bootstrap artifact or creation claims changed before reuse")
        return self._append(
            "bootstrap_reused",
            {
                "artifact_name": claims["artifact_name"],
                "artifact_identity": identity,
                "artifact_sha256": digest,
                "creation_event_id": creation["event_id"],
            },
        )

    def attest_bootstrap_reuse(self, request: dict[str, Any]) -> dict[str, Any]:
        claims = self._bootstrap_claims(request)
        creation = self._bootstrap_creation(claims["artifact_name"])
        _path, identity, digest = self._artifact_state(request["artifact_path"])
        reuses = [
            item["payload"]
            for item in self.ledger
            if item["payload"].get("event_type") == "bootstrap_reused"
            and item["payload"].get("creation_event_id") == creation["event_id"]
        ]
        if not reuses:
            raise TimeAuthorityError("bootstrap artifact has no distinct prior reuse event")
        reuse = reuses[-1]
        if (
            reuse["sequence"] <= creation["sequence"]
            or identity != creation["artifact_identity"]
            or digest != creation["artifact_sha256"]
        ):
            raise TimeAuthorityError("bootstrap reuse history no longer matches the artifact")
        payload = {
            "schema_version": 1,
            "authority": self.identity["authority"],
            "artifact_name": claims["artifact_name"],
            "artifact_sha256": digest,
            "creation_elapsed_seconds": claims["creation_elapsed_seconds"],
            "creation_protocol": claims["creation_protocol"],
            "immutable_inputs": claims["immutable_inputs"],
            "reuse_conditions": claims["reuse_conditions"],
            "artifact_identity": identity,
            "creation_event": {
                "event_id": creation["event_id"],
                "sequence": creation["sequence"],
                "artifact_sha256": digest,
            },
            "prior_reuse_event": {
                "event_id": reuse["event_id"],
                "sequence": reuse["sequence"],
                "artifact_sha256": digest,
                "artifact_identity": identity,
            },
        }
        return {
            "payload": payload,
            "public_key": self.identity["public_key"],
            "signature": _signature(self.private_key, payload),
        }

    def verify_bootstrap_reuse(self, attestation: dict[str, Any]) -> None:
        payload = attestation.get("payload")
        if not isinstance(payload, dict) or payload.get("authority") != self.identity["authority"]:
            raise TimeAuthorityError("bootstrap attestation has the wrong authority identity")
        if attestation.get("public_key") != self.identity["public_key"]:
            raise TimeAuthorityError("bootstrap attestation has the wrong authority identity")
        _verify_signature(self.public_key, attestation.get("signature"), payload)
        creation = payload.get("creation_event")
        reuse = payload.get("prior_reuse_event")
        if not isinstance(creation, dict) or not isinstance(reuse, dict):
            raise TimeAuthorityError("bootstrap attestation has incomplete reuse history")
        events = {item["payload"]["event_id"]: item["payload"] for item in self.ledger}
        stored_creation = events.get(creation.get("event_id"))
        stored_reuse = events.get(reuse.get("event_id"))
        if (
            stored_creation is None
            or stored_reuse is None
            or stored_creation.get("event_type") != "bootstrap_created"
            or stored_reuse.get("event_type") != "bootstrap_reused"
            or stored_reuse.get("creation_event_id") != stored_creation.get("event_id")
            or stored_creation.get("sequence", 0) >= stored_reuse.get("sequence", 0)
            or stored_creation.get("artifact_identity") != payload.get("artifact_identity")
            or stored_reuse.get("artifact_identity") != payload.get("artifact_identity")
            or stored_creation.get("artifact_sha256") != payload.get("artifact_sha256")
            or stored_reuse.get("artifact_sha256") != payload.get("artifact_sha256")
        ):
            raise TimeAuthorityError(
                "bootstrap attestation is not backed by chronological ledger history"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gradoom-time-authority")
    parser.add_argument("--state-directory", type=Path, required=True)
    parser.add_argument(
        "--witness-directory",
        type=Path,
        help=(
            "independently retained monotonic witness directory; defaults to a sibling "
            "directory for development use"
        ),
    )
    parser.add_argument(
        "operation",
        choices=(
            "init",
            "identity",
            "start-attempt",
            "sign-journal-head",
            "verify-latest-journal-head",
            "recover-latest-journal-head",
            "create-bootstrap",
            "record-bootstrap-reuse",
            "attest-bootstrap-reuse",
            "verify-bootstrap-reuse",
        ),
    )
    return parser


def _stdin_object() -> dict[str, Any]:
    try:
        value = json.load(sys.stdin)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TimeAuthorityError("operation input must be a JSON object") from error
    if not isinstance(value, dict):
        raise TimeAuthorityError("operation input must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "init":
            authority = ReusableTimeAuthority.initialize(
                args.state_directory, args.witness_directory
            )
            result: object = authority.identity
        else:
            authority = ReusableTimeAuthority(args.state_directory, args.witness_directory)
            if args.operation == "identity":
                result = authority.identity
            elif args.operation == "start-attempt":
                result = authority.start_attempt(_stdin_object().get("seed"))
            elif args.operation == "sign-journal-head":
                result = authority.sign_journal_head(_stdin_object())
            elif args.operation == "verify-latest-journal-head":
                authority.verify_latest_journal_head(_stdin_object())
                result = {"status": "latest"}
            elif args.operation == "recover-latest-journal-head":
                result = authority.recover_latest_journal_head(_stdin_object())
            elif args.operation == "create-bootstrap":
                result = authority.create_bootstrap(_stdin_object())
            elif args.operation == "record-bootstrap-reuse":
                result = authority.record_bootstrap_reuse(_stdin_object())
            elif args.operation == "attest-bootstrap-reuse":
                result = authority.attest_bootstrap_reuse(_stdin_object())
            else:
                authority.verify_bootstrap_reuse(_stdin_object())
                result = {"status": "reused"}
        print(json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":")))
    except (OSError, TimeAuthorityError) as error:
        print(f"gradoom-time-authority: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
