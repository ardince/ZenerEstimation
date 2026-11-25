"""M20-B2: local, workflow-level authorization for bounded optimization.

Not a security boundary against arbitrary code executing in this Python process.
No optimizer, model, dataset, or holdout is accessed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import re
import secrets
from threading import RLock
from typing import Callable

CONTRACT_VERSION = "M20_OPT_EXEC_V1"
SCOPE = "internal_development_optimization"
_STRATEGIES = frozenset({"grid", "random", "bayesian"})
_MODELS = frozenset({"ARIMA", "Kalman", "LSTM", "GRU", "LinearTrendLSTM", "KalmanLSTM"})
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _identifier(value: object, name: str) -> None:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical nonempty identifier")


def _digest(value: object, name: str) -> None:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical sha256 digest")


def _positive_int(value: object, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class OptimizationExecutionRequest:
    dataset_id: str
    development_fingerprint: str
    development_cutoff: str
    model_family: str
    strategy: str
    max_trials: int
    max_epoch_folds: int | None
    validation_protocol_digest: str
    contract_version: str = CONTRACT_VERSION
    scope: str = SCOPE

    def __post_init__(self) -> None:
        if self.contract_version != CONTRACT_VERSION or self.scope != SCOPE:
            raise ValueError("unsupported contract version or authorization scope")
        _identifier(self.dataset_id, "dataset_id")
        _digest(self.development_fingerprint, "development_fingerprint")
        _digest(self.validation_protocol_digest, "validation_protocol_digest")
        if type(self.development_cutoff) is not str:
            raise ValueError("development_cutoff must be YYYY-MM-DD")
        try:
            parsed = date.fromisoformat(self.development_cutoff)
        except ValueError as exc:
            raise ValueError("invalid development_cutoff") from exc
        if parsed.isoformat() != self.development_cutoff:
            raise ValueError("development_cutoff must be canonical YYYY-MM-DD")
        if self.model_family not in _MODELS or type(self.model_family) is not str:
            raise ValueError("unsupported model_family")
        if self.strategy not in _STRATEGIES or type(self.strategy) is not str:
            raise ValueError("unsupported strategy")
        _positive_int(self.max_trials, "max_trials")
        if self.model_family in {"LSTM", "GRU", "LinearTrendLSTM", "KalmanLSTM"}:
            _positive_int(self.max_epoch_folds, "max_epoch_folds")
        elif self.max_epoch_folds is not None:
            _positive_int(self.max_epoch_folds, "max_epoch_folds")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "contract_version", "scope", "dataset_id", "development_fingerprint",
            "development_cutoff", "model_family", "strategy", "max_trials",
            "max_epoch_folds", "validation_protocol_digest",
        )}

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
        return "sha256:" + sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class OptimizationExecutionGrant:
    """Opaque handle; copying/serializing its fields does not confer authority."""
    grant_id: str
    authority_id: str
    request_digest: str
    expires_at: datetime | None
    _proof: str = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, object]:
        """Audit-only representation, intentionally omitting the private proof."""
        return {
            "grant_id": self.grant_id,
            "authority_id": self.authority_id,
            "request_digest": self.request_digest,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


class OptimizationAuthorizationAuthority:
    """Process-local grant registry; use through a trusted local coordinator."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._authority_id = secrets.token_hex(16)
        self._lock = RLock()
        self._records: dict[str, dict[str, object]] = {}
        self.__issue_capability = object()

    def _now(self) -> datetime:
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock must return timezone-aware datetime")
        return now

    def _issue(self, request: OptimizationExecutionRequest, *, capability: object,
               expires_at: datetime | None = None) -> OptimizationExecutionGrant:
        if capability is not self.__issue_capability:
            raise PermissionError("issuance requires local coordinator approval")
        if type(request) is not OptimizationExecutionRequest:
            raise TypeError("request must be OptimizationExecutionRequest")
        if expires_at is not None:
            if not isinstance(expires_at, datetime) or expires_at.tzinfo is None or expires_at.utcoffset() is None:
                raise ValueError("expires_at must be timezone-aware")
            if expires_at <= self._now():
                raise ValueError("expires_at must be in the future")
        with self._lock:
            grant_id = secrets.token_hex(16)
            proof = secrets.token_hex(32)
            grant = OptimizationExecutionGrant(grant_id, self._authority_id,
                                               request.digest, expires_at, proof)
            self._records[grant_id] = {"grant": grant, "state": "issued"}
            return grant

    def _record(self, grant: OptimizationExecutionGrant,
                request: OptimizationExecutionRequest) -> dict[str, object]:
        if type(grant) is not OptimizationExecutionGrant or type(request) is not OptimizationExecutionRequest:
            raise PermissionError("invalid authorization inputs")
        record = self._records.get(grant.grant_id)
        if record is None or record["grant"] is not grant:
            raise PermissionError("unrecognized grant")
        if grant.authority_id != self._authority_id or grant.request_digest != request.digest:
            raise PermissionError("authorization scope mismatch")
        if record["state"] != "issued":
            raise PermissionError("grant is not active")
        if grant.expires_at is not None and self._now() >= grant.expires_at:
            record["state"] = "expired"
            raise PermissionError("grant expired")
        return record

    def verify(self, grant: OptimizationExecutionGrant,
               request: OptimizationExecutionRequest) -> bool:
        """Diagnostic check only. Never use as permission to start work."""
        with self._lock:
            self._record(grant, request)
            return True

    def consume(self, grant: OptimizationExecutionGrant,
                request: OptimizationExecutionRequest) -> None:
        """Atomically claim permission for one bounded run."""
        with self._lock:
            record = self._record(grant, request)
            record["state"] = "consumed"

    def revoke(self, grant: OptimizationExecutionGrant) -> None:
        with self._lock:
            if type(grant) is not OptimizationExecutionGrant:
                raise PermissionError("invalid grant")
            record = self._records.get(grant.grant_id)
            if record is None or record["grant"] is not grant:
                raise PermissionError("unrecognized grant")
            if record["state"] != "issued":
                raise PermissionError("only active grants can be revoked")
            record["state"] = "revoked"


class LocalOptimizationCoordinator:
    """Explicit local approval workflow; not identity authentication.

    The application must restrict access to this coordinator. Its approve()
    method represents the researcher's explicit local approval decision.
    """

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self.__authority = OptimizationAuthorizationAuthority(clock=clock)
        # Store the capability only inside this local workflow object.
        self.__capability = self.__authority._OptimizationAuthorizationAuthority__issue_capability

    def approve(self, request: OptimizationExecutionRequest, *,
                expires_at: datetime | None = None) -> OptimizationExecutionGrant:
        return self.__authority._issue(request, capability=self.__capability,
                                       expires_at=expires_at)

    def verify(self, grant: OptimizationExecutionGrant,
               request: OptimizationExecutionRequest) -> bool:
        return self.__authority.verify(grant, request)

    def consume(self, grant: OptimizationExecutionGrant,
                request: OptimizationExecutionRequest) -> None:
        self.__authority.consume(grant, request)

    def revoke(self, grant: OptimizationExecutionGrant) -> None:
        self.__authority.revoke(grant)
