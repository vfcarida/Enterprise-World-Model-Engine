"""Base protocols, authentication contracts, and bitemporal schema mapping for enterprise connectors."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from ewm_engine.core.bitemporal import BitemporalLedger, BitemporalRecord
from ewm_engine.core.state import WorldState


class ConnectorAuthMode(StrEnum):
    """Supported read-only authentication modes for enterprise system connectors."""

    OAUTH2_CLIENT_CREDENTIALS = "oauth2_client_credentials"
    BEARER_TOKEN = "bearer_token"
    BASIC_READONLY = "basic_readonly"
    OFFLINE_SNAPSHOT = "offline_snapshot"


class ConnectorAuthConfig(BaseModel):
    """Read-only authentication configuration enforcing pilot airgap safety."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    auth_mode: ConnectorAuthMode = ConnectorAuthMode.OFFLINE_SNAPSHOT
    endpoint_url: str = Field(
        default="https://sandbox.example.internal",
        description="Base API endpoint URL for the enterprise source system.",
    )
    client_id: str | None = None
    client_secret: SecretStr | None = None
    bearer_token: SecretStr | None = None
    read_only_enforced: bool = Field(
        default=True,
        description="Mandatory read-only airgap guard; write-back is prohibited during shadow pilots.",
    )
    pii_salt: str = Field(
        default="ewm-default-pilot-salt",
        description="Cryptographic salt used to pseudonymize entity identifiers prior to ingestion.",
    )

    @model_validator(mode="after")
    def _enforce_readonly_airgap(self) -> ConnectorAuthConfig:
        if not self.read_only_enforced:
            raise ValueError(
                "ConnectorAuthConfig requires read_only_enforced=True; production write-back is prohibited."
            )
        if self.auth_mode == ConnectorAuthMode.OAUTH2_CLIENT_CREDENTIALS and (
            not self.client_id or self.client_secret is None
        ):
            raise ValueError(
                "OAUTH2_CLIENT_CREDENTIALS requires both client_id and client_secret."
            )
        if self.auth_mode == ConnectorAuthMode.BEARER_TOKEN and self.bearer_token is None:
            raise ValueError("BEARER_TOKEN mode requires bearer_token.")
        return self

    def pseudonymize_id(self, raw_id: str, prefix: str = "ent") -> str:
        """Deterministically hash raw enterprise IDs using salted SHA-256."""
        digest = hashlib.sha256(f"{self.pii_salt}:{raw_id}".encode()).hexdigest()[:12]
        return f"{prefix}_{digest}"


class ConnectorIngestionResult(BaseModel):
    """Auditable result of ingesting enterprise records into a WorldState and BitemporalLedger."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    connector_name: str
    records_ingested: int
    entities_mapped: int
    resources_mapped: int
    relationships_mapped: int
    state_fingerprint: str
    decision_time: float


class EnterpriseDataConnector(ABC):
    """Abstract base class for read-only bitemporal enterprise data connectors."""

    def __init__(self, auth_config: ConnectorAuthConfig | None = None) -> None:
        self._auth = auth_config or ConnectorAuthConfig()
        self._authenticated: bool = False

    @property
    def auth_config(self) -> ConnectorAuthConfig:
        """Active read-only authentication configuration."""
        return self._auth

    @property
    def is_authenticated(self) -> bool:
        """Return True if the connector session has been verified."""
        return self._authenticated

    def authenticate(self) -> bool:
        """Verify read-only credentials or initialize offline snapshot session."""
        if not self._auth.read_only_enforced:
            raise PermissionError("Write-enabled connector sessions are prohibited.")
        self._authenticated = True
        return True

    @abstractmethod
    def ingest_to_ledger(
        self,
        raw_records: Sequence[Mapping[str, Any]],
        ledger: BitemporalLedger,
    ) -> int:
        """Validate and append raw source records into a BitemporalLedger."""

    @abstractmethod
    def map_to_world_state(
        self,
        raw_records: Sequence[Mapping[str, Any]],
        *,
        decision_time: float,
    ) -> tuple[WorldState, BitemporalLedger, ConnectorIngestionResult]:
        """Map source records as of decision_time into a canonical WorldState and BitemporalLedger."""

    @staticmethod
    def _build_bitemporal_record(
        record_id: str,
        stream: str,
        event_time: float,
        knowledge_time: float,
        payload: dict[str, float | int | str | bool],
    ) -> BitemporalRecord:
        return BitemporalRecord(
            record_id=record_id,
            stream=stream,
            event_time=event_time,
            knowledge_time=knowledge_time,
            payload=payload,
        )
