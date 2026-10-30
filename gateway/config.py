"""Gateway configuration, loaded from YAML. See config.example.yaml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    type: Literal["anthropic", "openai", "mock"]
    api_key_env: str | None = None  # name of the env var holding the key; keys never live in the file
    base_url: str | None = None
    timeout_s: float = 120.0
    # mock only
    latency_ms: float = 0
    failure_rate: float = 0.0
    fail_status: int = 503


class Target(BaseModel):
    provider: str
    model: str
    weight: float = 1.0


class Route(BaseModel):
    """An alias clients ask for. Targets with the highest priority group are
    tried first (load-balanced by weight); later groups are fallbacks."""

    primary: list[Target]
    fallbacks: list[Target] = []


class Price(BaseModel):
    input_per_mtok: float
    output_per_mtok: float


class KeyConfig(BaseModel):
    name: str
    requests_per_minute: int = 60
    tokens_per_minute: int = 100_000
    budget_usd: float | None = None  # lifetime spend cap; None = unlimited
    allowed_models: list[str] | None = None


class RetryConfig(BaseModel):
    attempts: int = 2  # per target, including the first try
    base_delay_s: float = 0.25
    max_delay_s: float = 4.0


class BreakerConfig(BaseModel):
    failure_threshold: int = 5  # consecutive failures that open the circuit
    cooldown_s: float = 30.0  # how long it stays open before one trial request


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl_s: float = 300.0
    max_entries: int = 10_000


class Config(BaseModel):
    providers: dict[str, ProviderConfig]
    routes: dict[str, Route]
    pricing: dict[str, Price] = {}
    keys: dict[str, KeyConfig] = Field(default_factory=dict)  # api key -> settings
    keys_env: str | None = None  # optional env var with "key:name,key:name" for secrets kept out of the file
    retry: RetryConfig = RetryConfig()
    breaker: BreakerConfig = BreakerConfig()
    cache: CacheConfig = CacheConfig()
    default_max_tokens: int = 16000

    def resolved_keys(self) -> dict[str, KeyConfig]:
        keys = dict(self.keys)
        if self.keys_env and (raw := os.environ.get(self.keys_env)):
            for pair in raw.split(","):
                k, _, name = pair.partition(":")
                if k:
                    keys[k.strip()] = KeyConfig(name=name.strip() or "env")
        return keys

    def validate_refs(self) -> None:
        for alias, r in self.routes.items():
            for t in r.primary + r.fallbacks:
                if t.provider not in self.providers:
                    raise ValueError(f"route {alias!r} references unknown provider {t.provider!r}")
            if not r.primary:
                raise ValueError(f"route {alias!r} has no primary targets")


def load(path: str | Path) -> Config:
    cfg = Config.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    cfg.validate_refs()
    return cfg
