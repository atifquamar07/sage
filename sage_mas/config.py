"""Configuration for SAGE and for model endpoints.

``configs/sage.yaml`` maps one-to-one onto :class:`SageConfig`; unknown keys are
rejected. Any field can be overridden with ``--set path.to.key=value``.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .prompts.roles import ROLE_POOL, SPECIALTIES


@dataclass
class PrefixConfig:
    tau: float = 0.6  # τ: fraction of whitespace tokens kept as the prefix
    weight: float = 0.5  # λ: weight of prefix consistency in ρ
    temperature: float = 0.6


@dataclass
class ReviewConfig:
    sample_size: int = 2  # m: agents sampled from each score group


@dataclass
class RewriteConfig:
    temperature: float = 0.2
    max_attempts: int = 3
    max_chars: int = 6000
    min_chars: int = 80
    materiality_threshold: float = 0.98  # rewrites at least this similar to the original are rejected


@dataclass
class RoutingConfig:
    top_k: int = 2  # K: maximum number of parents per agent
    max_rounds: int = 3  # T: maximum number of collaboration rounds


@dataclass
class VoteConfig:
    leader_bonus: float = 0.5  # β: extra vote weight of each stage's leader


@dataclass
class SageConfig:
    num_agents: int = 4  # N
    roles: list[str] = field(default_factory=lambda: list(ROLE_POOL))
    backbones: list[str] | None = None  # per-agent model names; default: one model for all
    temperature: float = 0.5  # initial answers, reviews and revisions
    prefix: PrefixConfig = field(default_factory=PrefixConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    rewrite: RewriteConfig = field(default_factory=RewriteConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    vote: VoteConfig = field(default_factory=VoteConfig)
    answer_extractor: str = "xfinder"  # "xfinder" (paper) or "regex"
    max_workers: int = 8  # parallel calls within one question (reviews, rewrites)

    def validate(self) -> SageConfig:
        if self.num_agents < 2:
            raise ValueError("SAGE needs at least two agents")
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("roles must be unique")
        unknown = sorted(set(self.roles) - set(SPECIALTIES))
        if unknown:
            raise ValueError(f"unknown roles: {', '.join(unknown)}")
        if len(self.roles) < self.num_agents:
            raise ValueError(f"need at least {self.num_agents} roles, got {len(self.roles)}")
        if self.backbones is not None and len(self.backbones) != self.num_agents:
            raise ValueError(f"backbones must list exactly {self.num_agents} models")
        if not 0 <= self.routing.top_k < self.num_agents:
            raise ValueError("routing.top_k must be in [0, num_agents - 1]")
        if self.routing.max_rounds < 1:
            raise ValueError("routing.max_rounds must be at least 1")
        if not 0.0 < self.prefix.tau < 1.0:
            raise ValueError("prefix.tau must be in (0, 1)")
        if self.prefix.weight < 0:
            raise ValueError("prefix.weight must be non-negative")
        if self.answer_extractor not in {"xfinder", "regex"}:
            raise ValueError("answer_extractor must be 'xfinder' or 'regex'")
        return self


@dataclass
class ModelConfig:
    """An OpenAI-compatible model endpoint (typically a local vLLM server)."""

    name: str  # model name served by the endpoint
    endpoints: list[str]
    hf_repo: str | None = None
    hf_revision: str | None = None
    api_key_env: str | None = None  # environment variable holding the API key; vLLM needs none
    max_tokens: int = 2048
    timeout: float = 1200.0
    max_concurrency: int = 128  # simultaneous requests per endpoint
    sampling: dict[str, Any] = field(default_factory=dict)  # top_p, top_k, repetition_penalty

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "") if self.api_key_env else "EMPTY"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


_NESTED = {
    "PrefixConfig": PrefixConfig,
    "ReviewConfig": ReviewConfig,
    "RewriteConfig": RewriteConfig,
    "RoutingConfig": RoutingConfig,
    "VoteConfig": VoteConfig,
}


def _build(cls, data: dict, path: str = ""):
    if not isinstance(data, dict):
        raise ValueError(f"{path or cls.__name__} must be a mapping")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ValueError(f"unknown config keys in {path or cls.__name__}: {', '.join(unknown)}")
    kwargs = {}
    for name, value in data.items():
        nested = _NESTED.get(str(fields[name].type))
        kwargs[name] = _build(nested, value, f"{path}{name}.") if nested else value
    return cls(**kwargs)


def _apply_overrides(data: dict, overrides: list[str]) -> dict:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        node = data
        *parents, leaf = dotted.strip().split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = yaml.safe_load(raw)
    return data


def load_sage_config(path: str | Path | None = None, overrides: list[str] | None = None) -> SageConfig:
    data = yaml.safe_load(Path(path).read_text()) if path else {}
    return _build(SageConfig, _apply_overrides(data or {}, overrides or [])).validate()


def load_model_config(path: str | Path, overrides: list[str] | None = None) -> ModelConfig:
    data = _apply_overrides(yaml.safe_load(Path(path).read_text()) or {}, overrides or [])
    env_endpoints = os.environ.get(f"SAGE_ENDPOINTS_{data.get('name', '').upper().replace('-', '_').replace('.', '_')}")
    if env_endpoints:
        data["endpoints"] = [url for url in env_endpoints.replace(",", " ").split() if url]
    return _build(ModelConfig, data)


def config_dict(config: Any) -> dict:
    return dataclasses.asdict(config)
