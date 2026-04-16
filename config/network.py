"""Unified network configuration helpers for REST, WS, and webhook paths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Retry policy for transient network operations."""

    max_retries: int
    base_delay_seconds: float


@dataclass(frozen=True, slots=True)
class RestNetworkConfig:
    """REST transport settings."""

    proxy_url: str | None
    timeout_seconds: float
    retry: RetryConfig


@dataclass(frozen=True, slots=True)
class WsNetworkConfig:
    """WebSocket transport settings."""

    proxy_url: str | None
    connect_timeout_seconds: float
    receive_timeout_seconds: float
    idle_timeout_seconds: float
    reconnect_base_delay_seconds: float
    reconnect_max_delay_seconds: float


@dataclass(frozen=True, slots=True)
class WebhookNetworkConfig:
    """Webhook transport settings."""

    proxy_url: str | None
    timeout_seconds: float
    retry: RetryConfig


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    """Unified network settings for all transport lanes."""

    rest: RestNetworkConfig
    ws: WsNetworkConfig
    webhook: WebhookNetworkConfig


def _as_mapping(value: object) -> Mapping[str, Any]:
    """Return mapping view when config section is a dict-like object."""
    if isinstance(value, Mapping):
        return value
    return {}


def _get_proxy_url(config: Mapping[str, Any]) -> str | None:
    """Resolve legacy proxy config into a single proxy URL or None."""
    proxy = _as_mapping(config.get("proxy", {}))
    enabled = bool(proxy.get("enable", False))
    raw_url = str(proxy.get("url", "")).strip()
    if not enabled or not raw_url:
        return None
    return raw_url


def load_network_config(config: Mapping[str, Any]) -> NetworkConfig:
    """Build unified network config with backward-compatible defaults."""
    proxy_url = _get_proxy_url(config)
    network = _as_mapping(config.get("network", {}))
    rest = _as_mapping(network.get("rest", {}))
    ws = _as_mapping(network.get("ws", {}))
    webhook = _as_mapping(network.get("webhook", {}))

    return NetworkConfig(
        rest=RestNetworkConfig(
            proxy_url=proxy_url,
            timeout_seconds=float(rest.get("timeout_seconds", 30.0)),
            retry=RetryConfig(
                max_retries=int(rest.get("max_retries", 2)),
                base_delay_seconds=float(rest.get("base_delay_seconds", 1.0)),
            ),
        ),
        ws=WsNetworkConfig(
            proxy_url=proxy_url,
            connect_timeout_seconds=float(ws.get("connect_timeout_seconds", 65.0)),
            receive_timeout_seconds=float(ws.get("receive_timeout_seconds", 65.0)),
            idle_timeout_seconds=float(ws.get("idle_timeout_seconds", 25.0)),
            reconnect_base_delay_seconds=float(ws.get("reconnect_base_delay_seconds", 2.0)),
            reconnect_max_delay_seconds=float(ws.get("reconnect_max_delay_seconds", 30.0)),
        ),
        webhook=WebhookNetworkConfig(
            proxy_url=proxy_url,
            timeout_seconds=float(webhook.get("timeout_seconds", 10.0)),
            retry=RetryConfig(
                max_retries=int(webhook.get("max_retries", 2)),
                base_delay_seconds=float(webhook.get("base_delay_seconds", 1.0)),
            ),
        ),
    )
