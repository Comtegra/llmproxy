"""Explicit decision wire protocols. New engines need an adapter, not HTTP changes."""

from . import basal
from .decision_contract import DecisionAdapter


_ADAPTERS: dict[str, DecisionAdapter] = {"basal": basal.ADAPTER}


def get_adapter(backend) -> DecisionAdapter:
    # Existing decision configs used Basal implicitly; retain their behavior.
    name = backend.get("decision_adapter", "basal")
    if not isinstance(name, str) or name not in _ADAPTERS:
        raise ValueError("unsupported decision_adapter")
    return _ADAPTERS[name]


def backend_path(backend):
    return get_adapter(backend).path
