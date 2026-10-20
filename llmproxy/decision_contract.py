"""Internal decision adapter contract, independent of HTTP frontend formats.

Adapters own backend payloads and answer semantics. The service owns transport,
limits, authentication, billing and metrics. Neither layer performs inference.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class DecisionCapabilities:
    question_types: frozenset[str]
    streaming: bool = False


@dataclass(frozen=True)
class DecisionUsage:
    """Actual backend token counts, never estimates from serialized output."""

    input_tokens: int
    output_tokens: int

    def __post_init__(self):
        for value in (self.input_tokens, self.output_tokens):
            if type(value) is not int or not 0 <= value <= 2 ** 63 - 1:
                raise ValueError("invalid_usage")


@dataclass(frozen=True)
class DecisionResult:
    model: str
    answers: Mapping[str, Any]
    usage: DecisionUsage
    native: Mapping[str, Any]


class PreparedDecision(Protocol):
    """Per-request state; decoding must validate answers against this request.

    ValueError denotes invalid backend data. Frontends may use the normalized
    result or preserve the native response, but must bill only validated usage.
    """

    @property
    def body(self) -> dict[str, Any]: ...

    @property
    def question_count(self) -> int: ...

    @property
    def branches(self) -> int: ...

    def decode(self, data: Any) -> DecisionResult: ...


class DecisionAdapter(Protocol):
    path: str
    capabilities: DecisionCapabilities

    def prepare(self, body: dict[str, Any], limits: Mapping[str, int]) -> PreparedDecision:
        """Return owned backend payload/state, or raise ValueError for bad input."""
        ...
