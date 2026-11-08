"""Every failure this library can produce, as a named type.

Each one corresponds to a trap that is otherwise *silent*: a prompt that
quintuples its own price, a fixture that falls through to the network, a model
whose price is unknown so it would be costed at zero. Raising beats logging
here, because the whole point of this package is that cost surprises become
test failures.
"""

from __future__ import annotations

__all__ = [
    "BeaconLlmError",
    "CapabilityError",
    "FixtureMissingError",
    "PromptTooLargeError",
    "RefusalError",
    "StructuredOutputError",
    "UnknownModelPriceError",
]


class BeaconLlmError(Exception):
    """Base class, so a caller can catch everything from this seam at once."""


class PromptTooLargeError(BeaconLlmError):
    """The prompt would cross a pricing cliff, so the call was never made.

    This is the guard that makes a "just batch all 20 trials into one call"
    refactor fail loudly instead of quintupling the bill in silence.
    """

    def __init__(self, *, model: str, prompt_tokens: int, ceiling: int) -> None:
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.ceiling = ceiling
        super().__init__(
            f"{model}: prompt is {prompt_tokens:,} tokens, over the configured ceiling of "
            f"{ceiling:,}. Above this threshold {model} is billed at 5x "
            f"(input and output alike), and the threshold counts cache reads and writes "
            f"as well as fresh input. Split the work across calls instead of batching it; "
            f"raise settings.anthropic.haiku_prompt_token_ceiling only deliberately."
        )


class UnknownModelPriceError(BeaconLlmError):
    """No verified price for this model, so its cost cannot be metered.

    Refusing is deliberate: defaulting to zero would make an expensive model
    look free in the ``llm_calls`` table, which is where every cost number in
    the README comes from.
    """

    def __init__(self, model: str, known: tuple[str, ...]) -> None:
        self.model = model
        self.known = known
        super().__init__(
            f"no verified price table entry for {model!r}; known models are "
            f"{', '.join(known)}. Add a verified row to beacon_llm.pricing.PRICE_TABLE "
            f"rather than letting the call be costed at $0."
        )


class FixtureMissingError(BeaconLlmError):
    """Replay mode was asked for a response it has not recorded.

    Raising here is what guarantees CI makes zero network calls: there is no
    code path from a cache miss to the live API while the mode is ``replay``.
    """

    def __init__(self, *, key: str, path: str) -> None:
        self.key = key
        self.path = path
        super().__init__(
            f"no recorded response for prompt hash {key} (expected at {path}). "
            f"Replay mode never falls through to the network. Re-record with "
            f"BEACON_LLM_MODE=record, or fix the prompt so it matches an existing fixture."
        )


class StructuredOutputError(BeaconLlmError):
    """The response did not validate against the requested Pydantic model."""

    def __init__(self, *, model_name: str, detail: str, raw: str) -> None:
        self.model_name = model_name
        self.detail = detail
        self.raw = raw
        super().__init__(f"response did not validate as {model_name}: {detail}")


class RefusalError(BeaconLlmError):
    """The model declined to answer (``stop_reason == "refusal"``).

    A refusal is a real outcome, not a transport error: retrying the identical
    prompt will refuse again, so it must not be swallowed by retry logic.
    """

    def __init__(self, *, model: str, request_id: str | None) -> None:
        self.model = model
        self.request_id = request_id
        super().__init__(f"{model} refused to answer (request_id={request_id})")


class CapabilityError(BeaconLlmError):
    """The model does not support something the call site asked for.

    Checked against the live Models API ``capabilities`` object rather than a
    hardcoded table, because a hardcoded table is wrong the moment a model ships.
    """
