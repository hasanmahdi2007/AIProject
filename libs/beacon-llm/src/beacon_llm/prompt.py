"""The cache-ordered prompt builder.

One patient means roughly twenty per-trial rule-check calls. Those twenty
prompts share two large blocks - the standing instructions and the numbered
patient note - and differ only in the trial's eligibility rules. Put the shared
part first and mark a cache breakpoint after it and nineteen of the twenty
calls read that prefix from cache at a tenth (Haiku) or a twentieth
(Opus/Sonnet) of the input price. Put the trial rules first instead and the
prefix is no longer byte-identical, every call pays full price, and **nothing
fails**: there is no error, no warning, just a bill five to twenty times
larger.

So the ordering here is structural, not a convention. :class:`CachedPrompt`
stores three *strings* and renders them in a fixed order. There is no
constructor that takes a list of blocks, no parameter that reorders them, and
no way for a caller to put the variable text in the cached prefix. The only
thing a caller chooses is what goes in each of the three slots.

Layout produced:

===========================  ==============================  ===============
Slot                         Where it lands                  Cached?
===========================  ==============================  ===============
``instructions`` (stable)    ``system[0]``                   yes
``note`` (per patient)       ``system[1]`` + breakpoint      yes, up to here
``variable`` (per trial)     the user turn                   no
===========================  ==============================  ===============

Note that both cached blocks live in ``system`` and the variable block is the
user turn. That is the second half of the guarantee: the cached prefix and the
variable remainder are not merely ordered, they are in different fields of the
request, so appending to the conversation cannot accidentally land inside the
cached region.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from anthropic.types import MessageParam, TextBlockParam

from beacon_core.schemas import NumberedNote
from beacon_llm.pricing import MIN_CACHEABLE_PREFIX_TOKENS, CacheTtl

__all__ = [
    "APPROX_CHARS_PER_TOKEN",
    "CachedPrompt",
    "build_cached_prompt",
    "render_numbered_note",
]

APPROX_CHARS_PER_TOKEN = 4
"""Offline rule of thumb, used only where an exact count is unavailable.

The live path counts tokens with the API's own counter. This constant exists so
that the replay path, which must make zero network calls, can still apply the
prompt-token guard and can still warn about a prefix too short to be cacheable.
"""


def render_numbered_note(note: NumberedNote) -> str:
    """Render a note as ``"<id>. <text>"`` lines.

    Rendered here and nowhere else, because the sentence IDs a
    ``CriterionJudgment`` cites must be the IDs the model actually saw. A second
    renderer elsewhere that numbered from zero, or dropped a blank sentence,
    would produce citations that fail verification for reasons no prompt change
    explains.
    """
    return "\n".join(f"{s.sentence_id}. {s.text}" for s in note.sentences)


@dataclass(frozen=True, slots=True)
class CachedPrompt:
    """A prompt split into a cacheable prefix and a variable remainder.

    Build with :func:`build_cached_prompt`; the three fields are plain text and
    the request shape is derived, so the block order cannot be got wrong.
    """

    instructions: str
    note_text: str
    variable: str
    cache_ttl: CacheTtl = "5m"

    @property
    def cacheable_prefix(self) -> str:
        """Exactly the text the cache breakpoint covers.

        This is what must be byte-identical across all twenty per-trial calls
        for one patient.
        """
        return f"{self.instructions}\n\n{self.note_text}"

    @property
    def prefix_hash(self) -> str:
        """Fingerprint of the cacheable prefix alone.

        Two prompts for the same patient and different trials must share this
        hash; ``test_prompt`` asserts it over twenty generated prompts, which is
        the cheapest available proxy for "the cache will actually hit".
        """
        return hashlib.sha256(self.cacheable_prefix.encode("utf-8")).hexdigest()

    @property
    def estimated_prefix_tokens(self) -> int:
        """Rough token count of the cacheable prefix.

        Only ever used to warn that the prefix may be under the 512-token
        minimum. Never used for billing.
        """
        return len(self.cacheable_prefix) // APPROX_CHARS_PER_TOKEN

    @property
    def maybe_too_short_to_cache(self) -> bool:
        """True when the prefix is near or below the minimum cacheable size.

        A prefix under 512 tokens takes the ``cache_control`` breakpoint without
        complaint and caches nothing. Deliberately conservative - it flags
        anything within 20% of the floor - because the failure is invisible and
        the warning is cheap.
        """
        return self.estimated_prefix_tokens < int(MIN_CACHEABLE_PREFIX_TOKENS * 1.2)

    def system_blocks(self) -> list[TextBlockParam]:
        """``system`` as two text blocks, the breakpoint on the second.

        Fixed order, derived from the fields. The breakpoint sits on the *note*
        block, so the cached region is instructions + note and stops there.
        """
        instructions: TextBlockParam = {"type": "text", "text": self.instructions}
        note: TextBlockParam = {
            "type": "text",
            "text": self.note_text,
            "cache_control": {"type": "ephemeral", "ttl": self.cache_ttl},
        }
        return [instructions, note]

    def messages(self) -> list[MessageParam]:
        """The user turn: the variable part, and only the variable part.

        No assistant turn is appended. Assistant prefill returns 400 on the
        current models, so the shape that would tempt someone into it is simply
        not available here.
        """
        return [{"role": "user", "content": [{"type": "text", "text": self.variable}]}]

    def fingerprint(self) -> str:
        """Hash of the whole prompt, prefix and variable part together.

        Used as the record/replay fixture key and written to
        ``LlmCall.prompt_hash``, so an audit row and the fixture that served it
        carry the same identifier.
        """
        canonical = json.dumps(
            {
                "instructions": self.instructions,
                "note": self.note_text,
                "variable": self.variable,
                "cache_ttl": self.cache_ttl,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_cached_prompt(
    *,
    instructions: str,
    note: NumberedNote | str,
    variable: str,
    cache_ttl: CacheTtl = "5m",
) -> CachedPrompt:
    """Assemble a cache-ordered prompt.

    :param instructions: Standing instructions. Must not vary between the calls
        made for one patient - anything per-trial belongs in ``variable``.
    :param note: The patient note. A :class:`~beacon_core.schemas.NumberedNote`
        is rendered with :func:`render_numbered_note`; a pre-rendered string is
        accepted for prompts that are not note-shaped (query generation, say).
    :param variable: The per-trial part - eligibility rules, the trial's ID,
        anything that changes call to call.
    :param cache_ttl: ``"5m"`` or ``"1h"``. Twenty calls for one patient finish
        well inside five minutes, so ``"5m"`` is the default; ``"1h"`` only pays
        for itself when the same patient is revisited.
    """
    note_text = note if isinstance(note, str) else render_numbered_note(note)
    if not instructions.strip():
        raise ValueError("instructions must not be empty: it is the stable cached prefix")
    if not note_text.strip():
        raise ValueError("note must not be empty")
    if not variable.strip():
        raise ValueError(
            "variable must not be empty: with nothing per-call there is nothing to vary, "
            "and the prompt belongs in a single uncached call instead"
        )
    return CachedPrompt(
        instructions=instructions,
        note_text=note_text,
        variable=variable,
        cache_ttl=cache_ttl,
    )
