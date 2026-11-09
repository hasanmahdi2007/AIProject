"""Building the text that gets embedded, and measuring what the model drops.

Field order is the whole design
-------------------------------
``BAAI/bge-small-en-v1.5`` has a 512-token window and its tokenizer is
configured ``{'max_length': 512, 'strategy': 'longest_first', 'direction':
'right'}`` -- it silently discards everything past token 512 from the **right**
and returns a perfectly ordinary-looking vector. There is no warning and no
error, so the only defence is to put the text that distinguishes one trial from
another at the front and let the cut land on the part that does not:

1. **title** -- the single most discriminative line in the record.
2. **conditions** -- the disease terms a patient note is matched on.
3. **interventions** -- the drug and procedure names.
4. **brief summary** -- prose, largely boilerplate, and the only field long
   enough to overflow the window on its own.

``detailed_description`` is deliberately *not* embedded: it is several thousand
tokens of protocol text in many records, so including it would push the summary
out of the window for a large share of the corpus while adding almost nothing a
query matches on. It is still stored in Postgres.

Eligibility text is not here either. The retrieval unit is a trial, and rules
are judged against a specific patient later; embedding them would both blow the
window and make every trial look alike (nearly every trial excludes pregnancy).

Measuring the truncation rate
-----------------------------
``overview_token_count`` is the token count of the **full** text as built,
including the two special tokens, so ``count > 512`` is exactly the truncation
test and the overflow is quantified rather than merely flagged.
``overview_text`` stores the text *after* truncation, so the column means what
``Trial.overview_text`` promises -- "exactly what was embedded" -- and the index
can be rebuilt from Postgres alone.

Truncation happens here, on character offsets taken from the tokenizer, rather
than being left to the model: the stored text and the embedded text then cannot
drift, and the cut lands on a token boundary instead of mid-word.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "CONDITIONS_LABEL",
    "INTERVENTIONS_LABEL",
    "Encoding",
    "Overview",
    "OverviewBuilder",
    "TokenizerLike",
    "build_overview_text",
]

CONDITIONS_LABEL = "Conditions:"
INTERVENTIONS_LABEL = "Interventions:"

_SPECIAL_TOKENS = 2
"""``[CLS]`` and ``[SEP]``, which count against the 512-token window."""


class Encoding(Protocol):
    """The part of a ``tokenizers.Encoding`` this module uses."""

    @property
    def ids(self) -> list[int]: ...

    @property
    def offsets(self) -> list[tuple[int, int]]: ...


@runtime_checkable
class TokenizerLike(Protocol):
    """The part of a ``tokenizers.Tokenizer`` this module uses.

    Structural rather than a concrete import so that ``tokenizers`` stays an
    implementation detail of FastEmbed instead of becoming a declared
    dependency of this service, and so a test can pass a stub.
    """

    def encode(self, text: str) -> Encoding: ...


@dataclass(frozen=True, slots=True)
class Overview:
    """The embedded text for one trial, with its measurement.

    Attributes:
        text: exactly the string handed to the embedder -- truncated when the
            full overview overflowed the window.
        token_count: tokens in the *untruncated* overview, specials included.
        truncated: whether ``token_count`` exceeded the model's window.
        max_tokens: the window the count was judged against.
    """

    text: str
    token_count: int
    truncated: bool
    max_tokens: int

    @property
    def dropped_tokens(self) -> int:
        """Tokens the window could not hold; ``0`` when nothing was cut."""
        return max(0, self.token_count - self.max_tokens)


def build_overview_text(
    title: str,
    conditions: Sequence[str],
    interventions: Sequence[str],
    brief_summary: str | None,
) -> str:
    """Assemble the overview, most discriminative field first.

    Conditions and interventions are labelled and semicolon-joined. The labels
    cost about three tokens each and earn it: without them a bare list of drug
    names sits next to a bare list of diseases with nothing to tell the encoder
    which is which.

    Empty fields are omitted rather than emitted as empty sections, so a record
    with no conditions does not spend window on the word "Conditions".
    """
    parts: list[str] = []
    if title.strip():
        parts.append(" ".join(title.split()))
    if conditions:
        parts.append(f"{CONDITIONS_LABEL} {'; '.join(conditions)}")
    if interventions:
        parts.append(f"{INTERVENTIONS_LABEL} {'; '.join(interventions)}")
    if brief_summary and brief_summary.strip():
        parts.append(" ".join(brief_summary.split()))
    return "\n".join(parts)


class OverviewBuilder:
    """Builds overviews and truncates them at the embedding model's window.

    The tokenizer handed in must have truncation **disabled**, or the full
    token count cannot be observed -- which is the whole point.
    :meth:`from_fastembed` arranges that.
    """

    def __init__(self, tokenizer: TokenizerLike, max_tokens: int = 512) -> None:
        if max_tokens <= _SPECIAL_TOKENS:
            raise ValueError(f"max_tokens must exceed {_SPECIAL_TOKENS}, got {max_tokens}")
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @classmethod
    def from_fastembed(cls, model: Any, max_tokens: int = 512) -> OverviewBuilder:
        """Clone the tokenizer out of a FastEmbed ``TextEmbedding``.

        ``TextEmbedding.model.tokenizer`` is the live tokenizer the encoder
        uses, and it has truncation enabled at 512. Calling ``no_truncation()``
        on it would silently change how every subsequent embedding is produced,
        so it is serialised and re-parsed into an independent instance first --
        through the live object's own class, so ``tokenizers`` need not be
        imported here.

        Raises:
            TypeError: if the object exposes no tokenizer, which would mean
                FastEmbed's internals moved and the truncation rate was about
                to be estimated instead of measured.
        """
        inner = getattr(model, "model", model)
        tokenizer = getattr(inner, "tokenizer", None)
        if tokenizer is None:
            raise TypeError(
                f"{type(model).__name__} exposes no tokenizer; the 512-token "
                "truncation rate cannot be measured without the model's own tokenizer"
            )
        clone = type(tokenizer).from_str(tokenizer.to_str())
        clone.no_truncation()
        clone.no_padding()
        return cls(clone, max_tokens=max_tokens)

    def count_tokens(self, text: str) -> int:
        """Tokens in ``text``, including ``[CLS]``/``[SEP]``."""
        return len(self._tokenizer.encode(text).ids)

    def truncate(self, text: str) -> tuple[str, int, bool]:
        """``(text_as_embedded, full_token_count, truncated)``.

        The cut point is the character offset of the last content token that
        fits, so the result is always a prefix of ``text`` ending on a token
        boundary.
        """
        encoding = self._tokenizer.encode(text)
        total = len(encoding.ids)
        if total <= self._max_tokens:
            return text, total, False
        # The full encoding is [CLS] c1..cn [SEP]; the window holds
        # [CLS] c1..c(max-2) [SEP], so the last content token kept is at index
        # max_tokens - 2.
        last_kept = self._max_tokens - _SPECIAL_TOKENS
        end = encoding.offsets[last_kept][1]
        return text[:end].rstrip(), total, True

    def build(
        self,
        title: str,
        conditions: Sequence[str],
        interventions: Sequence[str],
        brief_summary: str | None,
    ) -> Overview:
        """Build, measure and truncate in one step."""
        full = build_overview_text(title, conditions, interventions, brief_summary)
        text, token_count, truncated = self.truncate(full)
        return Overview(
            text=text,
            token_count=token_count,
            truncated=truncated,
            max_tokens=self._max_tokens,
        )
