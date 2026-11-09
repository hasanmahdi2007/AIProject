"""The slice of Qdrant this service actually uses.

Three methods, named in one place. Two reasons that is worth a module rather
than just typing against :class:`~qdrant_client.AsyncQdrantClient`:

* **It is the honest dependency.** A reader can see at a glance that retrieval
  only ever *reads* - there is no ``upsert``, no ``delete``, no
  ``create_collection`` here. Writing to the index is B1's job, and a protocol
  that cannot express a write makes that structural rather than a convention.
* **It makes the container-free tests real tests.** The collection verifier's
  job is to notice a missing ``modifier: IDF``, and the fusion pipeline's job is
  to combine ranked lists correctly. Both are exercised against a stub in the
  fast unit suite, and a ``cast`` to the concrete client would have let the stub
  drift out of step with the real signatures without mypy noticing.

The real :class:`AsyncQdrantClient` satisfies this structurally, so nothing has
to declare that it implements it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from qdrant_client import models

# Not ``models.QueryResponse``. ``qdrant_client.models`` re-exports
# ``http.models``, ``fastembed_common`` and ``embed.models`` in that order, and
# ``fastembed_common`` defines its own unrelated ``QueryResponse`` (id,
# embedding, document, score) which shadows the REST one. Importing the wrong
# name type-checks fine and fails at runtime with a validation error about a
# missing ``document`` field. This is the alias the client itself is annotated
# with.
from qdrant_client.conversions.common_types import QueryResponse

__all__ = ["QdrantReader"]


class QdrantReader(Protocol):
    async def collection_exists(self, collection_name: str, **kwargs: Any) -> bool: ...

    async def get_collection(
        self, collection_name: str, **kwargs: Any
    ) -> models.CollectionInfo: ...

    async def query_batch_points(
        self,
        collection_name: str,
        requests: Sequence[models.QueryRequest],
        **kwargs: Any,
    ) -> list[QueryResponse]: ...
