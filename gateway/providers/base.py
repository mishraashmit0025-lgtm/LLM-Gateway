from __future__ import annotations

import abc
from collections.abc import AsyncIterator

from ..schemas import ChatRequest, Completion, StreamChunk


class ProviderError(Exception):
    """A failed upstream call. retryable decides whether the router retries or
    falls back; client errors (bad request, auth) are returned to the caller."""

    def __init__(self, message: str, *, status: int = 502, retryable: bool = True, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after


class Provider(abc.ABC):
    name: str

    @abc.abstractmethod
    async def complete(self, req: ChatRequest, model: str) -> Completion: ...

    @abc.abstractmethod
    def stream(self, req: ChatRequest, model: str) -> AsyncIterator[StreamChunk]: ...

    async def aclose(self) -> None:
        return None
