"""Synchronous event bus with bounded derived-event propagation."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

from wphr.core.event import Event
from wphr.logging_setup import get_logger

Handler = Callable[[Event], Iterable[Event] | Event | None]
MAX_PROPAGATION = 512


class EventBus:
    """Topic subscribers plus a derivation queue.

    A handler may return one or more new events; those are queued and dispatched
    in the same publish call until the queue drains, with a hard cap that turns
    a runaway feedback loop into a logged error instead of a hang.
    """

    def __init__(self, history_size: int = 0) -> None:
        self._subs: dict[str, list[Handler]] = {}
        self._log = get_logger("bus")
        self.errors: list[str] = []
        self.history: deque[Event] = deque(maxlen=history_size or None) if history_size else deque()
        self.history_size = history_size
        self.dispatch_count = 0

    # ------------------------------------------------------------- subscribe
    def subscribe(self, topic: str, handler: Handler) -> None:
        self._subs.setdefault(topic, []).append(handler)

    def subscribe_many(self, topics: Iterable[str], handler: Handler) -> None:
        for topic in topics:
            self.subscribe(topic, handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        handlers = self._subs.get(topic, [])
        if handler in handlers:
            handlers.remove(handler)

    def topics(self) -> list[str]:
        return sorted(self._subs)

    # --------------------------------------------------------------- publish
    def publish(self, event: Event) -> list[Event]:
        """Dispatch ``event`` and any derived events; return everything emitted."""
        emitted: list[Event] = []
        queue: deque[Event] = deque([event])
        processed = 0
        while queue:
            current = queue.popleft()
            processed += 1
            if processed > MAX_PROPAGATION:
                msg = f"event propagation exceeded {MAX_PROPAGATION} events; loop dropped the rest"
                self._log.error(msg)
                self.errors.append(msg)
                break
            if self.history_size:
                self.history.append(current)
            emitted.append(current)
            for handler in self._handlers_for(current):
                self.dispatch_count += 1
                try:
                    result = handler(current)
                except Exception as exc:  # handlers must not break the loop
                    msg = f"handler {getattr(handler, '__qualname__', handler)} failed: {exc!r}"
                    self._log.exception(msg)
                    self.errors.append(msg)
                    continue
                if result is None:
                    continue
                if isinstance(result, Event):
                    queue.append(result)
                else:
                    for derived in result:
                        if derived is not None:
                            queue.append(derived)
        return emitted

    def _handlers_for(self, event: Event) -> list[Handler]:
        handlers: list[Handler] = []
        for topic in ("*", event.event_type, event.modality):
            handlers.extend(self._subs.get(topic, ()))
        return handlers

    def stats(self) -> dict[str, Any]:
        return {
            "topics": len(self._subs),
            "subscribers": sum(len(v) for v in self._subs.values()),
            "dispatches": self.dispatch_count,
            "errors": len(self.errors),
        }
