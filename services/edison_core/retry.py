"""
Retry / backoff utilities used across EDISON service calls.

Phase 1 goal: clearer failure messaging and bounded retry behaviour for tool
calls, web searches, model invocations, and connector requests. The helpers
are intentionally framework-agnostic so they can be used from sync code,
async code, or wrapped around third-party clients.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Optional, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryPolicy:
    """Bounded exponential backoff with optional jitter.

    ``max_attempts`` is inclusive of the initial attempt — so
    ``max_attempts=3`` performs at most 1 try + 2 retries.
    """

    max_attempts: int = 3
    base_delay: float = 0.4  # seconds
    max_delay: float = 8.0
    multiplier: float = 2.0
    jitter: float = 0.25  # +/- fraction of computed delay
    retry_on: Tuple[Type[BaseException], ...] = (Exception,)
    give_up_on: Tuple[Type[BaseException], ...] = field(
        default_factory=lambda: (KeyboardInterrupt, SystemExit, asyncio.CancelledError)
    )

    def delay_for(self, attempt: int) -> float:
        """Delay (seconds) to wait *before* attempt number ``attempt`` (1-indexed).

        Attempt 1 has no delay. Subsequent attempts grow exponentially up to
        ``max_delay``, with optional jitter applied symmetrically.
        """
        if attempt <= 1:
            return 0.0
        raw = self.base_delay * (self.multiplier ** (attempt - 2))
        capped = min(raw, self.max_delay)
        if self.jitter > 0:
            spread = capped * self.jitter
            capped = max(0.0, capped + random.uniform(-spread, spread))
        return capped


class RetryError(RuntimeError):
    """Raised when every retry attempt has failed.

    The original exception is available on ``__cause__``; the list of all
    exceptions seen during retries is available on ``attempts``.
    """

    def __init__(self, attempts: list[BaseException]):
        self.attempts = attempts
        last = attempts[-1]
        super().__init__(
            f"Operation failed after {len(attempts)} attempt(s): "
            f"{type(last).__name__}: {last}"
        )


def _should_retry(exc: BaseException, policy: RetryPolicy) -> bool:
    if isinstance(exc, policy.give_up_on):
        return False
    return isinstance(exc, policy.retry_on)


def retry_sync(
    func: Callable[..., T],
    *args: Any,
    policy: Optional[RetryPolicy] = None,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
    **kwargs: Any,
) -> T:
    """Call ``func(*args, **kwargs)`` with retry semantics.

    ``on_retry(attempt, exc, delay)`` is invoked between attempts so callers
    can log / record telemetry without hooking the policy itself.
    """
    pol = policy or RetryPolicy()
    errors: list[BaseException] = []
    for attempt in range(1, pol.max_attempts + 1):
        delay = pol.delay_for(attempt)
        if delay > 0:
            time.sleep(delay)
        try:
            return func(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            if not _should_retry(exc, pol) or attempt >= pol.max_attempts:
                if on_retry:
                    on_retry(attempt, exc, 0.0)
                break
            if on_retry:
                on_retry(attempt, exc, pol.delay_for(attempt + 1))
            logger.warning(
                "retry_sync attempt %d/%d failed: %s",
                attempt, pol.max_attempts, exc,
            )
    raise RetryError(errors) from errors[-1]


async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    policy: Optional[RetryPolicy] = None,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
    **kwargs: Any,
) -> T:
    """Async version of :func:`retry_sync`."""
    pol = policy or RetryPolicy()
    errors: list[BaseException] = []
    for attempt in range(1, pol.max_attempts + 1):
        delay = pol.delay_for(attempt)
        if delay > 0:
            await asyncio.sleep(delay)
        try:
            return await func(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            if not _should_retry(exc, pol) or attempt >= pol.max_attempts:
                if on_retry:
                    on_retry(attempt, exc, 0.0)
                break
            if on_retry:
                on_retry(attempt, exc, pol.delay_for(attempt + 1))
            logger.warning(
                "retry_async attempt %d/%d failed: %s",
                attempt, pol.max_attempts, exc,
            )
    raise RetryError(errors) from errors[-1]


def friendly_error(exc: BaseException, *, action: str = "operation") -> str:
    """Render a human-readable error string for the chat UI.

    Mirrors the tone used by Copilot / ChatGPT: brief, actionable, and
    avoids leaking stack traces.
    """
    name = type(exc).__name__
    msg = str(exc).strip() or "no additional details"
    if isinstance(exc, RetryError):
        last = exc.attempts[-1]
        return (
            f"The {action} failed after {len(exc.attempts)} attempts. "
            f"Last error: {type(last).__name__}: {last}"
        )
    if isinstance(exc, asyncio.TimeoutError):
        return f"The {action} timed out. Please try again or simplify the request."
    if isinstance(exc, ConnectionError):
        return f"Could not reach the {action} service. Check your connection and try again."
    return f"The {action} failed ({name}): {msg}"
