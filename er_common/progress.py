"""Progress bars (tqdm) shared by every stage.

``progress`` wraps an iterable, ``stages`` gives a bar that ticks once per named step (for
stretches of vectorised work that have no natural loop). Bars go to stderr and cooperate with
logging (see ``runner.main``); ``set_enabled(False)`` / ``--no-progress`` turns them off.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")

_ENABLED = True


def set_enabled(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = enabled


def _kw(desc: str, total: int | None, unit: str) -> dict[str, Any]:
    return {
        "desc": desc,
        "total": total,
        "unit": unit,
        "disable": not _ENABLED,
        "dynamic_ncols": True,
        "mininterval": 0.5,
        "leave": False,
        "unit_scale": True,
    }


def progress(it: Iterable[T], desc: str, total: int | None = None, unit: str = "it") -> Iterator[T]:
    """Iterate ``it`` with a progress bar."""
    yield from tqdm(it, **_kw(desc, total, unit))


def bar(desc: str, total: int, unit: str = "it") -> tqdm:
    """Manually updated bar (``with bar(...) as pb: pb.update(n)``)."""
    return tqdm(**_kw(desc, total, unit))


class _Stages:
    def __init__(self, pb: tqdm) -> None:
        self.pb = pb

    def __call__(self, name: str) -> None:
        """Mark the previous step done and show ``name`` as the current one."""
        if self.pb.n or self.pb.postfix:
            self.pb.update(1)
        self.pb.set_postfix_str(name, refresh=True)


@contextmanager
def stages(desc: str, total: int) -> Iterator[_Stages]:
    """Bar that advances once per named step: ``with stages("features", 5) as step: step("names") ...``."""
    with tqdm(**{**_kw(desc, total, "step"), "unit_scale": False}) as pb:
        s = _Stages(pb)
        yield s
        pb.update(pb.total - pb.n)
