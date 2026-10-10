"""Small terminal-only batch progress display shared by training loops."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sized
from typing import TypeVar

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

T = TypeVar("T")


def progress_batches(
    batches: Iterable[T], description: str
) -> Iterator[tuple[T, Callable[[str], None]]]:
    """Yield batches and a metric reporter; count only completed batch bodies."""
    console = Console(stderr=True)
    total = len(batches) if isinstance(batches, Sized) else None
    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[metrics]}"),
        console=console,
        disable=not console.is_terminal,
        refresh_per_second=2,
        redirect_stdout=False,
        redirect_stderr=False,
    ) as progress:
        task = progress.add_task(description, total=total, metrics="")

        def report(metrics: str) -> None:
            progress.update(task, metrics=metrics)

        for batch in batches:
            yield batch, report
            progress.advance(task)
