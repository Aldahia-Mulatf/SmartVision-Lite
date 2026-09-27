"""Exactly-once conveyor counting for tracked cardboard boxes."""

from __future__ import annotations

from dataclasses import dataclass

import config
from pipeline.quality_checker import QualityResult
from pipeline.tracker import TrackResult
from utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class CountResult:
    """Snapshot of the counting line totals."""

    total_count: int
    good_count: int
    defective_count: int
    counted_ids: frozenset[int]


class Counter:
    """Count each track once when its center crosses the configured line."""

    def __init__(self, counting_line_y: int = config.COUNTING_LINE_Y) -> None:
        """Initialize the line-crossing state."""
        self._counting_line_y = counting_line_y
        self._previous_y: dict[int, int] = {}
        self._counted_ids: set[int] = set()
        self._quality_by_id: dict[int, str] = {}

    def update(
        self,
        tracks: list[TrackResult],
        quality_results: dict[int, QualityResult],
    ) -> CountResult:
        """Update crossings and return current totals.

        Args:
            tracks: Active tracks from the tracking stage.
            quality_results: Current quality state by track ID.

        Returns:
            Exactly-once count snapshot.
        """
        for track in tracks:
            previous_y = self._previous_y.get(track.track_id)
            current_y = track.center[1]
            if previous_y is not None and track.track_id not in self._counted_ids:
                crossed = (previous_y - self._counting_line_y) * (
                    current_y - self._counting_line_y
                ) <= 0 and previous_y != current_y
                if crossed:
                    self._counted_ids.add(track.track_id)
                    LOGGER.info("Track %s crossed the counting line", track.track_id)
            self._previous_y[track.track_id] = current_y
            if (
                track.track_id in self._counted_ids
                and track.track_id in quality_results
            ):
                self._quality_by_id[track.track_id] = quality_results[
                    track.track_id
                ].status
        return self._snapshot()

    def update_quality(self, quality_results: dict[int, QualityResult]) -> CountResult:
        """Refresh good/defective buckets after final quality evaluation."""
        for track_id in self._counted_ids:
            if track_id in quality_results:
                self._quality_by_id[track_id] = quality_results[track_id].status
        return self._snapshot()

    def _snapshot(self) -> CountResult:
        """Build an immutable count result."""
        defective_count = sum(
            status == "DEFECTIVE" for status in self._quality_by_id.values()
        )
        return CountResult(
            total_count=len(self._counted_ids),
            good_count=len(self._counted_ids) - defective_count,
            defective_count=defective_count,
            counted_ids=frozenset(self._counted_ids),
        )
