"""Unit tests for exactly-once counting."""

from pipeline.counter import Counter
from pipeline.quality_checker import QualityResult
from pipeline.tracker import TrackResult


def _track(center_y: int) -> TrackResult:
    """Create a track centered at a requested vertical coordinate."""
    return TrackResult(
        track_id=1,
        bbox=(100, center_y - 20, 200, center_y + 20),
        confidence=0.9,
        class_name="cardboard_box",
        center=(150, center_y),
        area=4000,
        trajectory=((150, center_y),),
        is_occluded=False,
    )


def _quality(status: str = "OK") -> QualityResult:
    """Create a compact quality result for counter tests."""
    return QualityResult(1, status, (), 1.0, 0.0, None, False, 1, True)


def test_counter_counts_one_track_once() -> None:
    """Verify a line crossing increments total only once."""
    counter = Counter(50)
    counter.update([_track(40)], {1: _quality()})
    crossed = counter.update([_track(60)], {1: _quality()})
    repeated = counter.update([_track(70)], {1: _quality()})
    assert crossed.total_count == 1
    assert repeated.total_count == 1
    assert repeated.good_count == 1


def test_counter_tracks_defective_bucket() -> None:
    """Verify a defective quality result is placed in its own bucket."""
    counter = Counter(50)
    counter.update([_track(40)], {1: _quality("DEFECTIVE")})
    result = counter.update([_track(60)], {1: _quality("DEFECTIVE")})
    assert result.defective_count == 1
    assert result.good_count == 0
