"""Tests for the intersection geometry and lane ROIs."""
from __future__ import annotations

from src.traffic.intersection import Intersection


def test_normalised_lane_assignment_and_unknown(intersection: Intersection):
    assert intersection.lane_at(640, 100, 1280, 720) == "north"
    assert intersection.lane_at(640, 650, 1280, 720) == "south"
    assert intersection.lane_at(120, 360, 1280, 720) == "west"
    assert intersection.lane_at(1160, 360, 1280, 720) == "east"
    assert intersection.lane_at(640, 360, 1280, 720) is None


def test_groups_and_conflicts(intersection: Intersection):
    assert intersection.group_of("north") == "north_south"
    assert intersection.group_of("south") == "north_south"
    assert intersection.group_of("east") == "east_west"
    assert intersection.conflicts("north", "east")
    assert not intersection.conflicts("north", "south")
    assert set(intersection.conflicting("north")) == {"east", "west"}
    assert set(intersection.non_conflicting("north")) == {"north", "south"}


def test_exit_zones_are_far_side(intersection: Intersection):
    assert intersection.in_exit("north", 640, 650, 1280, 720)
    assert not intersection.in_exit("north", 640, 100, 1280, 720)
    assert intersection.in_exit("south", 640, 80, 1280, 720)
    assert intersection.in_exit("west", 1160, 360, 1280, 720)
    assert intersection.in_exit("east", 120, 360, 1280, 720)


def test_distance_to_center_is_lane_independent(intersection: Intersection):
    north = intersection.normalized_distance_to_center(640, 100, 1280, 720)
    south = intersection.normalized_distance_to_center(640, 620, 1280, 720)
    west = intersection.normalized_distance_to_center(100, 360, 1280, 720)
    assert north == south
    assert 0 < north < 1
    assert 0 < west < 1


def test_signal_state_validation_rejects_conflicting_greens(intersection: Intersection):
    intersection.validate_signal_state({"north": "GREEN", "south": "GREEN", "east": "RED", "west": "RED"})
    try:
        intersection.validate_signal_state({"north": "GREEN", "south": "RED", "east": "GREEN", "west": "RED"})
    except AssertionError:
        pass
    else:
        raise AssertionError("conflicting greens should be rejected")
