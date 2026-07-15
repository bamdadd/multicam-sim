"""Movement DSL: paths evaluate analytically and compile to the right frames.

CPU-only, no renderer, no GL.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from multicam_sim.dsl import Path


def test_linear_endpoints_and_midpoint() -> None:
    p = Path.linear((0.0, 0.0, 0.0), (2.0, 4.0, 6.0))
    assert p.point(0.0) == [0.0, 0.0, 0.0]
    assert p.point(1.0) == [2.0, 4.0, 6.0]
    assert p.point(0.5) == [1.0, 2.0, 3.0]
    assert p.length() == pytest.approx(math.sqrt(4 + 16 + 36))


def test_linear_compiles_to_uniform_frames() -> None:
    # matches smoke.py's straight y-sweep: x=0, z=0.5, y from -0.6 to 0.6.
    p = Path.linear((0.0, -0.6, 0.5), (0.0, 0.6, 0.5))
    frames = p.compile_frames(fps=30.0, num_frames=11, name="center")
    assert len(frames) == 11
    assert frames[0].points["center"] == pytest.approx([0.0, -0.6, 0.5])
    assert frames[-1].points["center"] == pytest.approx([0.0, 0.6, 0.5])
    ys = [f.points["center"][1] for f in frames]
    expected = [-0.6 + (i / 10) * 1.2 for i in range(11)]
    assert ys == pytest.approx(expected)


def test_circle_radius_axis_and_period() -> None:
    c = Path.circle(center=(0.0, 0.0, 0.0), radius=2.0, axis=(0.0, 0.0, 1.0))
    for u in (0.0, 0.25, 0.5, 0.75):
        p = np.array(c.point(u))
        assert p[2] == pytest.approx(0.0)  # stays in z=0 plane
        assert np.linalg.norm(p) == pytest.approx(2.0)  # on the radius
    assert c.point(0.0) == pytest.approx(c.point(1.0))  # closed loop
    assert c.length() == pytest.approx(2 * math.pi * 2.0)


def test_waypoints_hits_each_node() -> None:
    w = Path.waypoints([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)])
    assert w.point(0.0) == pytest.approx([0.0, 0.0, 0.0])
    assert w.point(0.5) == pytest.approx([1.0, 0.0, 0.0])  # middle node
    assert w.point(1.0) == pytest.approx([1.0, 1.0, 0.0])
    assert w.length() == pytest.approx(2.0)


def test_bezier_passes_through_endpoints() -> None:
    b = Path.bezier([(0.0, 0.0, 0.0), (1.0, 2.0, 0.0), (2.0, 0.0, 0.0)])
    assert b.point(0.0) == pytest.approx([0.0, 0.0, 0.0])
    assert b.point(1.0) == pytest.approx([2.0, 0.0, 0.0])
    mid = b.point(0.5)  # quadratic bezier midpoint = 0.25 a + 0.5 b + 0.25 c
    assert mid == pytest.approx([1.0, 1.0, 0.0])


def test_then_concatenates_durations_and_geometry() -> None:
    a = Path.linear((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)).over(1.0)
    b = Path.linear((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)).over(1.0)
    seq = a.then(b)
    assert seq.total_duration() == pytest.approx(2.0)
    assert seq.at_time(0.0) == pytest.approx([0.0, 0.0, 0.0])
    assert seq.at_time(1.0) == pytest.approx([1.0, 0.0, 0.0])  # handoff point
    assert seq.at_time(2.0) == pytest.approx([1.0, 1.0, 0.0])


def test_repeat_loops() -> None:
    seg = Path.linear((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)).over(1.0)
    looped = seg.repeat(3)
    assert looped.total_duration() == pytest.approx(3.0)
    assert looped.at_time(0.5) == pytest.approx([0.5, 0.0, 0.0])
    assert looped.at_time(1.5) == pytest.approx([0.5, 0.0, 0.0])  # second lap
    assert looped.at_time(3.0) == pytest.approx([1.0, 0.0, 0.0])  # final endpoint held


def test_over_rescales_timeline() -> None:
    p = Path.linear((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)).over(5.0)
    assert p.total_duration() == pytest.approx(5.0)
    assert p.at_time(2.5) == pytest.approx([5.0, 0.0, 0.0])


def test_at_speed_sets_duration_from_length() -> None:
    p = Path.linear((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)).at_speed(2.0)  # 10 units / 2 = 5s
    assert p.total_duration() == pytest.approx(5.0)


def test_timed_path_holds_past_end_when_scene_is_longer() -> None:
    # a 1s motion sampled over a 2s (61-frame @30fps) scene holds at its endpoint.
    p = Path.linear((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)).over(1.0)
    frames = p.compile_frames(fps=30.0, num_frames=61)
    assert frames[30].points["center"] == pytest.approx([1.0, 0.0, 0.0])  # t=1.0s
    assert frames[-1].points["center"] == pytest.approx([1.0, 0.0, 0.0])  # held


def test_validation_at_construction() -> None:
    with pytest.raises(ValueError):
        Path.circle(center=(0.0, 0.0, 0.0), radius=-1.0)
    with pytest.raises(ValueError):
        Path.waypoints([(0.0, 0.0, 0.0)])
    with pytest.raises(ValueError):
        Path.linear((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)).over(0.0)
