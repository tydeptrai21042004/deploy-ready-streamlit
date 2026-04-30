"""Geometry helpers for line-crossing vehicle counting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

Point = Tuple[float, float]
Direction = Literal["both", "positive_to_negative", "negative_to_positive"]


@dataclass(frozen=True)
class CountingLine:
    """A line defined by two points in pixel coordinates."""

    p1: Point
    p2: Point

    @classmethod
    def from_normalized(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: int,
        height: int,
    ) -> "CountingLine":
        """Create a pixel line from normalized coordinates in [0, 1]."""
        return cls(
            p1=(float(x1) * width, float(y1) * height),
            p2=(float(x2) * width, float(y2) * height),
        )

    def side(self, point: Point) -> float:
        """Return signed side of point relative to the directed line p1 -> p2."""
        x1, y1 = self.p1
        x2, y2 = self.p2
        px, py = point
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def crossed(self, previous_side: float, current_side: float, direction: Direction = "both") -> bool:
        """Return True if a track crossed the line between two side values."""
        if previous_side == 0 or current_side == 0:
            return False
        if previous_side * current_side > 0:
            return False

        if direction == "both":
            return True
        if direction == "positive_to_negative":
            return previous_side > 0 and current_side < 0
        if direction == "negative_to_positive":
            return previous_side < 0 and current_side > 0
        raise ValueError(f"Unsupported direction: {direction}")


def box_center_xyxy(box: Tuple[float, float, float, float]) -> Point:
    """Return center point from an xyxy box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
