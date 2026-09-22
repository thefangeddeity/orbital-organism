"""The 2D vector type.

Split out of `physics` in batch 8 for one reason: `planet` needs Vec2 to
express gravity and the ground, and `physics` needs `planet` to compute
acceleration. Leaving Vec2 in `physics` would have made that a cycle. It is
re-exported from `physics`, so `from .physics import Vec2` keeps working.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Vec2:
    """A 2D vector in metres, metres per second, or newtons as context says."""

    x: float
    y: float

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vec2(self.x * scalar, self.y * scalar)

    __rmul__ = __mul__

    def __neg__(self):
        return Vec2(-self.x, -self.y)

    def length(self):
        return math.hypot(self.x, self.y)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def cross(self, other):
        """2D scalar cross product, z-component of the 3D cross."""
        return self.x * other.y - self.y * other.x

    def perp(self):
        """Rotated a quarter turn counter-clockwise.

        For angular velocity w and lever arm r, the contact-point velocity
        contribution w x r is `r.perp() * w`.
        """
        return Vec2(-self.y, self.x)

    def normalized(self):
        length = math.hypot(self.x, self.y)
        if length == 0.0:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / length, self.y / length)

    def rotated(self, angle_rad):
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        return Vec2(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a,
        )
