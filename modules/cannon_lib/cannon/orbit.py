"""Orbital elements and the conic a state vector is travelling on.

2D, in the plane. No inclination and no ascending node, so an orbit is
fully described by its semi-major axis, eccentricity, and the direction of
periapsis - plus which way round it is going.

THIS IS THE PAYOFF OF THE ROUND-EARTH BATCH. The cannon's arc was never a
parabola. It is a segment of an ellipse with the Earth's centre at the far
focus, and for the factory shot that ellipse has a ~3186 km semi-major axis
- about half the Earth's radius - so almost all of it lies INSIDE the
planet. The flown part is the small piece that happens to be above the
ground. Drawing the whole conic is what makes that visible: the parabola was
never real, the ground merely got in the way.

THE CONIC IS EXACT ONLY DRAG-FREE. With drag it is the OSCULATING orbit -
the ellipse the body is instantaneously travelling on, tangent to the true
path at this moment and diverging from it immediately afterwards. It is not
a prediction and must not be presented as one; the render layer draws the
integrated path alongside it so the divergence is visible rather than
hidden.

Imports `vec` only. It knows nothing about planets, rendering or the shot -
it takes a state vector and a gravitational parameter and returns geometry.
"""

import math
from dataclasses import dataclass

from .vec import Vec2

#: Below this eccentricity an orbit is treated as circular and the periapsis
#: direction is arbitrary. Purely to keep atan2 from amplifying noise into a
#: meaningless angle on a near-perfect circle.
CIRCULAR_ECCENTRICITY = 1e-12


@dataclass(frozen=True, slots=True)
class OrbitElements:
    """The conic a body is on, in the plane."""

    semi_major_axis_m: float  # negative for a hyperbola
    eccentricity: float
    #: 1 - e, carried SEPARATELY and computed without ever forming the
    #: difference. For a near-radial orbit - which every cannon shot is,
    #: at e = 0.999893 - subtracting a float64 eccentricity from 1 leaves
    #: only ~12 significant digits in the result, and that shortfall
    #: propagates straight into every radius. Derived instead from
    #: p / (a*(1+e)), in which every factor is well conditioned because
    #: (1+e) is near 2 rather than near 0.
    one_minus_eccentricity: float
    periapsis_angle_rad: float  # argument of periapsis, from world +x
    semi_latus_rectum_m: float  # p = h^2 / mu
    specific_angular_momentum: float  # signed; sign is the direction of travel
    mu_m3_s2: float

    @property
    def direction(self):
        """+1 counter-clockwise, -1 clockwise."""
        return 1.0 if self.specific_angular_momentum >= 0.0 else -1.0

    @property
    def is_closed(self):
        """False for a parabola or hyperbola, which must not be drawn."""
        return self.eccentricity < 1.0 and self.semi_major_axis_m > 0.0

    @property
    def periapsis_radius_m(self):
        return self.semi_latus_rectum_m / (1.0 + self.eccentricity)

    @property
    def apoapsis_from_one_minus_e_m(self):
        """Apoapsis via the well-conditioned 1 - e. Equals apoapsis_radius_m."""
        return self.semi_latus_rectum_m / self.one_minus_eccentricity

    @property
    def apoapsis_radius_m(self):
        """Only meaningful for a closed orbit; infinite otherwise."""
        if not self.is_closed:
            return math.inf
        return self.semi_latus_rectum_m / self.one_minus_eccentricity

    @property
    def period_s(self):
        if not self.is_closed:
            return math.inf
        return 2.0 * math.pi * math.sqrt(
            self.semi_major_axis_m**3 / self.mu_m3_s2
        )


def elements_from_state(position_m, velocity_m_s, mu_m3_s2):
    """Orbital elements from a state vector.

    GUARDED AGAINST THE DEGENERATE CASES rather than trusting they cannot
    happen. A hyperbolic solution will not arise at cannon speeds, but if it
    ever does this must not divide by zero or return NaN - it returns
    elements with `is_closed` False, and the render layer declines to draw
    them.
    """
    radius_m = position_m.length()
    speed_ms = velocity_m_s.length()
    if radius_m <= 0.0 or mu_m3_s2 <= 0.0:
        raise ValueError("elements are undefined at the centre of attraction")

    angular_momentum = position_m.cross(velocity_m_s)

    # a = 1 / (2/r - v^2/mu). The reciprocal form is guarded because the
    # denominator vanishes exactly at escape velocity.
    inverse_a = 2.0 / radius_m - speed_ms * speed_ms / mu_m3_s2
    semi_major_axis_m = math.inf if inverse_a == 0.0 else 1.0 / inverse_a

    eccentricity_vector = (
        position_m * (speed_ms * speed_ms - mu_m3_s2 / radius_m)
        - velocity_m_s * position_m.dot(velocity_m_s)
    ) * (1.0 / mu_m3_s2)
    eccentricity = eccentricity_vector.length()

    periapsis_angle_rad = (
        0.0
        if eccentricity < CIRCULAR_ECCENTRICITY
        else math.atan2(eccentricity_vector.y, eccentricity_vector.x)
    )

    semi_latus_rectum_m = angular_momentum * angular_momentum / mu_m3_s2
    # 1 - e without forming the difference; see the field's docstring.
    if math.isfinite(semi_major_axis_m) and semi_major_axis_m != 0.0:
        one_minus_eccentricity = semi_latus_rectum_m / (
            semi_major_axis_m * (1.0 + eccentricity)
        )
    else:
        one_minus_eccentricity = 1.0 - eccentricity

    return OrbitElements(
        semi_major_axis_m=semi_major_axis_m,
        eccentricity=eccentricity,
        one_minus_eccentricity=one_minus_eccentricity,
        periapsis_angle_rad=periapsis_angle_rad,
        semi_latus_rectum_m=semi_latus_rectum_m,
        specific_angular_momentum=angular_momentum,
        mu_m3_s2=mu_m3_s2,
    )


def true_anomaly_at(elements, position_m):
    """Angle from periapsis to a position, along the direction of travel."""
    position_angle = math.atan2(position_m.y, position_m.x)
    return _wrap(
        elements.direction * (position_angle - elements.periapsis_angle_rad)
    )


def _one_plus_e_cos(one_minus_eccentricity, eccentricity, true_anomaly_rad):
    """1 + e*cos(nu), computed WITHOUT catastrophic cancellation.

    The direct form loses most of its significant digits on a near-radial
    orbit, which is exactly what a cannon shot is: the factory shot has
    e = 0.999893, and near apoapsis 1 + e*cos(nu) evaluates to 1.07e-4 from
    two terms of magnitude 1. That is four digits gone, and it showed up as
    a 1.9e-12 round-trip error against a 1e-12 budget - the algebra was
    right and the arithmetic was not.

    The identity 1 + cos(nu) = 2*cos^2(nu/2) turns it into

        (1 - e) + 2*e*cos^2(nu/2)

    which for e < 1 is a sum of two NON-NEGATIVE terms and so cannot cancel.
    Same value, several digits better conditioned where it matters.
    """
    half_cosine = math.cos(0.5 * true_anomaly_rad)
    return (
        one_minus_eccentricity
        + 2.0 * eccentricity * half_cosine * half_cosine
    )


def radius_at(elements, true_anomaly_rad):
    """Orbital radius at a true anomaly, m. Infinite on the asymptote."""
    denominator = _one_plus_e_cos(
        elements.one_minus_eccentricity, elements.eccentricity, true_anomaly_rad
    )
    if denominator <= 0.0:
        return math.inf
    return elements.semi_latus_rectum_m / denominator


def state_at(elements, true_anomaly_rad):
    """Reconstruct (position, velocity) at a true anomaly.

    The inverse of `elements_from_state`. Gate 27 round-trips through both.
    """
    radius_m = radius_at(elements, true_anomaly_rad)
    if not math.isfinite(radius_m):
        raise ValueError("no state on the asymptote of an open orbit")

    position_angle = (
        elements.periapsis_angle_rad + elements.direction * true_anomaly_rad
    )
    radial = Vec2(math.cos(position_angle), math.sin(position_angle))
    # Transverse unit vector, along the direction of travel.
    transverse = radial.perp() * elements.direction

    angular_momentum = abs(elements.specific_angular_momentum)
    radial_speed = (
        elements.mu_m3_s2 / angular_momentum
        * elements.eccentricity * math.sin(true_anomaly_rad)
    )
    transverse_speed = (
        elements.mu_m3_s2 / angular_momentum
        * _one_plus_e_cos(
            elements.one_minus_eccentricity,
            elements.eccentricity,
            true_anomaly_rad,
        )
    )
    return (
        radial * radius_m,
        radial * radial_speed + transverse * transverse_speed,
    )


def radius_crossings(elements, radius_m):
    """True anomalies where the orbit crosses a given radius.

    Returns () if it never does - which is the common case for a circle
    against the ground, and for an orbit entirely above or below it.
    """
    if elements.eccentricity < CIRCULAR_ECCENTRICITY or radius_m <= 0.0:
        return ()
    cosine = (elements.semi_latus_rectum_m / radius_m - 1.0) / elements.eccentricity
    if not -1.0 <= cosine <= 1.0:
        return ()
    anomaly = math.acos(cosine)
    return (-anomaly, anomaly)


def arcs_above_radius(elements, radius_m):
    """True-anomaly spans on which the orbit is at or above `radius_m`.

    Returns a tuple of (from_anomaly, to_anomaly) pairs - empty when the whole
    orbit lies below.

    `radius_crossings` alone cannot answer this: it returns () both for an
    orbit entirely above the radius and for one entirely below it, and those
    need opposite drawings. The apsides disambiguate.

    r increases monotonically with |nu| from periapsis to apoapsis, so above
    the crossing the above-surface set is the single span running from +nu_c
    through apoapsis to 2*pi - nu_c. It is ONE span, not two, because it does
    not straddle periapsis.
    """
    if not elements.is_closed:
        return ()
    if elements.periapsis_radius_m >= radius_m:
        return ((-math.pi, math.pi),)
    if elements.apoapsis_radius_m <= radius_m:
        return ()
    crossings = radius_crossings(elements, radius_m)
    if not crossings:
        # Periapsis below and apoapsis above, yet no crossing: only reachable
        # through rounding on a near-circular orbit, where the two apsides are
        # within float noise of each other and of the radius. Draw nothing
        # rather than guess which side of the ground it is on.
        return ()
    anomaly = abs(crossings[1])
    return ((anomaly, 2.0 * math.pi - anomaly),)


def polylines_above_radius(elements, radius_m, *, count=256):
    """The conic clipped to `radius_m`, as world-space polylines for drawing.

    THE UNDERGROUND SEGMENT IS NOT DRAWN. Batch 9 SS3 drew the whole ellipse
    deliberately, to make the point that a cannon shot's orbit lies almost
    entirely inside the planet. PM override in batch 15 SS0: at the near-radial
    eccentricity every cannon shot actually has, the full ellipse renders as a
    doubled vertical line through the planet and reads as a rendering artifact
    rather than as a lesson. The ground-crossing markers carry the point
    instead.

    Sampling the clipped span rather than the whole orbit also puts the
    samples where they can be seen. The factory shot's above-ground arc spans
    about 0.03 deg of true anomaly out of 360; sampling the whole ellipse at
    256 points put ONE sample on the visible arc.

    Returns a tuple of point tuples. Each polyline is open - the caller must
    not close it, because a closed conic clipped at the surface is no longer a
    closed curve.

    THE SAMPLE COUNT IS FORCED ODD so that one sample lands on the midpoint of
    the span. Every clipped span is symmetric about apoapsis, so its midpoint
    IS apoapsis - and with an even count the samples straddle it and the drawn
    arc stops short of its own apex. Gate 99 measured 4.7 m of shortfall on a
    200 km orbit at 256 points, which is the whole visible height of a cannon
    shot's arc twice over. Cheap to fix here, invisible if left.
    """
    odd_count = count if count % 2 else count + 1
    return tuple(
        sample(elements, count=odd_count, from_anomaly=start, to_anomaly=end)
        for start, end in arcs_above_radius(elements, radius_m)
    )


def sample(elements, *, count=256, from_anomaly=None, to_anomaly=None):
    """Positions along the conic, for drawing.

    Samples the whole closed orbit by default. An open orbit is never
    sampled - the caller checks `is_closed` first.
    """
    if not elements.is_closed:
        return ()
    start = -math.pi if from_anomaly is None else from_anomaly
    end = math.pi if to_anomaly is None else to_anomaly
    span = end - start
    return tuple(
        state_at(elements, start + span * index / (count - 1))[0]
        for index in range(count)
    )


def _wrap(angle_rad):
    """Fold an angle into (-pi, pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))
