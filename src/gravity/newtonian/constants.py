from enum import Enum


class PhysicalConstants(Enum):
    """Class for physical constants in SI units."""

    G: float = 6.674 * (10**-11)  # m^3 kg^-1 s^-2
    SOLAR_MASS: float = 2 * (10**30)  # kg
    AU: float = 1.5 * (10**11)  # metres
