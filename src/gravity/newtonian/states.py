from abc import ABC

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


class State(ABC, BaseModel):
    """An Abstract Base Class for the state of particles in a gravitational system.

    Params
    ------
    timestep: int,
        The current timestep that the state corresponds to.

    mass: np.ndarray,
        The mass of each entity in the state.

    position: np.ndarray,
        An array of 3D Cartesian position coordinates for every entity
        in the state.

    velocity: np.ndarray,
        An array of 3D velocities in Cartesian coordinates for every
        entity in the state.

    force: np.ndarray,
        An array of 3D force vectors in Cartesian coordinates for every
        entity in the state.
    """

    current_timestep: int
    mass: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    force: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __len__(self):
        """Find the number of entities in the state."""
        return len(self.mass)

    def __getitem__(self, idx: int):
        """Get the entity at a particular index in the state."""
        return StarState(
            current_timestep=self.current_timestep,
            mass=np.array([self.mass[idx : idx + 1]]),
            position=self.position[idx : idx + 1],
            velocity=self.velocity[idx : idx + 1],
            force=self.force[idx : idx + 1],
        )

    @model_validator(mode="after")
    def ensure_float64(self):
        """Ensure float64 precision on each input."""
        self.mass = self.mass.astype(np.float64)
        self.position = self.position.astype(np.float64)
        self.velocity = self.velocity.astype(np.float64)
        self.force = self.force.astype(np.float64)

        return self


class StarState(State):
    """A state for stars."""

    def __init__(self):
        """Inherit from State abstract class."""
        super().__init__()


class CloudState(State):
    """A state for non-star entities, such as planets or particles."""

    def __init__(self):
        """Inherit from State abstract class."""
        super().__init__()
