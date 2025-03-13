from collections import namedtuple

import numpy as np
import tqdm
from pydantic import BaseModel

from gravity.newtonian.constants import PhysicalConstants
from gravity.newtonian.states import CloudState, StarState

kinematics = namedtuple("Kinematics", "position velocity")


class Dynamics(BaseModel):
    """Simulate dynamics in which stars exert gravitational forces on cloud particles
    and each other (if multiple are provided), but cloud particles do not exert forces
    on the stars or each other (assuming the masses are significantly smaller cloud
    masses).

    Params
    ------
    stars: StarState,
        The state of the stars at timestep 0.

    cloud: CloudState,
        The state of the cloud at timestep 0.

    timestep_dur_days: float,
        The duration of each simulated timestep in float multiples of days.
    """

    stars: StarState
    cloud: CloudState
    timestep_dur_days: float

    # used to store the cumulative history
    star_history: list = []
    cloud_history: list = []

    @property
    def duration_seconds(self):
        """Duration of a single timestep in SI units (seconds)."""
        return self.timestep_dur_days * 24 * 60 * 60

    # TODO: Make callable function instead of class method
    # TODO: split out distance calculation as a utility function
    def gravitational_force(
        self,
        star_mass: np.ndarray,
        star_pos: np.ndarray,
        entity_mass: np.ndarray,
        entity_pos: np.ndarray,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Calculate gravitational force from a gravitational source to a reference mass or masses.

        Params
        ------
        star_mass: np.ndarray,
            The mass of the star in SI units (kg).

        star_pos: np.ndarray,
            The position of the star in 3D Cartesian coordinates.

        entity_mass: np.ndarray,
            The mass of the entity or entities in SI units (kg).

        entity_pos: np.ndarray,
            The position of the entity or entities in 3D Cartesian coordinates.

        eps: float,
            A correction factor, in multiples of Astronomical Units (AU) to add to
            the distances to prevent division by zero.

        Returns
        -------
        force: np.ndarray,
            A 3D vector array of the gravitational force on each object.

        """
        # get the vector distance between them in cartesian coords
        relative_positions = entity_pos - star_pos

        # add small correction to prevent zero division in denominator
        epsilon = eps * PhysicalConstants.AU.value
        relative_distances_sq = np.sum((relative_positions**2), axis=1).reshape(-1, 1) + epsilon
        normalization = relative_distances_sq**0.5

        # calculate the unit radial vector
        normed_relative_positions = relative_positions / normalization

        # calculate and update the total force
        force = -(
            (PhysicalConstants.G.value * star_mass * entity_mass.reshape(-1, 1) * normed_relative_positions)
            / relative_distances_sq.reshape(-1, 1)
        )

        return force

    def calculate_cloud_forces(self, star_state: StarState, cloud_state: CloudState) -> np.ndarray:
        """Calculate the cumulative gravitational force on each entity in the cloud from
        each star.

        Params
        ------
        star_state: StarState,
            The state of each star at the current timestep.

        cloud_state: CloudState,
            The state of each entity in the cloud at the current timestep.

        Returns
        -------
        force: np.ndarray,
            A 3D vector array of the gravitational force on each entity in the cloud.

        """
        # initialise empty force vector
        force = np.zeros_like(cloud_state.force)

        # loop over all stars to calculate the gravitational effects
        for star_idx in range(len(star_state)):
            # calculate and update the gravitational force
            force += self.gravitational_force(
                star_mass=star_state[star_idx].mass,
                star_pos=star_state[star_idx].position,
                entity_mass=cloud_state.mass,
                entity_pos=cloud_state.position,
            )

        return force

    def calculate_stellar_forces(self, star_state: StarState) -> np.ndarray:
        """Calculate the force on each star by every other star.

        Params
        ------
        star_state: np.ndarray,
            The stars to calculate the gravitational force for.

        Returns
        -------
        stellar_forces: np.ndarray,
            The 3D force vector for every star in the star state.

        """
        # make a copy of the initial stars and get number of stars
        new_stars = star_state.model_copy()
        num_stars = len(new_stars)

        stellar_forces = np.zeros((num_stars, 3))

        # loop over stars and calculate the gravitational effect caused by other stars
        # TODO: consider whether this can be made faster using numba
        for ref_idx in range(num_stars):
            for measured_idx in range(num_stars):
                if ref_idx != measured_idx:
                    # add the cumulative force on each star from every other star
                    stellar_forces[measured_idx] += self.gravitational_force(
                        grav_mass=new_stars.mass[ref_idx],
                        grav_pos=new_stars.position[ref_idx],
                        ref_mass=new_stars.mass[measured_idx],
                        ref_pos=new_stars.position[measured_idx],
                    )

        return stellar_forces

    def leapfrog_position_velocity_integrator(
        self, star_state: StarState, ref_state: CloudState | StarState
    ) -> kinematics:
        """Calculate the Leapfrog integrator for position and velocity for
        greater numerical stability than Euler integration.

        Half update velocity, full update position with the half velocity,
        then half update velocity again with a recalculated acceleration from
        the new position.

        Leapfrog integration respects energy conservation better than Euler
        integration, leading to more stable motion.

        Params
        ------
        star_state: StarState,
            The star state used to calculate gravitational forces in the intermediate
            acceleration update when ref_state is a CloudState instance.

        ref_state: CloudState | StarState,
            The reference state to integrate the position and velocity for.

        Returns
        -------
        kinematics: namedtuple[np.ndarray, np.ndarray],
            A named tuple of the integrated position and velocity vector arrays.

        """
        # calculate initial acceleration
        init_acceleration = ref_state.force / ref_state.mass.reshape(-1, 1)

        # create a temporary reference state
        half_ref_update = ref_state.model_copy()

        # update velocity using v_half = u + a(t/2)
        half_update_ref_velocity = ref_state.velocity + (init_acceleration * (self.duration_seconds / 2))

        # update position using d = d0 + v_half * t
        updated_position = ref_state.position + (half_update_ref_velocity * self.duration_seconds)

        # update the acceleration
        half_ref_update.velocity = half_update_ref_velocity
        half_ref_update.position = updated_position

        if isinstance(ref_state, CloudState):
            half_update_acceleration = self.calculate_cloud_forces(star_state, half_ref_update) / half_ref_update.mass.reshape(-1,1)
        elif isinstance(ref_state, StarState):
            half_update_acceleration = self.calculate_stellar_forces(half_ref_update) / half_ref_update.mass.reshape(-1,1)
        else:
            msg = "Cloud points should be of type CloudState or StarState"
            return TypeError(msg)

        # update velocity again using v_full = v_half + a (t/2)
        updated_cloud_velocity = half_update_ref_velocity + (half_update_acceleration * (self.duration_seconds / 2))

        # use named tuple for unambiguous extraction
        output = kinematics(updated_position, updated_cloud_velocity)

        return output

    def update_stars(self, star_state: StarState) -> StarState:
        """Propagate the star state forward by one timestep.

        Params
        ------
        star_state: StarState,
            The initial star state to update.

        Returns
        -------
        new_stars: StarState,
            The updated star state with new position, velocity and force.

        """
        # make a copy of the initial stars
        new_stars = star_state.model_copy()

        # calculate the force on the stars caused by other stars
        new_stars.force = self.calculate_stellar_forces(new_stars)

        # update the position and velocity of the cloud
        kinematics = self.leapfrog_position_velocity_integrator(star_state=new_stars, ref_state=new_stars)
        new_stars.velocity = kinematics.velocity
        new_stars.position = kinematics.position

        # update the time
        new_stars.current_timestep += 1

        return new_stars

    def update_cloud(self, star_state: StarState, cloud_state: CloudState) -> CloudState:
        """Propagate the cloud state forward one timestep based on the star state
        in the current timestep.

        Params
        ------
        star_state: StarState,
            The initial star state used to update the cloud state.

        cloud_state: CloudState,
            The initial cloud state to update.

        Returns
        -------
        new_stars: CloudState,
            The updated cloud state with new position, velocity and force.

        """
        # make a copy of the initial cloud state and get number of stars
        new_cloud = cloud_state.model_copy()

        # update the forces acting on the cloud
        new_cloud.force = self.calculate_cloud_forces(star_state, new_cloud)

        # update the position and velocity of the cloud
        kinematics = self.leapfrog_position_velocity_integrator(star_state, new_cloud)
        new_cloud.velocity = kinematics.velocity
        new_cloud.position = kinematics.position

        # update the time
        new_cloud.current_timestep += 1

        return new_cloud

    def simulate(self, num_timesteps: int) -> tuple[StarState, CloudState]:
        """Simulate gravitational dynamics with non-interacting test masses in
        the cloud and interacting massess in the star state.

        Params
        ------
        num_timesteps: int,
            The number of timesteps to iterate the simulation.

        Returns
        -------
        tuple[StarState, CloudState],
            The final star and cloud states.

        """
        # store the initial states
        self.star_history.append(self.stars)
        self.cloud_history.append(self.cloud)

        # loop over the timesteps and iteratively update the cloud and stars
        for _ in tqdm.tqdm(range(num_timesteps)):
            # update stars and clouds
            self.stars = self.update_stars(self.stars)
            self.cloud = self.update_cloud(self.stars, self.cloud)

            # continue to store each timestep
            self.star_history.append(self.stars)
            self.cloud_history.append(self.cloud)

        return self.stars, self.cloud
