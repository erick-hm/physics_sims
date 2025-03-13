import logging
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from pydantic import BaseModel

from gravity.newtonian.dynamics import Dynamics


class Plotter(BaseModel):
    """A class to plot dynamics from a Dynamics simulation.

    Params
    ------
    dynamics: Dynamics,
        A simulated run of gravitational dynamics.

    figsize: tuple[float, float],
        The size of the figure to use when plotting.

    xaxis_range: tuple[float, float],
        The minimum and maximum x-axis values to show.

    yaxis_range: tuple[float, float],
        The minimum and maximum y-axis values to show.

    zaxis_range: tuple[float, float],
        The minimum and maximum z-axis values to show.

    show_grid: bool,
        Whether to show a grid in the background image.

    color_config: dict,
        The colour configurations to use. [Currently unavailable]
    """

    dynamics: Dynamics

    figsize: tuple[float, float] = (9, 9)
    xaxis_range: tuple[float, float] | None = None
    yaxis_range: tuple[float, float] | None = None
    zaxis_range: tuple[float, float] | None = None
    show_grid: bool = False
    colour_config: dict = {}

    def model_post_init(self, __context) -> None:
        """Extract position and velocity arrays from the simulation."""
        self._get_positions()
        self._get_velocities()

    def _get_positions(self) -> None:
        self._star_positions = np.array([x.position for x in self.dynamics.star_history])
        self._cloud_positions = np.array([x.position for x in self.dynamics.cloud_history])

    def _get_velocities(self) -> None:
        self._star_velocities = np.array([x.velocity for x in self.dynamics.star_history])
        self._cloud_velocities = np.array([x.velocity for x in self.dynamics.cloud_history])

    def create_figure(self) -> None:
        """Create a backdrop figure for re-use in plotting."""
        self._anim_fig = plt.figure(figsize=self.figsize)
        self._ax = self._anim_fig.add_subplot(projection="3d")

        # set background colour
        self._ax.set_facecolor("black")
        self._ax.xaxis.set_pane_color("black")
        self._ax.yaxis.set_pane_color("black")
        self._ax.zaxis.set_pane_color("black")

    def _plot_timestep(self, step: int, n: int) -> None:
        """Plot a single timestep of the dynamics."""
        # clear the previous drawing - use the same figure
        self._ax.clear()

        self._ax.scatter(
            self._star_positions[step * n, :, 0],
            self._star_positions[step * n, :, 1],
            self._star_positions[step * n, :, 2],
            color="darkorange",
            s=300,
        )

        self._ax.scatter(
            self._cloud_positions[step * n, :, 0],
            self._cloud_positions[step * n, :, 1],
            self._cloud_positions[step * n, :, 2],
            color="dodgerblue",
        )

        # set static range limits
        self._ax.set_xlim(self.xaxis_range)
        self._ax.set_ylim(self.yaxis_range)
        self._ax.set_zlim(self.zaxis_range)

        # remove axes labels
        self._ax.set_yticklabels([])
        self._ax.set_xticklabels([])
        self._ax.set_zticklabels([])

        # whether to show the grid
        self._ax.grid(self.show_grid)

        plt.show()

    def plot_dynamics_at_timestep(self, n: int) -> None:
        """Plot a single timestep of the dynamics.

        Params
        ------
        n: int,
            The timestep of the dynamics to plot.

        Returns
        -------
        None

        """
        self.create_figure()
        self._plot_timestep(step=1, n=n)

    def create_dynamics_animation(
        self,
        filename: str,
        frames: int = 250,
        interval: int = 50,
        step: int = 1,
        fps: int | None = None,
        dpi: int | None = None,
        overwrite: bool = True,
    ) -> None:
        """Create an animation of the dynamics.

        Params
        ------
        filename: str,
            The file path of the saved animation.

        frames: int,
            The total number of frames to draw.

        interval: int,
            The time in milliseconds between frames.

        step: int,
            The number of timesteps to skip between each frame.
            Scales the timestep to count in intervals of `step`.

        fps: int | None,
            The number of frames to display per second.

        dpi: int,
            The resolution to create the plot at.

        overwrite: bool,
            Whether to overwrite an existing file with the same filename.

        Returns
        -------
        None

        """
        # create the base figure and extract the relevant data
        self.create_figure()

        # create the animation if the file doesn't exist or can be overwritten
        if not Path(filename).exists() or overwrite:
            plot_single_frame = partial(self._plot_timestep, step)

            # create the animation object
            animator = FuncAnimation(
                self._anim_fig, func=plot_single_frame, frames=frames, interval=interval, repeat=True
            )

            logging.info("Saving the animation.")
            animator.save(filename, fps=fps, dpi=dpi)
