from perlin_array import (
    get_perlin_numpy,
    get_perlin_grid_numpy,
    get_perlin_from_grid_numpy,
)
import numpy as np
import random


class PerlinArray:
    def __init__(
        self,
        shape: tuple,
        grid_size: int = 10,
        octaves: list[int] = [1],
        circular: bool = False,
        circular_axis: list[int] = None,
        seed: int = None,
    ):
        self.octaves = [item / sum(octaves) for item in octaves]
        self.width = shape[0]
        self.height = shape[1]

        match circular_axis:
            case [0, 1] | None:
                self.axis = "both"
            case [0] | 0:
                self.axis = "horizontal"
            case [1] | 1:
                self.axis = "vertical"
            case _:
                raise ValueError("Invalid circular_axis value")

        self.grid_size = grid_size
        self.octaves = octaves
        self.circular = circular

        if seed is None:
            seed = random.randint(0, 1147483647)
        self.seed = seed
        self._array = None
        self._grid = None

    @property
    def array(self):
        if self._array is None:
            self._array = self._get_perlin_numpy()
        return self._array

    def _get_perlin_numpy(self):
        # from rust lib
        array = get_perlin_numpy(
            grid_size=self.grid_size,
            width=self.width,
            height=self.height,
            octaves=self.octaves,
            circular=self.circular,
            axis=self.axis,
            seed=self.seed,
        )
        return array

    @property
    def grid(self):
        if self._grid is None:
            self._grid = self._get_grid()
        return self._grid

    def _get_grid(self):
        return get_perlin_grid_numpy(
            self.grid_size,
            len(self.octaves),
            self.seed,
            self.circular,
            self.axis,
        )

    def _perlin_from_grid_numpy(
        self,
        grid: list[np.ndarray],
        octaves: list[float],
        seed: int,
        circular: bool,
        axis: list[int] | int,
        width: int,
        height: int,
    ) -> np.ndarray:
        array = get_perlin_from_grid_numpy(
            grid,
            octaves,
            seed,
            circular,
            axis,
            width,
            height,
        )
        return array
