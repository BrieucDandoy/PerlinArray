from perlin_array import get_perlin_numpy, get_perlin_grid_numpy
import numpy as np
from functools import lru_cache
import random


class PerlinArray:
    def __init__(
        self,
        shape: tuple,
        grid_size=10,
        octaves=[1],
        circular: bool = False,
        circular_axis=None,
        seed=42,
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
        self.seed = seed
        self._array = None

    @property
    def array(self):
        if self._array is None:
            self._array = self._get_perlin_numpy()
        return self._array

    @lru_cache(maxsize=None)
    def _get_perlin_numpy(self):
        # from rust lib
        return get_perlin_numpy(
            grid_size=self.grid_size,
            width=self.width,
            height=self.height,
            octaves=self.octaves,
            circular=self.circular,
            axis=self.axis,
            seed=self.seed,
        )

    @property
    def grid(self):
        if self.seed is None:
            self.seed = random.randint(0, 1000)
        return self._get_grid()

    @lru_cache
    def _get_grid(self):
        return get_perlin_grid_numpy(
            self.grid_size,
            len(self.octaves),
            self.seed,
            self.circular,
            self.axis,
        )


if __name__ == "__main__":
    ar1 = PerlinArray((1024, 1024), grid_size=10, octaves=[1, 0.75, 0.5, 0.25])
    for grid in ar1.grid:
        print(grid[1].shape)
