from perlin_array import (
    get_perlin_numpy,
    get_perlin_grid_numpy,
    get_perlin_from_grid_numpy,
)
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
    ar1 = PerlinArray(
        (512, 512),
        grid_size=10,
        octaves=[1, 0.75, 0.5, 0.25],
        seed=42,
        circular_axis=[0, 1],
    )
    grid = ar1.grid
    # array = ar1.array
    import matplotlib.pyplot as plt

    grid2 = [(item1.T, item2.T) for item1, item2 in grid]
    array = get_perlin_from_grid_numpy(
        grid, [1, 0.75, 0.5, 0.25], 42, False, "both", 512, 512
    )
    print(array.shape)
    print(ar1.array.shape)
    ar3 = array - ar1.array
    print(ar3.max())
    print(ar3.min())
    # plt.imshow(ar3, cmap="gray")
    # plt.show()
