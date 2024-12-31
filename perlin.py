from perlin_array import (
    get_perlin_numpy,
    get_perlin_grid_numpy,
    get_perlin_from_grid_numpy,
)
from typing import Tuple
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
        # array = (array - np.min(array)) / (np.max(array) - np.min(array))
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

    def expend(
        self,
        direction: str = "left",
        seed: int = None,
        output: str = "inplace",
    ):
        """
        Expand the Perlin noise grid in a specified direction, optionally generating new noise values.

        Parameters:

            direction (str): The direction to expand the grid. Must be one of ["left", "right", "top", "bottom"].
            seed (int, optional): Seed for random number generation. If not provided, a random seed is used.
            new_shape (Tuple[int], optional): Shape of the expanded grid. Defaults to the current grid dimensions.
            output (str): Specifies how the result should be returned:
                - "inplace" (default): Modifies the current grid in place.
                - "merged": Returns the merged array with the expanded grid.
                - "new": Returns only the new expanded grid without merging it.

        Raises:

            ValueError: If `direction` is not one of the allowed values.

        Returns:

            numpy.ndarray or None:

                - If `output` is "new", returns the newly generated array.
                - If `output` is "merged", returns the merged array.
                - If `output` is "inplace", modifies the instance array and returns None.

        Notes:
            - The function first validates the input parameters and calculates a new grid using `_get_grid`.
            - Depending on the `direction`, it adjusts the new grid's boundary to align with the current grid.
            - It can either update the existing grid in place, return a merged grid, or return only the new grid.

        Examples:
            Expand the grid to the right and update the instance grid in place:
            >>> instance.expend(direction="right")

            Expand the grid to the top and get the merged array:
            >>> merged_array = instance.expend(direction="top", output="merged")

            Generate a new grid expanded to the left without modifying the original:
            >>> new_array = instance.expend(direction="left", output="new")
        """
        new_shape = (self.width, self.height)
        if seed is None:
            seed = random.randint(0, 1147483647)
        if direction not in ["left", "right", "top", "bottom"]:
            raise ValueError("Invalid direction")
        new_grids = get_perlin_grid_numpy(
            self.grid_size,
            len(self.octaves),
            seed,
            self.circular,
            self.axis,
        )

        for (grid_x, grid_y), (new_grid_x, new_grid_y) in zip(self.grid, new_grids):
            match direction:
                case "left":
                    if new_shape[1] != self.height:
                        raise ValueError("Invalid shape")
                    new_grid_x[0, :] = grid_x[-1, :]
                    new_grid_y[0, :] = grid_y[-1, :]
                case "right":
                    if new_shape[1] != self.height:
                        raise ValueError("Invalid shape")
                    new_grid_x[-1, :] = grid_x[0, :]
                    new_grid_y[-1, :] = grid_y[0, :]
                case "top":
                    if new_shape[0] != self.width:
                        raise ValueError("Invalid shape")
                    new_grid_x[:, -1] = grid_x[:, 0]
                    new_grid_y[:, -1] = grid_y[:, 0]
                    assert new_grid_x[:, -1] == grid_x[:, 0]
                case "bottom":
                    if new_shape[0] != self.width:
                        raise ValueError("Invalid shape")
                    new_grid_x[:, 0] = grid_x[:, -1]
                    new_grid_y[:, 0] = grid_y[:, -1]
                case _:
                    pass

        new_array = self._perlin_from_grid_numpy(
            new_grids,
            self.octaves,
            self.seed,
            self.circular,
            self.axis,
            self.width,
            self.height,
        )
        if output == "new":
            return new_array, new_grids

        match direction:
            case "left":
                merged_array = np.hstack((self.array, new_array))
            case "right":
                merged_array = np.hstack((new_array, self.array))
            case "bottom":
                merged_array = np.vstack((new_array, self.array))
            case "top":
                merged_array = np.vstack((self.array, new_array))
        if output == "inplace":
            self.arary = merged_array
        elif output == "merged":
            return merged_array, new_grids

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ar1 = PerlinArray(
        (512, 512),
        grid_size=10,
        octaves=[1, 0.75, 0.5, 0.25],
        seed=42,
        circular_axis=[0, 1],
        circular=False,
    )
    c, grid = ar1.expend(seed=54, output="new", direction="top")
    print(grid[0][0])
    print("-" * 100)
    print(ar1.grid[0][0])
    # plt.imshow(c, cmap="gray")
    # plt.show()
