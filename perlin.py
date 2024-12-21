from perlin_array import get_perlin_numpy

class PerlinArray:
    def __init__(self,shape : tuple,grid_size= 10,octaves=5,persistance=0.5,circular : bool = False,circular_axis=None,seed=None):
        width = shape[0]
        height = shape[1]

        match circular_axis:
            case [0,1] | None:
                axis = "both"
            case [0] | 0:
                axis = "horizontal"                
            case [1] | 1:
                axis = "vertical"
            case _:
                raise ValueError("Invalid circular_axis value")

        self.array = get_perlin_numpy(
            grid_size=grid_size,
            width=width,
            height=height,
            octaves=octaves,
            persistence=persistance,
            circular=circular,
            axis=axis,
            seed=seed)
    
    
if __name__ == "__main__":
    array = PerlinArray((1024,1024),grid_size=10)
    import matplotlib.pyplot as plt
    plt.imshow(array.array,cmap='gray') 
    plt.show()