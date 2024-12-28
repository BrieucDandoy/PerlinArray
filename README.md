# Perlin Array

This library builds perlin noise 2D numpy array using rust.

example 
```python

from perlin_array import PerlinArray
import matplotlib.pyplot as plt

perlin = PerlinArray(shape = (512,512),grid_size=10,octaves=[1,0.75,0.5],seed=42)
plt.imshow(perlin.array,cmap="gray")
plt.show()
```