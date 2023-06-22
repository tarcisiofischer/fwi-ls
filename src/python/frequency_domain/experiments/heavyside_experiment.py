import numpy as np
from frequency_domain.function_builders import HeavySideBasedFunctionBuilder,\
    SmoothHeavySideBasedFunctionBuilder


N = 20 # Number of mesh elements in each coordinate
LENGTH = 5.0 # [km]
C0 = 1.0
C1 = 3.5

x = np.linspace(0.0, LENGTH, N)
y = np.linspace(0.0, LENGTH, N)
X, Y = np.meshgrid(x, y)
phi = (X**2 + Y**2) / 50.0 - 0.5

flavor = 'SmoothHeavySide'
# flavor = 'HeavySide'

if flavor == 'HeavySide':
    f = HeavySideBasedFunctionBuilder(
        phi,
        LENGTH,
        LENGTH,
        -2.0,
        +2.0,
    ).build()
elif flavor == 'SmoothHeavySide':
    f = SmoothHeavySideBasedFunctionBuilder(
        phi,
        LENGTH,
        LENGTH,
        -2.0,
        +2.0,
        0.1
    ).build()

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # @unusedimport: Necessary for 3d plotting

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, phi)
ax.plot_surface(X, Y, f(x, y))
fig = plt.figure()
ax = fig.gca()
ax.imshow(f(x, y))
plt.show()
