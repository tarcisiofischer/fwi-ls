import numpy as np
from scipy.optimize.nonlin import newton_krylov
import functools
from matplotlib import pyplot as plt

length = 1.0
xx = 40
ds = length / xx
dt = 0.001

f_M = 1.0
def ricker_time_domain(t):
    return (1.0 - 2 * np.pi ** 2. * f_M * t ** 2.0) * np.power(np.e, -np.pi ** 2. * f_M ** 2. * t ** 2.)

def ricker_freq_domain(omega):
    return (2. / np.sqrt(np.pi)) * (omega ** 2. / f_M ** 3.0) * np.power(np.e, -omega ** 2. / f_M ** 2.)
 
def residual(X_1d, Xo_1d, Xoo_1d, t, c):
    X = X_1d.reshape(xx, xx)
    Xo = Xo_1d.reshape(xx, xx)
    Xoo = Xoo_1d.reshape(xx, xx)
    residual = np.zeros_like(X) * np.nan
    extended_X = np.zeros(shape=(xx + 2, xx + 2))
    extended_X[1:-1, 1:-1] = X
    extended_X[0:1, :] = 0.0
    extended_X[-1:, :] = 0.0
    extended_X[:, 0:1] = 0.0
    extended_X[:, -1:] = 0.0
    X = extended_X
     
    source_f = lambda t, i, j: ricker_time_domain(t) if i == 0 and j == 0 else 0.0
 
    for i in range(xx):
        for j in range(xx):
            ii = i + 1
            jj = j + 1
            residual[i, j] = (
                (X[ii + 1, jj] - 2 * X[ii, jj] + X[ii - 1, jj]) / ds ** 2. +
                (X[ii, jj + 1] - 2 * X[ii, jj] + X[ii, jj - 1]) / ds ** 2. -
                (1. / c ** 2.0) * (X[ii, jj] - 2 * Xo[i, j] + Xoo[i, j]) / dt ** 2
                + source_f(t, i, j)
            )
 
    assert not np.any(np.isnan(residual))
 
    return residual.reshape(xx * xx)
 
X = np.zeros(shape=(xx * xx,))
Xo = np.zeros(shape=(xx * xx,))
Xoo = np.zeros(shape=(xx * xx,))
t = 0.0
for i in range(50):
    print(i)
    s = newton_krylov(functools.partial(residual, t=t, c=10.0, Xo_1d=Xo, Xoo_1d=Xoo), X)
    t += dt
    Xoo[:] = Xo
    Xo[:] = s
plt.imshow(s.reshape(xx, xx))
plt.show()
