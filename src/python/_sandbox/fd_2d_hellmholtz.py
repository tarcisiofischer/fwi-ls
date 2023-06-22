import numpy as np
from scipy.optimize.nonlin import newton_krylov
import functools
from matplotlib import pyplot as plt

length = 1.0
xx = 40
ds = length / xx

f_M = 1.0
def ricker_time_domain(t):
    return (1.0 - 2 * np.pi ** 2. * f_M * t ** 2.0) * np.power(np.e, -np.pi ** 2. * f_M ** 2. * t ** 2.)

def ricker_freq_domain(omega):
    return 1e+6 * (2. / np.sqrt(np.pi)) * (omega ** 2. / f_M ** 3.0) * np.power(np.e, -omega ** 2. / f_M ** 2.)
 
def residual(X_1d, omega, c):
    X = X_1d.reshape(xx, xx)
    residual = np.zeros_like(X) * np.nan
    extended_X = np.zeros(shape=(xx + 2, xx + 2))
    extended_X[1:-1, 1:-1] = X
    extended_X[0:1, :] = 0.0
    extended_X[-1:, :] = 0.0
    extended_X[:, 0:1] = 0.0
    extended_X[:, -1:] = 0.0
    X = extended_X
     
#     source_f = lambda omega, i, j: ricker_freq_domain(omega) if i == 0 and j == 0 else 0.0
    source_f = lambda omega, i, j: 1.0 if i == 0 and j == 0 else 0.0
 
    for i in range(xx):
        for j in range(xx):
            ii = i + 1
            jj = j + 1
            residual[i, j] = (
                + (X[ii + 1, jj] - 2 * X[ii, jj] + X[ii - 1, jj]) / ds ** 2.
                + (X[ii, jj + 1] - 2 * X[ii, jj] + X[ii, jj - 1]) / ds ** 2.
                - (omega ** 2.0 / c(i, j) ** 2.0) * X[ii, jj]
                + source_f(omega, i, j)
            )
 
    assert not np.any(np.isnan(residual))
 
    return residual.reshape(xx * xx)

results = []
for omega in np.linspace(0.0, 5.5, 20):
    print(omega)

#     def c_func(x, y):
#         return 0.001 if x in range(5, 20) and y in range(5, 20) else 0.1
    def c_func(x, y):
        return 1.0

    domain = []
    for i in range(xx):
        domain.append([])
        for j in range(xx):
            domain[-1].append(c_func(i, j))
#     plt.imshow(domain)
#     plt.show()

    X = np.zeros(shape=(xx * xx,))
    s = newton_krylov(functools.partial(residual, omega=omega, c=c_func), X)
    s = s.reshape(xx, xx)
    results.append([omega, s[15, 0]])

    plt.imshow(s)
    plt.show()
results = np.array(results)

plt.plot(results[:,0], results[:,1])
plt.show()
