import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-10.0, 10.0, 2000)
f_M = 1.0

def ricker_time_domain(t):
    return (1.0 - 2 * np.pi ** 2. * f_M ** 2. * t ** 2.) * \
        np.power(np.e, -np.pi ** 2. * f_M ** 2. * t ** 2.)

def ricker_freq_domain(omega):
    return (2. / np.sqrt(np.pi)) * \
        (omega ** 2. / f_M ** 3.) * \
        np.power(np.e, -omega ** 2. / f_M ** 2.)

ricker_in_time = ricker_time_domain(t)
sample_freq = np.fft.fftfreq(ricker_in_time.size, d=t[1] - t[0])
fft_ricker = np.fft.fft(ricker_in_time)
print(fft_ricker)
#inverse_fft_ricker = np.fft.ifft(fft_ricker)

# sample_freq = []
ricker_in_freq = ricker_freq_domain(sample_freq)
inverse_fft_ricker = np.fft.ifft(ricker_in_freq)

# I don't know why this FACTOR is necessary. I just observed that
# dividing it will fix the y-axis from the obtained fft.
# http://lagrange.univ-lyon1.fr/docs/numpy/1.11.0/reference/routines.fft.html#normalization
FACTOR = 100.0
 
f, (ax1, ax2) = plt.subplots(2, 1, sharey=False)
# ax1.plot(sample_freq, np.abs(fft_ricker.real) / FACTOR)
ax1.plot(sample_freq, ricker_in_freq, 'r--')
ax1.set_xlim(0.0, 4.0)
ax1.grid(True)
 
# ax2.plot(t, ricker_in_time)
ax2.plot(t, inverse_fft_ricker.real, 'r--')
# ax2.set_xlim(-2.5, 2.5)
ax2.grid(True)
 
plt.show()
