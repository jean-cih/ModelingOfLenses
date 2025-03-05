import numpy as np
import matplotlib.pyplot as plt

dpi = 100

tau = 2 * np.pi
t = np.linspace(0, 3, 100)
x_1 = 0.9 * np.sin(100 * tau * t)
#x_2 = 0.4 * np.sin(150 * tau * t)
x = x_1# + x_2
N = x.size

x = x * np.hamming(N)

fig = plt.figure(dpi = dpi, figsize = (512 / dpi, 384 / dpi) )
plt.plot(t, x)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
fig.savefig("x1.png")
plt.close()

X = np.fft.fft(x)

# t[1] - t[0] = sample rate
# 1/(t[1] - t[0]) = frequency
freq = np.linspace(0, 1 / (t[1] - t[0]), N)[: (N // 2)]

# 1 / N is a normalization factor
X_amp = (1/N) * np.abs(X)[: (N // 2)]
X_phs = (1/N) * np.angle(X)[: (N // 2)]

fig = plt.figure(dpi = dpi, figsize = (512 / dpi, 384 / dpi) )
plt.bar(freq, X_amp)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
fig.savefig("X_amp.png")
plt.close()

fig = plt.figure(dpi = dpi, figsize = (512 / dpi, 384 / dpi) )
plt.bar(freq, X_phs)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
fig.savefig("X_phs.png")
plt.close()

fig = plt.figure(dpi = dpi, figsize = (512 / dpi, 384 / dpi) )
x2 = np.fft.ifft(X)
plt.plot(t, x2.real)
plt.ylabel("Amplitude")
plt.xlabel("Time (s)")
fig.savefig("x2.png")
plt.close()
