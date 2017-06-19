import numpy as np
from numpy.fft import fft, ifft, fftfreq, fftshift

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

N = 64
L = 2*np.pi    # domain size
a = 1          # wave speed
step = L/N

dt = 0.001
times = 10**4

x = np.linspace(0., L, N)

k = L*fftfreq(N, d=step)
# k = fftshift(np.arange(-N/2, N/2))
ks = k**2

u0 = np.exp(-100 * ((x-np.pi)**2))
# u0 = 2 + np.sin(x) + np.sin(2*x)
ut0 = np.zeros(N)
u = np.zeros((times, N))
u[0] = u0

uf = fft(u0)
uft = fft(ut0)

for i in range(1, times):
    uft_ = uft - a * dt * ks * uf
    uf = uf + 0.5 * dt * (uft + uft_)
    uft = uft_

    # # fixed boundary conditions
    u_ = np.real(ifft(uf))
    u_[0] = u_[-1] = 0
    uf = fft(u_)

    u[i, :] = np.real(ifft(uf))

#animation
fig, ax = plt.subplots()
line, = ax.plot(x, u0)

def animate(i):
    line.set_ydata(u[i])
    return line,

vs = 5
ani = animation.FuncAnimation(fig, animate, np.arange(1, times, vs), interval=1, blit=True, repeat_delay=500)

plt.show()