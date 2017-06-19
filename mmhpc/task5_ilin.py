import numpy as np
from numpy.fft import fft2, ifft2, fftfreq, fftshift

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
y = np.linspace(0., L, N)
xv, yv = np.meshgrid(x, y)

freq = L*fftfreq(N, d=step)
k1 = np.tile(freq, (N, 1))
k2 = k1.transpose()
ks = k1**2 + k2**2

u0 = np.exp(-100 * ((xv-np.pi)**2 + (yv-np.pi)**2))
ut0 = np.zeros((N, N))
u = np.zeros((times, N, N))
u[0, :, :] = u0

uf = fft2(u0)
uft = fft2(ut0)

for i in range(1, times) :
    uft_ = uft - a * dt * ks * uf
    uf = uf + 0.5 * dt * (uft + uft_)
    uft = uft_

    # fixed boundary conditions
    u0 = np.real(ifft2(uf))
    u0[0, :] = u0[-1, :] = 0
    u0[:, 0] = u0[:, -1] = 0
    uf = fft2(u0)

    u[i] = np.real(ifft2(uf))

#animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def animate(i):
    ax.clear()
    ax.set_zlim([0, 4])
    ax.plot_surface(xv, yv, u[i], cmap='coolwarm', linewidth=0, rstride=2, cstride=2)
    ax.set_title('%03d' % (i))
    return ax

vs = 50
ani = animation.FuncAnimation(fig, animate, np.arange(1, times, vs), interval=1, blit=False, repeat_delay=10)
plt.show()