import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt


def pl_wf(x,y,v, title):
    fig = plt.figure(title)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, v)
    ax.set(xlabel='x', ylabel='y', zlabel=title)
    # fig.show()


N = 512      # number of poits
L = 10.      # domain size
k = (2 * np.pi/L) * fftfreq(N)

step = L/N
kx, ky = np.meshgrid(k, k)
delsq  = -(kx**2 + ky**2)
delsq[0, 0] = 1

x = np.linspace(0., L, N)
y = np.linspace(0., L, N)
xv, yv = np.meshgrid(x, y)

charge = np.zeros((N, N))
charge[N//4:N//3, N//4:N//3] = 1.
charge[2*N//3:3*N//4, 2*N//3:3*N//4] = -1.

pl_wf(xv, yv, charge, 'charge')

# calculate potential
uf = fft2(charge)
uf0 = np.real(ifft2(uf/delsq))
uf0[0, :] = uf0[-1, :] = 0
uf0[:, 0] = uf0[:, -1] = 0
uf = fft2(uf0)
potential = np.real(ifft2(uf/delsq))
# pot[0, 0] = 0
pl_wf(xv, yv, potential, 'potential')

# calculate field
field = np.zeros((N, N))
field[:, 1:-2] = potential[:, 0:-3] - potential[:, 2:-1]
field[1:-2, :] += potential[0:-3, :] - potential[2:-1, :]
# fixed boundary conditions
field[0, :] = field[-1, :] = 0
field[:, 0] = field[:, -1] = 0
pl_wf(xv, yv, field, 'field')

plt.show()