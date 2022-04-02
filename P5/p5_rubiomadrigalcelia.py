import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import math

""" Apartado 1 """

u = np.linspace(0, np.pi, 25)
v = np.linspace(0, 2 * np.pi, 50)

x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))


t2 = np.linspace(0.5, 1, 30)
x2 = abs(t2) * np.sin(t2 ** 3)
y2 = -abs(t2) * np.cos(2 * t2 ** 3)
z2 = np.sqrt(1 - x2 ** 2 - y2 ** 2)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)

ax.plot_surface(x, y, z, cmap="gist_earth", alpha=0.5, rstride=1, cstride=1)

ax.plot(x2, y2, z2, "-b", c="gray", zorder=3)
ax.scatter(x2, y2, z2 + 0.01, c=x2 + y2, cmap="jet")


def proj(x, z, z0=1, alpha=1):
    z0 = z * 0 + z0
    eps = 1e-16
    x_trans = x / (abs(z0 - z) ** alpha + eps)
    return x_trans


px = proj(x, z, alpha=1 / 2)
py = proj(y, z, alpha=1 / 2)
pz = 0 * px - 1
px2 = proj(x2, z2, alpha=1 / 2)
py2 = proj(y2, z2, alpha=1 / 2)
pz2 = 0 * px2 - 1

ax.plot(px2, py2, pz2, "-b", c="gray", zorder=3)
ax.scatter(px2, py2, pz2, c=x2 + y2, cmap="jet")

ax.set_title("surface")
plt.show()
1

""" Apartado 2 """

from matplotlib import animation

def animate(t):
    xt = proj(x, z) * t + x * (1-t)
    yt = proj(y, z) * t + y * (1-t)
    zt = (z * 0) * t + z * (1-t)
    x2t = proj(x2, z2) * t + x2 * (1-t)
    y2t = proj(y2, z2) * t + y2 * (1-t)
    z2t = t + z2 * (1-t)

    ax = plt.axes(projection="3d")
    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(-3, 3)
    ax.set_zlim3d(-3, 3)
    ax.plot_surface(
        xt, yt, zt, rstride=1, cstride=1, alpha=0.5, cmap="viridis", edgecolor="none"
    )
    ax.plot(x2t, y2t, z2t, "-b", c="gray")
    return (ax,)


def init():
    return (animate(0),)


animate(np.arange(0, 1, 0.1)[1])
plt.show()

fig = plt.figure(figsize=(6, 6))
ani = animation.FuncAnimation(
    fig, animate, np.arange(0, 1, 0.05), init_func=init, interval=20
)
ani.save("s.mp4", fps=5)
# ani.save("ejemplo.gif", fps = 5)
