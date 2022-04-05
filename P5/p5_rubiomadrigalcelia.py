import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

""" Apartado 1 """


def get_sphere(n, m):
    u = np.linspace(0.1, np.pi, n)
    v = np.linspace(0, 2 * np.pi, m)
    x = np.outer(np.sin(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.cos(u), np.ones_like(v))
    return x, y, z


def plot_sphere(x, y, z, ax=None, lim=3, alpha=1, cmap="gist_earth_r"):
    if ax == None:
        fig = plt.figure()
        fig.tight_layout()
        ax = Axes3D(fig)
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(-lim, lim)
    ax.plot_surface(x, y, z, cmap=cmap, alpha=alpha, rstride=1, cstride=1)
    return ax


def plot_curve(x2, y2, z2, ax=None, lim=3):
    if ax == None:
        fig = plt.figure()
        fig.tight_layout()
        ax = Axes3D(fig)
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(-lim, lim)
    ax.plot(x2, y2, z2 + 0.1, "-b", c="black", zorder=3)
    ax.scatter(x2, y2, z2 + 0.1, c=x2 + y2, cmap="jet")
    return ax


t2 = np.linspace(0.5, 1, 30)
x2 = abs(t2) * np.sin(t2 ** 3)
y2 = -abs(t2) * np.cos(2 * t2 ** 3)
z2 = np.sqrt(1 - x2 ** 2 - y2 ** 2)

x, y, z = get_sphere(30, 60)
ax = plot_sphere(x, y, z, alpha=0.66, lim=1.5)
plot_curve(x2, y2, z2, ax, lim=1 - 5)


def proj(x, y, z, alpha=1):
    eps = 1e-16
    aux = 1 / ((1 - z) ** alpha + eps)
    return aux * x, aux * y, 0 * z - 1


px, py, pz = proj(x, y, z, alpha=1 / 2)
px2, py2, pz2 = proj(x2, y2, z2, alpha=1 / 2)

plot_sphere(px, py, pz, ax, alpha=0.33, lim=1.5, cmap="gist_gray")
plot_curve(px2, py2, pz2, ax, lim=1.5)

ax.view_init(50, 10)
plt.savefig("1")


""" Apartado 2 """

from matplotlib import animation


def proj2(x, y, z, t):
    eps = 1e-16
    aux = 2 / (2 * (1 - t) + (1 - z) * t + eps)
    return aux * x, aux * y, -t + z * (1 - t)


def animate(t):
    xt, yt, zt = proj2(x, y, z, t)
    ax = Axes3D(fig)
    plot_sphere(xt, yt, zt, ax, alpha=1)
    ax.view_init(5, 45)
    return (ax,)


def init():
    return (animate(0),)


fig = plt.figure(figsize=(6, 6))
fig.tight_layout()
ani = animation.FuncAnimation(
    fig, animate, np.linspace(0, 0.9, 20), init_func=init, interval=20
)
ani.save("2.mp4", fps=5)
