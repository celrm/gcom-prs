import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib import animation

delta = 10 ** (-3)

# q = variable de posición, dq0 = \dot{q}(0) = valor inicial de la derivada
# d = granularidad del parámetro temporal
def deriv(q, dq0, d=delta):
    dq = (q[1 : len(q)] - q[0 : (len(q) - 1)]) / d
    dq = np.insert(dq, 0, dq0)
    return dq


# Ecuación de un sistema dinámico continuo
def F(q):
    ddq = -2 * q * (q ** 2 - 1)
    return ddq


# Resolución de la ecuación dinámica \ddot{q} = F(q), obteniendo la órbita q(t)
# Los valores iniciales son la posición q0 := q(0) y la derivada dq0 := \dot{q}(0)
def orb(n, q0, dq0, F, args=None, d=delta):
    q = np.empty([n + 1])
    q[0] = q0
    q[1] = q0 + dq0 * d
    for i in np.arange(2, n + 1):
        args = q[i - 2]
        q[i] = -q[i - 2] + d ** 2 * F(args) + 2 * q[i - 1]
    return q


#################################################################
#  CÁLCULO DE ÓRBITAS
#################################################################

## Pintamos el espacio de fases
def simplectica(q0, dq0, F, col=0, d=delta, n=int(10 / delta)):
    q = orb(n, q0=q0, dq0=dq0, F=F, d=d)
    dq = deriv(q, dq0=dq0, d=d)
    p = dq / 2
    plt.plot(q, p, c=plt.get_cmap("winter")(col))


""" Apartado i) """

fig = plt.figure(figsize=(8, 5))
plt.xlim(-2.2, 2.2)
plt.ylim(-1.2, 1.2)
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("q(t)", fontsize=12)
ax.set_ylabel("p(t)", fontsize=12)
# Condiciones iniciales:
seq_q0 = np.linspace(0.1, 1.0, num=4)
seq_dq0 = np.linspace(0.1, 2.0, num=4)
for i, q0 in enumerate(seq_q0):
    for j, dq0 in enumerate(seq_dq0):
        col = (1 + i + j * (len(seq_q0))) / (len(seq_q0) * len(seq_dq0))
        simplectica(
            q0=q0,
            dq0=dq0,
            F=F,
            col=col,
            d=delta,
            n=int(20 / delta),
        )
plt.savefig("1")

""" Apartado ii) """

fig = plt.figure(figsize=(8, 5))
plt.xlim(-2.2, 2.2)
plt.ylim(-1.2, 1.2)
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("q(t)", fontsize=12)
ax.set_ylabel("p(t)", fontsize=12)
# Condiciones iniciales:
seq_q0 = np.linspace(0.0, 1.0, num=10)
seq_dq0 = np.linspace(0.0, 2.0, num=10)
t = 0.25
ds = np.linspace(10 ** (-3), 10 ** (-4), num=10)
for d in ds:
    n = int(t / d)
    qall, pall, qbi, pbi, qbj, pbj = (
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    )
    for i, q0 in enumerate(seq_q0):
        for j, dq0 in enumerate(seq_dq0):
            q = orb(n, q0=q0, dq0=dq0, F=F, d=d)
            dq = deriv(q, dq0=dq0, d=d)
            p = dq / 2
            qall = np.append(qall, q[-1])
            pall = np.append(pall, p[-1])
            if i == len(seq_q0) - 1:
                qbi = np.append(qbi, q[-1])
                pbi = np.append(pbi, p[-1])
            if j == 0:
                qbj = np.append(qbj, q[-1])
                pbj = np.append(pbj, p[-1])
    ax.clear()
    totalhull = ConvexHull(np.array([qall, pall]).T)
    convex_hull_plot_2d(totalhull,ax=ax)
    ihull = ConvexHull(np.array([qbi, pbi]).T)
    convex_hull_plot_2d(ihull,ax=ax)
    jhull = ConvexHull(np.array([qbj, pbj]).T)
    convex_hull_plot_2d(jhull, ax=ax)
    area = totalhull.volume - ihull.volume - jhull.volume
    print("Área %f:\t" % d, area)
    plt.xlim(-2.2, 2.2)
    plt.ylim(-1.2, 1.2)
    plt.savefig("figs/2-%f.png" % d)

""" Apartado iii) """

seq_q0 = np.linspace(0.0, 1.0, num=20)
seq_dq0 = np.linspace(0.0, 2.0, num=20)
t = 5
n = int(t / delta)
# Guardar puntos:
q3 = np.empty([len(seq_q0), len(seq_dq0), n + 1])
p3 = np.empty([len(seq_q0), len(seq_dq0), n + 1])
for i, q0 in enumerate(seq_q0):
    for j, dq0 in enumerate(seq_dq0):
        q = orb(n, q0=q0, dq0=dq0, F=F, d=delta)
        dq = deriv(q, dq0=dq0, d=delta)
        p = dq / 2
        q3[i][j] = q
        p3[i][j] = p


def animate(t):
    n = int(t / delta)
    ax.clear()
    ax.set_xlabel("q(t)", fontsize=12)
    ax.set_ylabel("p(t)", fontsize=12)
    for i, q0 in enumerate(seq_q0):
        for j, dq0 in enumerate(seq_dq0):
            qt = q3[i][j][n]
            pt = p3[i][j][n]
            plt.plot(
                qt,
                pt,
                marker="o",
                markersize=8,
            )
    plt.xlim(-2.2, 2.2)
    plt.ylim(-1.2, 1.2)
    return (ax,)


fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(1, 1, 1)
ani = animation.FuncAnimation(fig, animate, np.linspace(delta, 5, 20))
ani.save("3.mp4", fps=5)
