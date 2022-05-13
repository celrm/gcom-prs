import numpy as np
from numpy import pi, cos, sin, sqrt, outer, ones
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

"""
2-esfera
"""
# latitudes
u = np.linspace(0, np.pi, 30)
# longitudes
v = np.linspace(0, 2 * np.pi, 60)

x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))


"""
Vector en la variedad de la 2-esfera
Definimos un punto origen (o) y una dirección (p) 
"""

o_phi_O = 0  # longitud
o_theta_A = 0  # latitud del punto original. Por lo tanto es THETHA0 (!!)
o_theta_B = -0.2*pi  # latitud del punto original. Por lo tanto es THETHA0 (!!)

p_phi_O = 0  # longitud
p_theta_O = pi / 5  # latitud
p_norm_O = sqrt((p_phi_O ** 2) * cos(p_theta_O) ** 2 + p_theta_O ** 2)

"""
Trasladamos paralelamente el bipunto anterior
"""

def transp(th0, ph, th, t, alpha=2):
    ph2 = th * np.sin((np.sin(th0)) * (ph) * (t**alpha)) / np.cos(th0)
    th2 = th * np.cos((np.sin(th0)) * (ph) * (t**alpha))
    return ph2, th2

def familia_param(o_theta_C, t, alpha=2):
    Dphi = math.tau
    o_phi2 = p_phi_O + Dphi * (t**alpha)
    o_theta2 = o_theta_C
    p_phi2, p_theta2 = transp(th0=o_theta_C, ph=Dphi, th=p_theta_O, t=t, alpha=alpha)
    return o_phi2, o_theta2, p_phi2, p_theta2


"""
CAMBIAMOS EL SISTEMA DE REFERENCIA PARA LA REPRESENTACIÓN
"""

phi0 = np.pi / 4

"""
VISTO COMO BIPUNTO
"""

def get_bipunto(o_phi2, o_theta2, p_phi2, p_theta2):
    o2 = np.array(
        [
            np.cos(o_theta2) * np.cos(o_phi2 - phi0),
            np.cos(o_theta2) * np.sin(o_phi2 - phi0),
            np.sin(o_theta2),
        ]
    )
    p2 = np.array(
        [
            np.cos(o_theta2 + p_theta2) * np.cos(o_phi2 + p_phi2 - phi0),
            np.cos(o_theta2 + p_theta2) * np.sin(o_phi2 + p_phi2 - phi0),
            np.sin(o_theta2 + p_theta2),
        ]
    )
    return np.concatenate((o2, p2 - o2))


"""
Curva (el paralelo) donde queremos transladar
"""


def paralelo(o_theta_C):
    phis = np.linspace(0, 2 * pi, 100)
    thetas = np.ones_like(phis) * o_theta_C
    return np.array(
        [
            np.cos(thetas) * np.cos(phis - phi0),
            np.cos(thetas) * np.sin(phis - phi0),
            np.sin(thetas),
        ]
    )


gamma_A = paralelo(o_theta_A)
gamma_B = paralelo(o_theta_B)

"""
FIGURA
"""

from matplotlib import animation


def animate(t):
    ax = plt.axes(projection="3d")
    ax.clear()
    fig.tight_layout()
    ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha = 0.4,
                    cmap='gist_earth', edgecolor='none')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.plot(gamma_A[0], gamma_A[1], gamma_A[2], "-b", c="black", zorder=3)
    ax.plot(gamma_B[0], gamma_B[1], gamma_B[2], "-b", c="black", zorder=3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(-10, 300)

    o_phi_a, o_theta_a, p_phi_a, p_theta_a = familia_param(o_theta_A, t)
    o_phi_b, o_theta_b, p_phi_b, p_theta_b = familia_param(o_theta_B, t)
    X_A, Y_A, Z_A, U_A, V_A, W_A = get_bipunto(o_phi_a, o_theta_a, p_phi_a, p_theta_a)
    X_B, Y_B, Z_B, U_B, V_B, W_B = get_bipunto(o_phi_b, o_theta_b, p_phi_b, p_theta_b)
    plt.quiver(
        X_A,
        Y_A,
        Z_A,
        U_A,
        V_A,
        W_A,
        colors="red",
        zorder=3,
        color="red",
        arrow_length_ratio=0.3,
    )
    plt.quiver(
        X_B,
        Y_B,
        Z_B,
        U_B,
        V_B,
        W_B,
        colors="green",
        zorder=3,
        color="green",
        arrow_length_ratio=0.3,
    )
    return (ax,)


fig = plt.figure()
fig.tight_layout()
ax = plt.axes(projection="3d")


def init():
    ax = plt.axes(projection="3d")
    return (ax,)

ani = animation.FuncAnimation(
    fig, animate, np.linspace(0, 1, 40), init_func=init, interval=10
)
ani.save("2.mp4", fps=5)