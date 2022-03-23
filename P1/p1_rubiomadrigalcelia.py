# Práctica 1 de Celia Rubio Madrigal

import numpy as np

""" Funciones """

# Función logística
# R es global
def logistica(x):
    return R * x * (1 - x)


# Calcula la subórbita de f desde x0 aplicada n veces
# Mejorada a O(n)
def orbita(x0, f, n):
    orb = np.empty([n])
    orb[0] = f(x0)
    for i in range(n - 1):
        orb[i + 1] = f(orb[i])
    return orb


# Calcula el m tal que el diámetro de la órbita no cambia más que epsilon
# desde m hasta m+N, donde N es una constante global predefinida
def tiempo_transitorio(orb, epsilon=0.001):
    n = len(orb)
    m = 0
    next_sup, next_inf = max(orb[N : 2 * N]), min(orb[N : 2 * N])
    while m + N <= n:
        sup, inf = max(orb[m : m + N]), min(orb[m : m + N])
        if abs((sup - inf) - (next_sup - next_inf)) < epsilon:
            return m + N
        m += N
        next_sup, next_inf = sup, inf
    return n


# Calcula el periodo de la subórbita, suponiendo que el tiempo transitorio ya es estable
def periodo(suborb, epsilon=0.001):
    n = len(suborb)
    for i in range(1, n):
        if abs(suborb[n - 1] - suborb[n - i - 1]) < epsilon:
            return i
    return n


# Calcula la cuenca de atracción de f desde x0
def atrac(x0, f, epsilon=0.001):
    orb = orbita(x0, f, N0)
    m = tiempo_transitorio(orb, epsilon)
    suborb = orb[-m:]
    p = periodo(suborb, epsilon)
    v0 = np.sort(suborb[-p:])
    return v0,m


eps = 0.001  # predefinido el error
N0 = 200  # nuestra capacidad de cómputo máxima
N = 20  # sabemos de antemano que nuestro conjunto será menor
X0 = 0.2

""" Apartado i) """

# Comprueba si dos vectores son iguales con precisión de epsilon
def equals_vectors(v1, v2, epsilon=0.001):
    if len(v1) != len(v2):
        return False
    for i, j in zip(v1, v2):
        if abs(i - j) >= epsilon:
            return False
    return True


# Comprueba la estabilidad (moviendo x0) y las bifurcaciones (moviendo R)
def test_v0(v0, epsilon=0.001):
    global R
    r = R
    print(v0)

    # Error de x0: en estos casos, todos los valores de x0 dan el mismo resultado
    for x0 in np.arange(epsilon, 1, epsilon):
        v1,_ = atrac(x0, logistica, epsilon)
        if not equals_vectors(v0, v1, epsilon):
            print("X0", x0, "no estable")

    # Error de R: mover R hasta que no dé el mismo resultado
    for delta in np.arange(epsilon, 1, epsilon):
        R = r + delta
        v1,_ = atrac(X0, logistica, epsilon)
        if not equals_vectors(v0, v1, epsilon):
            print("R", R, "es distinto (delta=", delta, ")")
            print("\t", v0)
            print("\t", v1)
            break
        R = r - delta
        v2,_ = atrac(X0, logistica, epsilon)
        if not equals_vectors(v0, v2, epsilon):
            print("R", R, "es distinto (delta=", delta, ")")
            print("\t", v0)
            print("\t", v2)
            break
    R = r


print("-" * 10)
print("Conjunto atractor a)")


R = 3.141
v0,m = atrac(X0, logistica, eps)
print("Pasos M=",m)
test_v0(v0, eps)

print("-" * 10)
print("Conjunto atractor b)")

R = 3.515
v0,m = atrac(X0, logistica, eps)
print("Pasos M=",m)
test_v0(v0, eps)


""" Apartado ii) """
print("-" * 10)

rss = []
vss = {}

first = True
for r in np.arange(3.544, 4, eps):
    R = r
    v0,m = atrac(X0, logistica, eps)

    # Primero, calculo si la cuenca de atracción tiene 8 elementos
    if len(v0) == 8:
        # Y después, si al mover x0 con error epsilon se mantiene el resultado
        test = True
        v1,_ = atrac(X0 + eps, logistica, eps)
        if not equals_vectors(v0, v1, eps):
            test = False
        v1,_ = atrac(X0 - eps, logistica, eps)
        if equals_vectors(v0, v1, eps):
            rss += [r]
            vss[r] = v0
            print("R", R, "\t", v0)
            print("Pasos M=",m) 
    elif first and len(v0) > 8:
        first = False
        rss += [r]
        vss[r] = v0
        print("NO R", R, "\t", v0)

""" Gráfica """

import matplotlib.pyplot as plt

plt.figure(figsize=(7.5, 10))

for r in rss:
    n = len(vss[r])
    plt.scatter([r] * n, vss[r], color="red")
plt.xlabel("r")
plt.ylabel("V0")

plt.axvline(x=3.564, ls="--")
plt.savefig("grafica.png")
