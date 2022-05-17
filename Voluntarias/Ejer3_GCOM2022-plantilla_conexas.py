# -*- coding: utf-8 -*-
"""
Plantilla
     
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from scipy.spatial import ConvexHull
from matplotlib import cm

# A class to represent a disjoint set
class DisjointSet:
    parent = {}
    rank = {}

    def makeSet(self, universe):
        for i in universe:
            self.parent[i] = i
            self.rank[i] = 0

    def Find(self, k):
        if self.parent[k] != k:
            self.parent[k] = self.Find(self.parent[k])
        return self.parent[k]

    def Union(self, a, b):
        x = self.Find(a)
        y = self.Find(b)

        if x == y:
            return
        if self.rank[x] > self.rank[y]:
            self.parent[y] = x
        elif self.rank[x] < self.rank[y]:
            self.parent[x] = y
        else:
            self.parent[x] = y
            self.rank[y] = self.rank[y] + 1


def find_intersections(segments):
    """
    Finds the intersections between segments.
    """
    intersections = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            s1 = segments[i]
            s2 = segments[j]
            if s1.intersects(s2):
                intersections.append((i, j))
    return intersections


def connected_components(xy=None, segments=None):
    """
    Finds the connected components of a system.
    """
    if xy is None and segments is None:
        return ()
    if segments is None:
        segments = [ LineString(
                [Point(xy[0][s][0], xy[1][s][0]), Point(xy[0][s][1], xy[1][s][1])]
            ) for s in range(len(xy[0])) ]

    ds = DisjointSet()
    ds.makeSet(range(len(segments)))
    all_intersections = find_intersections(segments)
    for i, j in all_intersections:
        ds.Union(i, j)
    return len(ds.parent), ds, segments


# ################################ PARTE 1 #####################################

# Generamos 1000 segmentos aleatorios, pero siempre serán los mismos

# Usaremos primero el concepto de coordenadas
X = []
Y = []

# Fijamos el modo aleatorio con una versión prefijada. NO MODIFICAR!!
random.seed(a=1, version=2)

# Generamos subconjuntos cuadrados del plano R2 para determinar los rangos de X e Y
xrango1 = random.sample(range(100, 1000), 200)
xrango2 = list(np.add(xrango1, random.sample(range(10, 230), 200)))
yrango1 = random.sample(range(100, 950), 200)
yrango2 = list(np.add(yrango1, random.sample(range(10, 275), 200)))

for j in range(len(xrango1)):
    for i in range(5):
        random.seed(a=i, version=2)
        xrandomlist = random.sample(range(xrango1[j], xrango2[j]), 4)
        yrandomlist = random.sample(range(yrango1[j], yrango2[j]), 4)
        X.append(xrandomlist[0:2])
        Y.append(yrandomlist[2:4])

# Representamos el Espacio topológico representado por los 1000 segmentos
for i in range(len(X)):
    plt.plot(X[i], Y[i], "b")
plt.savefig("1")

n_comp, ds, segments = connected_components(xy=(X, Y))

# Coloreamos los segmentos conexos del mismo color
colours = cm.rainbow(np.linspace(0, 1, len(X)))
for i in range(len(X)):
    plt.plot(*segments[i].xy, color=colours[ds.Find(i)])
plt.savefig("2")

points_hull = {i: [] for i in ds.parent}
for i in range(len(X)):
    px, py = segments[i].xy
    points_hull[ds.Find(i)] += [[px[0], py[0]], [px[1], py[1]]]

for i in ds.parent:
    points = np.array(points_hull[i])
    if len(points) < 3:
        continue
    hull = ConvexHull(points)
    plt.plot(
        points[hull.vertices, 0],
        points[hull.vertices, 1],
        "o",
        mec="r",
        color="none",
        lw=1,
        markersize=10,
    )

plt.savefig("3")
