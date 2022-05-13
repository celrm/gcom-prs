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
        for j in range(i+1, len(segments)):
            s1 = segments[i]
            s2 = segments[j]
            if s1.intersects(s2):
                intersections.append((i, j))
    return intersections


# ################################ PARTE 1 #####################################

#Generamos 1000 segmentos aleatorios, pero siempre serán los mismos

#Usaremos primero el concepto de coordenadas
X = []
Y = []

#Fijamos el modo aleatorio con una versión prefijada. NO MODIFICAR!!
random.seed(a=1, version=2)

#Generamos subconjuntos cuadrados del plano R2 para determinar los rangos de X e Y
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

#Representamos el Espacio topológico representado por los 1000 segmentos
        
for i in range(len(X)):
    plt.plot(X[i], Y[i], 'b')
plt.show()

segments = [ LineString([Point(X[s][0], Y[s][0]), Point(X[s][1], Y[s][1])]) for s in range(len(X)) ]

ds = DisjointSet()
ds.makeSet(range(len(X)))
for i,j in find_intersections(segments):
    ds.Union(i, j)

unique_segments_index = set([ ds.Find(i) for i in range(len(X)) ])
n_comps = len(unique_segments_index)
print(n_comps)

points_hull = { i:[] for i in unique_segments_index }
for i in range(len(X)):
    px, py = segments[i].xy
    points_hull[ds.Find(i)] += [[px[0], py[0]], [px[1], py[1]]]


colours = cm.rainbow(np.linspace(0, 1, len(X)))
for i in range(len(X)):
    plt.plot(*segments[i].xy, color=colours[ds.Find(i)])
# plt.show()
# 1

colours = cm.rainbow(np.linspace(0, 1, len(X)))
for i in unique_segments_index:
    points = np.array(points_hull[i])
    if len(points) < 3:
        continue
    hull = ConvexHull(points)
    plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'o', mec='r', color='none', lw=1, markersize=10)
    break

plt.show()
1