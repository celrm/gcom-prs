# -*- coding: utf-8 -*-
"""
Plantilla
     
"""
import random
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
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

#Comprobamos que podemos seleccionar un par de segmentos:

#Segment A
PointA1 = Point(X[0][0], Y[0][0])
PointA2 = Point(X[0][1], Y[0][1])
#Segment B
PointB1 = Point(X[1][0], Y[1][0])
PointB2 = Point(X[1][1], Y[1][1])

SegmentA = LineString([PointA1, PointA2])
SegmentB = LineString([PointB1, PointB2])

#Y pintamos esos segmentos seleccionados para ver donde están

for i in range(len(X)):
    plt.plot(X[i], Y[i], 'b')

plt.plot(*SegmentA.xy, color="red")
plt.plot(*SegmentB.xy, color="red")
plt.show()

segments = [ LineString([Point(X[s][0], Y[s][0]), Point(X[s][1], Y[s][1])]) for s in range(len(X)) ]

ds = DisjointSet()
ds.makeSet(range(len(X)))
for i in range(len(X)):
    for j in range(len(X)):
        if segments[i].intersects(segments[j]):
            ds.Union(i,j)

n_comps = len( set([ ds.Find(i) for i in range(len(X)) ]) )
print(n_comps)


colours = cm.rainbow(np.linspace(0, 1, len(X)))
for i in range(len(X)):
    plt.plot(*segments[i].xy, color=colours[ds.Find(i)])
plt.show()
1
