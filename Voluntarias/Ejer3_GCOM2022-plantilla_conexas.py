# -*- coding: utf-8 -*-
"""
Plantilla
     
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point

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

#Comprobamos la función "intersects"

print(SegmentA.intersects(SegmentB))

