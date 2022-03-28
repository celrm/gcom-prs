# -*- coding: utf-8 -*-
"""
Referencias:
    
    Fuente primaria del reanálisis
    https://psl.noaa.gov/data/gridded/data.ncep.reanalysis2.pressure.html
    
    Altura geopotencial en niveles de presión
    https://psl.noaa.gov/cgi-bin/db_search/DBListFiles.pl?did=59&tid=97457&vid=1498
    
    Temperatura en niveles de presión:
    https://psl.noaa.gov/cgi-bin/db_search/DBListFiles.pl?did=59&tid=97457&vid=4237

    Temperatura en niveles de superficie:
    https://psl.noaa.gov/cgi-bin/db_search/DBListFiles.pl?did=59&tid=97457&vid=1497
    
"""

import math
import datetime as dt  # Python standard library datetime  module
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from sklearn.decomposition import PCA

LEV = 30
CMAP = "jet"


f = Dataset("air.2021.nc", "r", format="NETCDF4")
level = f.variables["level"][:].copy()
lats = f.variables["lat"][:].copy()
lons = f.variables["lon"][:].copy()
air21 = f.variables["air"][:].copy()
f.close()
f = Dataset("hgt.2021.nc", "r", format="NETCDF4")
time21 = f.variables["time"][:].copy()
hgt21 = f.variables["hgt"][:].copy()
f.close()


f = Dataset("air.2022.nc", "r", format="NETCDF4")
air22 = f.variables["air"][:].copy()
f.close()
f = Dataset("hgt.2022.nc", "r", format="NETCDF4")
time22 = f.variables["time"][:].copy()
hgt22 = f.variables["hgt"][:].copy()
f.close()

# Cambio de coordenadas

air21[:,:,:,lons >= 180], air21[:,:,:,lons < 180] = air21[:,:,:,lons < 180], air21[:,:,:,lons >= 180]
air22[:,:,:,lons >= 180], air22[:,:,:,lons < 180] = air22[:,:,:,lons < 180], air22[:,:,:,lons >= 180]
hgt21[:,:,:,lons >= 180], hgt21[:,:,:,lons < 180] = hgt21[:,:,:,lons < 180], hgt21[:,:,:,lons >= 180]
hgt22[:,:,:,lons >= 180], hgt22[:,:,:,lons < 180] = hgt22[:,:,:,lons < 180], hgt22[:,:,:,lons >= 180]

lons = np.roll(lons, len(lons[lons < 180]))
lons[lons >= 180] -= 360


""" Apartado i) """


"""
Distribución espacial de la temperatura en el nivel de 500hPa, para el primer día
"""
cs = plt.contourf(lons, lats, air21[0, 0, :, :], cmap=CMAP, levels=LEV)
plt.colorbar(cs)
plt.savefig("1")

hgt21b = hgt21[:, level == 500.0, :, :].reshape(len(time21), len(lats) * len(lons))
n_components = 4


Y = hgt21b.transpose()
pca = PCA(n_components=n_components)

Element_pca0 = pca.fit_transform(Y)
Element_pca0 = Element_pca0.transpose(1, 0).reshape(n_components, len(lats), len(lons))
print(pca.explained_variance_ratio_)

fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))
fig.tight_layout()
plt.subplots_adjust(top=0.90)
for i in range(4):
    ax = axs[i]
    ax.set_title("PCA-" + str(i), fontsize=15, ha="center")
    cs = ax.contourf(lons, lats, Element_pca0[i, :, :], cmap=CMAP, levels=LEV)
    fig.colorbar(cs, ax=ax)

plt.savefig("2")


""" Apartado ii) """


dt_time21 = np.array([dt.date(1800, 1, 1) + dt.timedelta(hours=t) for t in time21])
dt_time22 = np.array([dt.date(1800, 1, 1) + dt.timedelta(hours=t) for t in time22])

lats_logic = np.logical_and(lats > 30, lats < 50)
lons_logic = np.logical_and(lons > -20, lons < 20)

hgt_sub = hgt21[:, :, lats_logic, :][:, :, :, lons_logic]
air_sub = air21[:, :, lats_logic, :][:, :, :, lons_logic]
lats_sub = lats[lats_logic]
lons_sub = lons[lons_logic]

dia_0 = dt.date(2022, 1, 11)
hgt_0 = hgt22[dt_time22 == dia_0, :, :, :][0][:, lats_logic, :][:, :, lons_logic]
air_0 = air22[dt_time22 == dia_0, :, :, :][0][:, lats_logic, :][:, :, lons_logic]


def dist_analogia(u, v):
    w = u - v
    a, b, c = w.shape
    total = 0
    for i, p in enumerate(level):
        wk = 0.5 if p == 500.0 or p == 1000.0 else 0
        if wk == 0:
            continue
        for j in range(b):
            for k in range(c):
                total += (w[i, j, k] ** 2) * wk
    return math.sqrt(total)


distancias = [
    (dist_analogia(hgt_0, hgt_sub[dt_time21 == d, :, :, :][0]), d) for d in dt_time21
]
distancias.sort()
top4 = [d for _, d in distancias[:4]]
print(top4)
hgt_1 = sum([hgt_sub[dt_time21==d,:,:,:][0] for d in top4 ])/4
air_1 = sum([air_sub[dt_time21==d,:,:,:][0] for d in top4 ])/4


fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))
fig.tight_layout()
plt.subplots_adjust(top=0.90)

ax = axs[0]
ax.set_title("Observación HGT", fontsize=15, ha="center")
plotted = hgt_0[level==500.,:,:]
cs = ax.contourf(lons_sub, lats_sub, plotted[0], cmap=CMAP, levels=LEV)
fig.colorbar(cs, ax=ax)

ax = axs[1]
ax.set_title("HGT-media", fontsize=15, ha="center")
plotted = hgt_1[level==500.,:,:]
cs = ax.contourf(lons_sub, lats_sub, plotted[0], cmap=CMAP, levels=LEV)
fig.colorbar(cs, ax=ax)

ax = axs[2]
ax.set_title("Observación AIR", fontsize=15, ha="center")
plotted = air_0[level==1000.,:,:]
cs = ax.contourf(lons_sub, lats_sub, plotted[0], cmap=CMAP, levels=LEV)
fig.colorbar(cs, ax=ax)

ax = axs[3]
ax.set_title("AIR-media", fontsize=15, ha="center")
plotted = air_1[level==1000.,:,:]
cs = plt.contourf(lons_sub, lats_sub, plotted[0], cmap=CMAP, levels=LEV)
fig.colorbar(cs, ax=ax)

plt.savefig("3")

print(np.array([abs(a) for a in air_1[level==1000.,:,:] - air_0[level==1000.,:,:]]))