"""
Práctica 3
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import make_blobs
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

Figure = 1

""" Apartado i) """

# #############################################################################
# Aquí tenemos definido el sistema X de 1000 elementos de dos estados
# construido a partir de una muestra aleatoria entorno a unos centros:
centers = [[-0.5, 0.5], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=1000, centers=centers, cluster_std=0.4, random_state=0
)

# #############################################################################
# Los clasificamos mediante el algoritmo KMeans
def plot_silhouette(param, xs, alg):
    global Figure
    ss = []
    for x in xs:
        m = alg(x).fit(X)
        silhouette = metrics.silhouette_score(X, m.labels_)
        k = len(set(m.labels_)) - (1 if -1 in m.labels_ else 0)
        print(param,"= %0.3f" % x, "\tk =", k, "\ts̄ = %0.3f" % silhouette)
        ss += [silhouette]
    plt.xlabel(param)
    plt.ylabel("Silhouette Coefficient s̄")
    plt.plot(xs, ss)
    plt.savefig("Figure_%d" % Figure)
    Figure += 1
    plt.show()


plot_silhouette("k",range(2, 16), lambda x: KMeans(n_clusters=x, random_state=0))
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print("\nCentros:\n", kmeans.cluster_centers_)

print("-" * 10)

# #############################################################################
# Representamos el resultado con un plot


def plot_points(alg, core_samples_mask=None):
    if type(core_samples_mask) is not np.ndarray:
        core_samples_mask = np.zeros_like(alg.labels_, dtype=bool)
        core_samples_mask[:] = True

    plt.xlim([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
    plt.ylim([X[:, 1].min() - 0.5, X[:, 1].max() + 0.5])

    unique_labels = set(alg.labels_)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = alg.labels_ == k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=5,
        )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=3,
        )
    plt.xlabel("x")
    plt.ylabel("y")


vor_regions = Voronoi(kmeans.cluster_centers_)
voronoi_plot_2d(vor_regions, show_points=False, show_vertices=False)

plot_points(kmeans)
plt.plot(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], "ro", markersize=8
)
plt.title("KMEANS algorithm\nNumber of clusters: %d" % len(kmeans.cluster_centers_))
plt.savefig("Figure_%d" % Figure)
Figure += 1
plt.show()


""" Apartado ii) """

# #############################################################################
# Los clasificamos mediante el algoritmo DBSCAN

plot_silhouette("eps",
    np.arange(0.1, 0.4, 0.02),
    lambda x: DBSCAN(eps=x, min_samples=10, metric="euclidean"),
)
db = DBSCAN(eps=0.28, min_samples=10, metric="euclidean").fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
plot_points(db, core_samples_mask)

n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
plt.title("DBSCAN+euclidean algorithm\nNumber of clusters: %d" % n_clusters_)
plt.savefig("Figure_%d" % Figure)
Figure += 1
plt.show()

print("-" * 10)

plot_silhouette("eps",
    np.arange(0.1, 0.4, 0.02),
    lambda x: DBSCAN(eps=x, min_samples=10, metric="manhattan"),
)
db = DBSCAN(eps=0.36, min_samples=10, metric="manhattan").fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
plot_points(db, core_samples_mask)

n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
plt.title("DBSCAN+manhattan algorithm\nNumber of clusters: %d" % n_clusters_)
plt.savefig("Figure_%d" % Figure)
Figure += 1
plt.show()

print("-" * 10)


""" Apartado iii) """

# #############################################################################
# Predicción de elementos para pertenecer a una clase:
problem = np.array([[0, 0], [0, -1]])
clases_pred = kmeans.predict(problem)
print(clases_pred)

vor_regions = Voronoi(kmeans.cluster_centers_)
voronoi_plot_2d(vor_regions, show_points=False, show_vertices=False)

plot_points(kmeans)
plt.plot(problem[:, 0], problem[:, 1], "ro", markersize=8)

plt.title("Class Predictions")
plt.savefig("Figure_%d" % Figure)
Figure += 1
plt.show()
