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
def plot_silhouette(xs,alg):
    ss = []
    for x in xs:
        m = alg(x).fit(X)
        silhouette = metrics.silhouette_score(X, m.labels_)
        k = len(set(m.labels_)) - (1 if -1 in m.labels_ else 0)
        print("Param. = %0.3f"% x, "\tk =",k,"\ts̄ = %0.3f"% silhouette)
        ss += [silhouette]
    plt.plot(xs, ss)
    plt.show()

plot_silhouette(range(2,4),lambda x : KMeans(n_clusters=x, random_state=0))

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print("Centros: ",kmeans.cluster_centers_)

print("-"*10)

# #############################################################################
# Representamos el resultado con un plot

def plot_kmeans():
    vor_regions = Voronoi(kmeans.cluster_centers_)
    voronoi_plot_2d(vor_regions, show_points=False, show_vertices=False)

    plt.xlim([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
    plt.ylim([X[:, 1].min() - 0.5, X[:, 1].max() + 0.5])

    unique_labels = set(kmeans.labels_)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = kmeans.labels_ == k
        xy = X[class_member_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=5,
        )

    plt.plot(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1],'ro', markersize=8)

    plt.title("Fixed number of KMeans clusters: %d" % 3)

plot_kmeans()
plt.show()

""" Apartado ii) """
 
# #############################################################################
# Los clasificamos mediante el algoritmo DBSCAN

plot_silhouette(np.arange(0.1,0.4,0.05),lambda x: DBSCAN(eps=x, min_samples=10, metric='euclidean'))
print("-"*10)
plot_silhouette(np.arange(0.1,0.4,0.05),lambda x: DBSCAN(eps=x, min_samples=10, metric='manhattan'))

""" Apartado iii) """


# #############################################################################
# Predicción de elementos para pertenecer a una clase:
problem = np.array([[0, 0], [0, -1]])
clases_pred = kmeans.predict(problem)
print(clases_pred)

plot_kmeans()
plt.plot(problem[:,0],problem[:,1],'ro', markersize=2)
plt.show()
