from faker import Factory
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, fowlkes_mallows_score, adjusted_rand_score
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time

from Point import Point
from Cluster import Cluster
from DBSCAN import DBSCAN


def unsupervised_validation(cluster):
    X = []
    labels = []
    for index in range(1, len(cluster)):
        for point in cluster[index].points:
            X.append(point)
            labels.append(index)
    silhouette_acc = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    print("the silhouette score is: {}".format(silhouette_acc))
    print("the davis-bouldin score is: {}".format(davies_bouldin))


def supervised_validation(points, true_labels):
    pred = []
    for point in points:
        pred.append(point.cluster)
    fowlkes = fowlkes_mallows_score(labels_true=true_labels, labels_pred=pred)
    random = adjusted_rand_score(labels_true=true_labels, labels_pred=pred)
    print("the fowlkes-mallows score is: {}".format(fowlkes))
    print("the random adjust score is: {}".format(random))


def varInit(path, data, variables):
    index = -1
    for dataset in data:
        index += 1
        if dataset in path:
            ep, minPts = variables[index]
            return ep, minPts
    return -1, -1


def original_plotting(dfns, dim):
    if dim == 2:
        x1 = dfns[:, 0]
        y1 = dfns[:, 1]
        fig, a1 = plt.subplots(1, 2)
        a1[0].plot(x1, y1, 'ob')
        return fig, a1
    elif dim == 3:
        fig = plt.figure()
        a1 = fig.add_subplot(1, 2, 1, projection='3d')
        x1 = dfns[:, 0]
        y1 = dfns[:, 1]
        z1 = dfns[:, 2]
        a1.scatter(x1, y1, z1, color='b', marker='o')
        a1.set_zlim(-2.5, 2.5)
        a1.set_ylim(-2.5, 2.5)
        a1.set_xlim(-2.5, 2.5)
        return fig, a1
    # the dimension is too high for visualization, we'll use PCA to reduce for 3D #
    else:
        lower_features = PCA(n_components=3)
        l_f = lower_features.fit_transform(dfns)
        fig = plt.figure()
        a1 = fig.add_subplot(1, 2, 1, projection='3d')
        x1 = l_f[:, 0]
        y1 = l_f[:, 1]
        z1 = l_f[:, 2]
        a1.scatter(x1, y1, z1, color='b', marker='o')
        a1.set_zlim(-2.5, 2.5)
        a1.set_ylim(-2.5, 2.5)
        a1.set_xlim(-2.5, 2.5)
        return fig, a1


def result_plotting(clusters, dim, fig, a1):
    if dim == 2:
        for clust in clusters:
            x = []
            y = []
            for pnt in clust.points:
                x.append(pnt[0])
                y.append(pnt[1])
            a1[1].plot(x, y, 'o', color=clust.color)
    elif dim == 3:
        a2 = fig.add_subplot(1, 2, 2, projection='3d')
        for clust in clusters:
            x = []
            y = []
            z = []
            for pnt in clust.points:
                x.append(pnt[0])
                y.append(pnt[1])
                z.append(pnt[2])
            a2.scatter(x, y, z, marker='o', color=clust.color)
        a2.set_zlim(-2.5, 2.5)
        a2.set_ylim(-2.5, 2.5)
        a2.set_xlim(-2.5, 2.5)
        # the dimension is too high for visualization, we'll use PCA to reduce for 3D #
    else:
        a2 = fig.add_subplot(1, 2, 2, projection='3d')
        group_clusters = []
        len_group = []
        for clust in clusters:
            len_group.append(len(clust.points))
            for point in clust.points:
                group_clusters.append(point)
        group_clusters = np.asarray(group_clusters)
        lower_features = PCA(n_components=3)
        l_f = lower_features.fit_transform(group_clusters)
        index = 0
        for ind in range(len(clusters)):
            x = []
            y = []
            z = []
            for j in range(len_group[ind]):
                x.append(l_f[j + index][0])
                y.append((l_f[j + index][1]))
                z.append(l_f[j + index][2])
            index += len_group[ind]
            a2.scatter(x, y, z, marker='o', color=clusters[ind].color)
        a2.set_ylim(-2.5, 2.5)
        a2.set_xlim(-2.5, 2.5)
        a2.set_zlim(-2.5, 2.5)


dataNames = ['aggregation', 'circles', 'flame', 'iris', 'jane', 'moons', 'wine', 'annual_balance_sheets', 'R15',
             'vehicle']
variables = [[0.14, 6], [0.2, 4], [0.29, 6], [0.61, 6], [0.3, 4], [0.15, 4], [2.2, 5], [7.8, 11], [0.105, 4], [1.56, 6]]

# loading the data #
path = "aggregation.csv"
dataset_ = pd.read_csv(path, delimiter=' ', engine='python')
dataset = np.array(dataset_)

# initializing the variables #
color_gen = Factory.create()
eps, minPnt = varInit(path, dataNames, variables)

if (eps, minPnt) == (-1, -1):
    print("error: the dataset is not familiar to the system ")
    exit(-1)

# normalize the data #
dataset = StandardScaler().fit_transform(X=dataset)

# plotting the original data
(fig, ax1) = original_plotting(dataset, dataset.shape[1])

# creating collection of all the points #
points = []
for row in dataset:
    points.append(Point(row))

# running the DBSCAN algorithm #
t1 = time.time()
cluster = [Cluster('#000000')]
visited = 0
while visited < len(points):
    visited = DBSCAN(points, eps, minPnt, cluster, color_gen, visited)

t2 = time.time()
print("runing time: {} sec".format((t2 - t1)))

# printing the results:
total_len = len(points)
print("\nnumber of cluster counts: {}".format(len(cluster) - 1))
for i in range(1, len(cluster)):
    print(
        "cluster number {}: {}% of the points".format(i, round(((len(cluster[i].points)) / total_len) * 100), ndigit=2))
print("noise cluster have {}% of the points".format(round(((len(cluster[0].points)) / total_len) * 100), ndigit=2))

# plotting the results according to the cluster array
result_plotting(cluster, dataset.shape[1], fig, ax1)

unsupervised_validation(cluster)
plt.title("after DBSCAN")
plt.show()
