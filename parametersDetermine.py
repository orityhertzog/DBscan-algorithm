import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


def elbowMethod(data, k, mean_k_distance):
    for i in range(len(data)):
        tmp_dist = []
        for j in range(len(data)):
            tmp_dist.append(np.linalg.norm(data[i] - data[j]))
        tmp_dist.sort()
        k_mean_dist = 0
        for z in range(k):
            k_mean_dist += tmp_dist[z]
        mean_k_distance.append(k_mean_dist / k)
    mean_k_distance.sort()
    index = list(range(len(mean_k_distance)))

    plt.plot(index, mean_k_distance)


path = "data/vehicle.csv"
k = 6
dataset_ = pd.read_csv(path, delimiter=' ', engine='python')
dataset = np.array(dataset_)
dataset = StandardScaler().fit_transform(X=dataset)
mean_k_distance = []


elbowMethod(dataset, k, mean_k_distance)

plt.show()
