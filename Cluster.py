import numpy as np


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# create cluster object that contains
# individual color and the points belong to the cluster
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class Cluster:
    def __init__(self, color):
        self.color = color
        self.points = []

    def insertNewPoint(self, point):
        self.points.append(point.features)

    def deleteBorderPoint(self, point):
        ind = 0
        length = len(self.points)
        while ind < length and not np.array_equal(point.features, self.points[ind]):
            ind += 1
        if ind != length:
            del self.points[ind]
