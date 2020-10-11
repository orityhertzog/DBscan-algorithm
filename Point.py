import numpy as np


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# declare Point object that will
# contain the point labels,features and cluster
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

class Point:

    def __init__(self, features_vector):
        self.label = "UNLABELED"
        self.features = features_vector
        self.cluster = -1

    def distance(self, pnt2):
        return np.linalg.norm(self.features - pnt2.features)
