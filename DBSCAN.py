import random
from Cluster import Cluster

cluster_no = 0


# checking amount of eps-neighbored for a point #
def exploreEpsNeighborhood(points, rand_point, eps):
    eps_neighborhood = []
    for point in points:
        dist = rand_point.distance(point)
        if dist <= eps:
            eps_neighborhood.append(point)
    return eps_neighborhood


# labeling each point by the results of eps_neighborhood, and assign to a cluster #
def labeling(eps_neighborhood, minPnt, rand_point, cluster, color_gen, points_to_explore, visited):
    global cluster_no
    if len(eps_neighborhood) >= minPnt:
        rand_point.label = "core"
        if rand_point.cluster == -1:
            cluster_no = cluster_no + 1
            rand_point.cluster = cluster_no
            rand_color = color_gen.hex_color()
            while rand_color == '#ffffff' or rand_color == '#000000':
                rand_color = color_gen.hex_color()
            cluster.append(Cluster(rand_color))
            cluster[rand_point.cluster].insertNewPoint(rand_point)
        for pnt in eps_neighborhood:
            points_to_explore.append(pnt)
            if pnt.cluster == -1:
                pnt.cluster = cluster_no
                cluster[pnt.cluster].insertNewPoint(pnt)
            elif pnt.label == 'border':
                if pnt.cluster == 0:
                    pnt.cluster = cluster_no
                    cluster[pnt.cluster].insertNewPoint(pnt)
                    cluster[0].deleteBorderPoint(pnt)

    elif len(eps_neighborhood) == 0 or rand_point.cluster == -1:
        rand_point.label = "noise"
        rand_point.cluster = 0
        cluster[0].insertNewPoint(rand_point)
    else:
        rand_point.label = "border"

    visited += 1
    return visited


def DBSCAN(points, eps, minPnt, cluster, color_gen, visited):
    rnd = random.randint(0, len(points) - 1)
    rand_point = points[rnd]
    points_to_explore = []
    if rand_point.label == "UNLABELED":
        eps_neighborhood = exploreEpsNeighborhood(points, rand_point, eps)
        visited = labeling(eps_neighborhood, minPnt, rand_point, cluster, color_gen, points_to_explore, visited)
        while len(points_to_explore) > 0:
            eps_neighborhood = []
            exp_pnt = points_to_explore[0]
            if exp_pnt.label == "UNLABELED":
                eps_neighborhood = exploreEpsNeighborhood(points, exp_pnt, eps)
                visited = labeling(eps_neighborhood, minPnt, exp_pnt, cluster, color_gen, points_to_explore, visited)
            del points_to_explore[0]
    return visited
