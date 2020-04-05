# Import Libraries
from sklearn.neighbors import NearestNeighbors
import numpy as np
import mxnet as mx

def turn_NearestNeighbor(features):
    X = np.array(features)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    print(indices)
    return

