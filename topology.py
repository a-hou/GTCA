import networkx as nx
import numpy as np

from knn import get_top_k_closest_and_furthest
from utils import load_network_data

adj1, features, Y = load_network_data('cora')
G = nx.from_scipy_sparse_matrix(adj1)
shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))
num_nodes = features.shape[0]


distance_matrix = np.zeros((num_nodes, num_nodes))
for node, lengths in shortest_path_lengths.items():
    for target, distance in lengths.items():
        distance_matrix[node, target] = distance

np.save('distance_matrix_cora.npy', distance_matrix)


