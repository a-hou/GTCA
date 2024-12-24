import contextlib
import io

import numpy as np
import torch
import torch.nn.functional as F

def cosine_similarity_torch(x):
    x = F.normalize(x, p=2, dim=1)
    return torch.mm(x, x.t())

def get_top_k_similar_and_dissimilar(similarity_matrix, k):
    most_similar = torch.topk(similarity_matrix, k=k+1, dim=1).indices[:, 1:].cpu().numpy()
    return most_similar

def get_top_k_closest_and_furthest(distance_matrix, k):
    closest_nodes = np.argsort(distance_matrix, axis=1)[:, :k]
    furthest_nodes = np.argsort(distance_matrix, axis=1)[:, -k:]
    return closest_nodes, furthest_nodes

def get_sample_sets(n, most_similar, closest):
    similar_set = set(most_similar)
    closest_set = set(closest)


    positive_set = similar_set.intersection(closest_set)
    negative_set = set(range(n)) - positive_set
    return list(positive_set), list(negative_set)


def normalize_features(features):
    norms = torch.norm(features, p=2, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-8)  # 避免除以零
    return features / norms
