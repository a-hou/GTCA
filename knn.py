import contextlib
import io

import numpy as np
import torch
import torch.nn.functional as F

def cosine_similarity_torch(x):
    x = F.normalize(x, p=2, dim=1)
    return torch.mm(x, x.t())

# 获取k个最相似和最不相似的节点
def get_top_k_similar_and_dissimilar(similarity_matrix, k):
    most_similar = torch.topk(similarity_matrix, k=k+1, dim=1).indices[:, 1:].cpu().numpy()
    #least_similar = torch.topk(similarity_matrix, k=k, dim=1, largest=False).indices.cpu().numpy()
    return most_similar

# 获取拓扑距离最近和最远的k个节点
def get_top_k_closest_and_furthest(distance_matrix, k):
    closest_nodes = np.argsort(distance_matrix, axis=1)[:, :k]
    furthest_nodes = np.argsort(distance_matrix, axis=1)[:, -k:]
    return closest_nodes, furthest_nodes

# 构建正样本和负样本集合
#def get_sample_sets(n, most_similar,least_similar, closest,furthest):
def get_sample_sets(n, most_similar, closest):
    similar_set = set(most_similar)
    #dissimilar_set = set(least_similar)
    closest_set = set(closest)
    #furthest_set = set(furthest)

    positive_set = similar_set.intersection(closest_set)
    #negative_set = dissimilar_set.intersection(furthest_set)
    #negative_set = dissimilar_set.union(furthest_set)
    negative_set = set(range(n)) - positive_set
    return list(positive_set), list(negative_set)


def normalize_features(features):
    norms = torch.norm(features, p=2, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-8)  # 避免除以零
    return features / norms
