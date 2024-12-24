import torch
import torch.nn.functional as F

from knn import normalize_features

def compute_loss(n, features_gcn, features_nodeformer, similar_cluster_intersection,
                 tau=0.5, device=3,batch_size=256):
    features_gcn = features_gcn.to(device)
    features_nodeformer = features_nodeformer.to(device)

    batch_size = batch_size
    loss = 0.0

    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)


        batch_features_gcn = features_gcn[start_idx:end_idx]
        batch_features_gcn= normalize_features(batch_features_gcn)

        batch_features_nodeformer = features_nodeformer[start_idx:end_idx]
        batch_features_nodeformer = normalize_features(batch_features_nodeformer)

        sim_gcn = torch.exp(F.cosine_similarity(batch_features_gcn.unsqueeze(1), features_gcn.unsqueeze(0), dim=2) / tau)
        sim_nodeformer = torch.exp(F.cosine_similarity(batch_features_nodeformer.unsqueeze(1), features_nodeformer.unsqueeze(0), dim=2) / tau)
        sim_gcn_nodeformer = torch.exp(F.cosine_similarity(batch_features_gcn.unsqueeze(1), features_nodeformer.unsqueeze(0), dim=2) / tau)
        sim_nodeformer_gcn = torch.exp(F.cosine_similarity(batch_features_nodeformer.unsqueeze(1), features_gcn.unsqueeze(0), dim=2) / tau)


        sim_gcn_sum = sim_gcn.sum(dim=1) - torch.diagonal(sim_gcn)
        sim_nodeformer_sum = sim_nodeformer.sum(dim=1) - torch.diagonal(sim_nodeformer)
        sim_gcn_nodeformer_sum = sim_gcn_nodeformer.sum(dim=1)
        sim_nodeformer_gcn_sum = sim_nodeformer_gcn.sum(dim=1)

        for i in range(start_idx, end_idx):
            similar_nodes = similar_cluster_intersection[i]
            if len(similar_nodes) != 0:
                loss += -torch.log(((sim_gcn_nodeformer[i - start_idx, i - start_idx] + sim_gcn[
                    i - start_idx, similar_nodes].sum() + sim_gcn_nodeformer[i - start_idx, similar_nodes].sum())) / (
                                               sim_gcn_sum[i - start_idx] + sim_gcn_nodeformer_sum[i - start_idx]))
                loss += -torch.log(((sim_nodeformer_gcn[i - start_idx, i - start_idx] + sim_nodeformer[
                    i - start_idx, similar_nodes].sum() + sim_nodeformer_gcn[i - start_idx, similar_nodes].sum())) / (
                                               sim_nodeformer_sum[i - start_idx] + sim_nodeformer_gcn_sum[
                                           i - start_idx]))

    return loss / (2*n)
