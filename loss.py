import torch
import torch.nn.functional as F

from knn import normalize_features


# def compute_loss(n, features_gcn, features_nodeformer, similar_cluster_intersection,
#                  tau=1.0, device=3,batch_size=256):
#     features_gcn = features_gcn.to(device)
#     features_nodeformer = features_nodeformer.to(device)
#
#     batch_size = batch_size # 可以根据需要调整
#     loss = 0.0
#
#     for start_idx in range(0, n, batch_size):
#         end_idx = min(start_idx + batch_size, n)
#
#         # 计算部分相似度矩阵
#         batch_features_gcn = features_gcn[start_idx:end_idx]
#         batch_features_nodeformer = features_nodeformer[start_idx:end_idx]
#
#         # 计算余弦相似度矩阵
#         sim_gcn = torch.exp(F.cosine_similarity(batch_features_gcn.unsqueeze(1), features_gcn.unsqueeze(0), dim=2) / tau)
#         sim_nodeformer = torch.exp(F.cosine_similarity(batch_features_nodeformer.unsqueeze(1), features_nodeformer.unsqueeze(0), dim=2) / tau)
#         similarity_gcn_sum = sim_gcn.sum(dim=1) - torch.diagonal(sim_gcn)
#         similarity_nodeformer_sum = sim_nodeformer.sum(dim=1) - torch.diagonal(sim_nodeformer)
#
#         for i in range(start_idx, end_idx):
#             similar_nodes = similar_cluster_intersection[i]
#             if len(similar_nodes) != 0:
#                 I1 = sim_gcn[i - start_idx, similar_nodes].sum()
#                 I3 = sim_nodeformer[i - start_idx, similar_nodes].sum()
#             else:
#                 # I1 = 0.0
#                 # I3 = 0.0
#                 I1 = similarity_gcn_sum[i - start_idx]
#                 I3 = similarity_nodeformer_sum[i - start_idx]
#             I2 = similarity_gcn_sum[i - start_idx] - I1
#             I4 = similarity_nodeformer_sum[i - start_idx] - I3
#
#             # 定义综合损失函数
#             # 这里使用了L2损失来衡量 GCN 和 Transformer 特征相似性的差异
#             #loss += torch.norm((I1 - I2) - (I3 - I4))  # L2损失
#             #loss += -(I1 -I2 + I3 - I4)
#             loss += -torch.log((I1+I3)/(I1+I2+I3+I4))
#         #torch.cuda.empty_cache()
#
#     return loss / batch_size

#InfoNCE损失
# def compute_loss(n, features_gcn, features_nodeformer, similar_cluster_intersection,
#                  tau=0.5, device=3,batch_size=256):
#     features_gcn = features_gcn.to(device)
#     features_nodeformer = features_nodeformer.to(device)
#
#     batch_size = batch_size
#     loss = 0.0
#
#     for start_idx in range(0, n, batch_size):
#         end_idx = min(start_idx + batch_size, n)
#
#         # 取出batch_size大小的特征矩阵，并归一化
#         batch_features_gcn = features_gcn[start_idx:end_idx]
#         batch_features_gcn= normalize_features(batch_features_gcn)
#
#         batch_features_nodeformer = features_nodeformer[start_idx:end_idx]
#         batch_features_nodeformer = normalize_features(batch_features_nodeformer)
#
#         # 计算四个余弦相似度矩阵,维度为（batch_size * n）
#         sim_gcn = torch.exp(F.cosine_similarity(batch_features_gcn.unsqueeze(1), features_gcn.unsqueeze(0), dim=2) / tau)
#         sim_nodeformer = torch.exp(F.cosine_similarity(batch_features_nodeformer.unsqueeze(1), features_nodeformer.unsqueeze(0), dim=2) / tau)
#         sim_gcn_nodeformer = torch.exp(F.cosine_similarity(batch_features_gcn.unsqueeze(1), features_nodeformer.unsqueeze(0), dim=2) / tau)
#         sim_nodeformer_gcn = torch.exp(F.cosine_similarity(batch_features_nodeformer.unsqueeze(1), features_gcn.unsqueeze(0), dim=2) / tau)
#
#
#         #求出每个节点与其他所有节点的e^sim(vi,vj)之和
#         sim_gcn_sum = sim_gcn.sum(dim=1) - torch.diagonal(sim_gcn)
#         sim_nodeformer_sum = sim_nodeformer.sum(dim=1) - torch.diagonal(sim_nodeformer)
#         sim_gcn_nodeformer_sum = sim_gcn_nodeformer.sum(dim=1)
#         sim_nodeformer_gcn_sum = sim_nodeformer_gcn.sum(dim=1)
#
#         for i in range(start_idx, end_idx):
#             similar_nodes = similar_cluster_intersection[i]
#             if len(similar_nodes) != 0:
#                 for j in similar_nodes:
#                     loss += -torch.log(sim_gcn_nodeformer[i - start_idx,j] /
#                                        (sim_gcn_sum[i - start_idx] + sim_gcn_nodeformer_sum[i - start_idx]))
#                     loss += -torch.log(sim_nodeformer_gcn[i - start_idx,j]
#                                        / (sim_nodeformer_sum[i - start_idx] + sim_nodeformer_gcn_sum[i - start_idx]))
#
#     return loss / (2*n)

#NCLA损失
def compute_loss(n, features_gcn, features_nodeformer, similar_cluster_intersection,
                 tau=0.5, device=3,batch_size=256):
    features_gcn = features_gcn.to(device)
    features_nodeformer = features_nodeformer.to(device)

    batch_size = batch_size
    loss = 0.0

    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)

        # 取出batch_size大小的特征矩阵，并归一化
        batch_features_gcn = features_gcn[start_idx:end_idx]
        batch_features_gcn= normalize_features(batch_features_gcn)

        batch_features_nodeformer = features_nodeformer[start_idx:end_idx]
        batch_features_nodeformer = normalize_features(batch_features_nodeformer)

        # 计算四个余弦相似度矩阵,维度为（batch_size * n）
        sim_gcn = torch.exp(F.cosine_similarity(batch_features_gcn.unsqueeze(1), features_gcn.unsqueeze(0), dim=2) / tau)
        sim_nodeformer = torch.exp(F.cosine_similarity(batch_features_nodeformer.unsqueeze(1), features_nodeformer.unsqueeze(0), dim=2) / tau)
        sim_gcn_nodeformer = torch.exp(F.cosine_similarity(batch_features_gcn.unsqueeze(1), features_nodeformer.unsqueeze(0), dim=2) / tau)
        sim_nodeformer_gcn = torch.exp(F.cosine_similarity(batch_features_nodeformer.unsqueeze(1), features_gcn.unsqueeze(0), dim=2) / tau)


        #求出每个节点与其他所有节点的e^sim(vi,vj)之和，并减去对角
        sim_gcn_sum = sim_gcn.sum(dim=1) - torch.diagonal(sim_gcn)
        sim_nodeformer_sum = sim_nodeformer.sum(dim=1) - torch.diagonal(sim_nodeformer)
        sim_gcn_nodeformer_sum = sim_gcn_nodeformer.sum(dim=1)
        sim_nodeformer_gcn_sum = sim_nodeformer_gcn.sum(dim=1)

        for i in range(start_idx, end_idx):
            similar_nodes = similar_cluster_intersection[i]
            if len(similar_nodes) != 0:
                # loss += -torch.log(((sim_gcn_nodeformer[i - start_idx,i - start_idx] + sim_gcn[i - start_idx,similar_nodes].sum() + sim_gcn_nodeformer[i - start_idx,similar_nodes].sum()) / (1 + 2*len(similar_nodes))) / (sim_gcn_sum[i - start_idx] + sim_gcn_nodeformer_sum[i - start_idx]))
                # loss += -torch.log(((sim_nodeformer_gcn[i - start_idx,i - start_idx] + sim_nodeformer[i - start_idx,similar_nodes].sum() + sim_nodeformer_gcn[i - start_idx,similar_nodes].sum()) / (1 + 2*len(similar_nodes))) / (sim_nodeformer_sum[i - start_idx] + sim_nodeformer_gcn_sum[i - start_idx]))
                loss += -torch.log(((sim_gcn_nodeformer[i - start_idx, i - start_idx] + sim_gcn[
                    i - start_idx, similar_nodes].sum() + sim_gcn_nodeformer[i - start_idx, similar_nodes].sum())) / (
                                               sim_gcn_sum[i - start_idx] + sim_gcn_nodeformer_sum[i - start_idx]))
                loss += -torch.log(((sim_nodeformer_gcn[i - start_idx, i - start_idx] + sim_nodeformer[
                    i - start_idx, similar_nodes].sum() + sim_nodeformer_gcn[i - start_idx, similar_nodes].sum())) / (
                                               sim_nodeformer_sum[i - start_idx] + sim_nodeformer_gcn_sum[
                                           i - start_idx]))

    return loss / (2*n)
