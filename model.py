from torch import nn
from gcn import GCN
from nodeformer import NodeFormer


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.model_gcn = GCN(input_size, hidden_size, output_size)
        self.model_nodeformer = NodeFormer(input_size, hidden_size, output_size)

    def forward(self, features,adj,adjs):
        x_gcn = self.model_gcn(features, adj)
        output_nodeformer = self.model_nodeformer(features, adjs)
        return x_gcn , output_nodeformer

