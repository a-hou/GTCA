import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, mlp_hidden_channels=128, dropout=0.5, save_mem=True):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=not save_mem)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, cached=not save_mem)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=not save_mem)
        self.dropout = dropout
        self.residual = nn.Linear(in_channels, out_channels)
        # MLP部分
        self.MLP = nn.ModuleList()
        self.MLP.append(nn.Linear(hidden_channels, hidden_channels))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        #x = self.MLP[-1](x)
        return x



