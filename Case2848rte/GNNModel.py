import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, InstanceNorm, LayerNorm

class GNNModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=1):
        super(GNNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.conv1 = GCNConv(self.input_dim, self.hidden_dim)
        self.bn1 = BatchNorm(self.hidden_dim)
        self.in1 = InstanceNorm(self.hidden_dim)
        self.ln1 = LayerNorm(self.hidden_dim)

        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.bn2 = BatchNorm(self.hidden_dim)
        self.in2 = InstanceNorm(self.hidden_dim)
        self.ln2 = LayerNorm(self.hidden_dim)

        self.conv3 = GCNConv(self.hidden_dim, self.output_dim)
        self.bn3 = BatchNorm(self.output_dim)
        self.in3 = InstanceNorm(self.output_dim)
        self.ln3 = LayerNorm(self.output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        # x = self.bn1(x)
        # x = self.in1(x)
        x = self.ln1(x)

        x = F.relu(self.conv2(x, edge_index))
        # x = self.bn2(x)
        # x = self.in2(x)
        x = self.ln2(x)

        x = F.relu(self.conv3(x, edge_index))
        # x = self.bn3(x)
        # x = self.in3(x)
        x = self.ln3(x)

        return x.flatten()
