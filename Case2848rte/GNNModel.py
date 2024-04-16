import torch.nn as nn
from torch.functional import F
from torch_geometric.nn import GCNConv, BatchNorm, InstanceNorm, LayerNorm


class GNNModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=1):
        super(GNNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.conv1 = GCNConv(self.input_dim, self.hidden_dim)
        self.batch_norm1 = BatchNorm(self.hidden_dim)
        self.instance_norm1 = InstanceNorm(self.hidden_dim)
        self.layer_norm1 = LayerNorm(self.hidden_dim)

        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.batch_norm2 = BatchNorm(self.hidden_dim)
        self.instance_norm2 = InstanceNorm(self.hidden_dim)
        self.layer_norm2 = LayerNorm(self.hidden_dim)

        self.conv3 = GCNConv(self.hidden_dim, self.output_dim)
        self.batch_norm3 = BatchNorm(self.output_dim)
        self.instance_norm3 = InstanceNorm(self.output_dim)
        self.layer_norm3 = LayerNorm(self.output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = self.instance_norm1(x)
        x = self.layer_norm1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = self.instance_norm2(x)
        x = self.layer_norm2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = self.batch_norm3(x)
        x = self.instance_norm3(x)
        x = self.layer_norm3(x)

        return x.flatten()