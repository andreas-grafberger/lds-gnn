import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_
from torchmeta.modules import MetaModule, MetaLinear
from torchmeta.modules.utils import get_subdict


class DenseGraphConvolution(nn.Module):
    """
    Simple Graph Convolutional Layer working on dense adjacency matrices.
    Note that this is very inefficient and whenever sparsity can be exploited,
    use the version from pytorch geometric.
    """

    def __init__(self, in_features, out_features, use_bias=True):
        super(DenseGraphConvolution, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=use_bias)
        self.reset_weights()

    def reset_weights(self):
        self.fc.weight = Parameter(xavier_uniform_(self.fc.weight.clone()))
        self.fc.bias = Parameter(self.fc.bias.clone().zero_())

    def forward(self, node_features, dense_adj):
        embeddings = self.fc(node_features)
        return torch.mm(dense_adj, embeddings)


class MetaDenseGraphConvolution(MetaModule):
    __doc__ = DenseGraphConvolution.__doc__

    def __init__(self, in_features, out_features, use_bias=True):
        super(MetaDenseGraphConvolution, self).__init__()
        self.fc = MetaLinear(in_features, out_features, bias=use_bias)
        self.reset_weights()

    def reset_weights(self):
        self.fc.weight = Parameter(xavier_uniform_(self.fc.weight.clone()))
        self.fc.bias = Parameter(self.fc.bias.clone().zero_())

    def forward(self, node_features, dense_adj, params=None):
        embeddings = self.fc.forward(node_features, params=get_subdict(params, "fc"))
        return torch.mm(dense_adj, embeddings)
