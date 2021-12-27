import torch.nn.functional as F
from torchmeta.modules import MetaModule
from torchmeta.modules.utils import get_subdict

from src.models.layers import MetaDenseGraphConvolution
from src.utils.graph import normalize_adjacency_matrix


class MetaDenseGCN(MetaModule):

    def __init__(self, in_features, hidden_features, out_features, dropout, normalize_adj: bool = True):
        super(MetaDenseGCN, self).__init__()
        self.layer_in = MetaDenseGraphConvolution(in_features, hidden_features)
        self.layer_out = MetaDenseGraphConvolution(hidden_features, out_features)

        self.dropout = dropout
        self.normalize_adj = normalize_adj

    def reset_weights(self):
        self.layer_in.reset_weights()
        self.layer_out.reset_weights()

    def forward_to_last_layer(self, node_features, dense_adj, params=None):
        if self.normalize_adj:
            dense_adj = normalize_adjacency_matrix(dense_adj)

        embeddings = F.dropout(node_features, self.dropout, training=self.training)
        embeddings = F.relu(self.layer_in(embeddings, dense_adj, params=get_subdict(params, 'layer_in')))
        embeddings = F.dropout(embeddings, self.dropout, training=self.training)
        return self.layer_out(embeddings, dense_adj, params=get_subdict(params, 'layer_out'))

    def forward(self, node_features, dense_adj, params=None):
        embeddings = self.forward_to_last_layer(node_features, dense_adj, params=params)
        return F.log_softmax(embeddings, dim=1)
