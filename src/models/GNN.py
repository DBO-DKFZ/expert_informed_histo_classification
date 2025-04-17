from typing import Optional, Type

import torch
import torch.nn as nn
import torch_geometric as geom
import torch_geometric.nn as geom_nn
from omegaconf import DictConfig, ListConfig
from torch_geometric.utils import unbatch


class GNN(nn.Module):
    def __init__(self, gnn: DictConfig[str, str | Type[geom_nn.MessagePassing]], pool: ListConfig[DictConfig[str, str | Type[geom_nn.aggr.Aggregation]]], head: Type[nn.Module]) -> None:
        """
        Initializes a GNN module. Builds the model as follows: GNN -> Pool1 -> Pool2 -> ... -> PoolN -> Head, and wraps
        it into a torch_geometric.Sequential module.
        For more info about in-out, and possible layers, refer to: https://pytorch-geometric.readthedocs.io

        :param gnn: A DictConfig containing the message passing module and its in-out configuration.
        :param pool: A ListConfig containing a list of DictConfigs containing the pooling/aggregation modules and their
                     in-out configuration.
        :param head: The classification head.
        """
        super().__init__()

        modules = [(gnn.module, gnn.in_out)] if 'module' in gnn else []
        modules += [(pool[x].module, pool[x].in_out) for x in pool]
        modules += [head]

        self.model = geom_nn.Sequential('x, edge_index, edge_weight, edge_attr, batch, pos',
                                        modules)

    def forward(self, x: geom.data.Data):
        """
        Computes the forward pass of the model.

        :param x: the input data (a torch_geometric data object).
        :return: the output of the model.
        """
        return self.model(x.x, edge_index=x.edge_index, edge_weight=x.edge_weight, edge_attr=x.edge_attr, batch=x.batch, pos=x.pos)


class TransformerAggregation(geom_nn.aggr.Aggregation):
    """
    A transformer aggregation layer, building upon torch_geometric's base aggregation class. Aggregates all input
    features into a single classification token, similar to a vision transformer encoder. Makes use of torch's
    nn.TransformerEncoder, and sinusoidal 2D positional encoding for positional encoding.
    """
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6, dropout: float = 0., pos_enc: Optional[bool] = None, granularity: float = 1.):
        """
        Initializes the transformer aggregation layer.

        :param d_model: dimensionality of the features.
        :param nhead: number of classification head for multi-head attention.
        :param num_layers: number of encoder layers.
        :param dropout: the dropout used in the transformer encoder.
        :param pos_enc: whether to use 2D sinusoidal positional encoding.
        :param granularity: the granularity of the positional encoding (i.e. scales the positional encoding values).
        """
        super().__init__()
        self.pos_enc = PositionalEncoding2D(d_model, granularity=granularity) if pos_enc else None
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout), num_layers=num_layers)
        self.cls = nn.Parameter(torch.rand((1,d_model)))

    def forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None, dim: int = -2, max_num_elements: Optional[int] = None) -> torch.Tensor:
        """
        Computes the forward pass.

        :param x: input tensor.
        :param index: batch vector, assigning each node to a specific example.
        :param ptr: the position (the sequential module resolves the arguments positional, which is why the ptr argument
                    is used).
        :param dim_size: The size of the output tensor at dimension `dim` after aggregation. Passed to parent.
        :param dim: The dimension in which to aggregate. Passed to parent.
        :param max_num_elements: The maximum number of elements within a single aggregation group. Passed to parent.
        :return: A tensor containing an aggregated, single feature vector for each sample.
        """
        if index is None:
            index = torch.tensor([0 for _ in range(len(x))])

        if self.pos_enc:
            x = self.pos_enc(x, batch_index=index, pos=ptr)
        x = unbatch(x, index, dim)
        out = []
        for sample in x:
            out.append(self.encoder(torch.cat((sample, self.cls)))[-1])

        return torch.stack(out)


def reduce_position(position: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Convenience function to adjust the positions when using pooling.

    :param position: the position matrix
    :param index: which indices to use
    :return: the reduced position matrix
    """
    return position[index]


class PositionalEncoding2D(geom_nn.PositionalEncoding):
    """
    Applies 2D sinusoidal position encoding:

    .. math::
        PE(x,y,2i)= \\sin(x / (10000^{2i / (d_model / 2)})) + \\sin(y / (10000^{2i / (d_model / 2)})) \\\\
        PE(x,y,2i+1)= \\cos(x / (10000^{2i / (d_model / 2)})) + cos(y / (10000^{2i / (d_model / 2)}))

    with:
        - x, y: spatial coordinates
        - :math:`d_model`: dimensionality of the model embedding

    """
    def forward(self, x: torch.Tensor, batch_index: Optional[torch.Tensor] = None, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass, computing the 2D positional encoding.

        :param x: input tensor
        :param batch_index: batch vector, assigning each node to a specific example
        :param pos: matrix containing the position coordinates for each element in x
        :return: the modified input tensor, with added positional encoding (scaled by granularity)
        """
        # computes <x|y> / (10000^{2i / (d_model / 2)})
        pos = pos.T
        pos_x = pos[0].view(-1, 1) * self.frequency.view(1, -1)
        pos_y = pos[1].view(-1, 1) * self.frequency.view(1, -1)

        pe_sin = torch.sin(pos_x) + torch.sin(pos_y)
        pe_cos = torch.cos(pos_x) + torch.cos(pos_y)

        return x + torch.cat([pe_sin, pe_cos], dim=1) / self.granularity
