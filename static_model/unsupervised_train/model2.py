# unsupervised_train/model.py
"""
无监督图表示学习模型（InfoGraph-like / contrastive）
使用 GINConv 作为编码器，训练目标是使图的局部节点表示与图的全局表示一致（对比学习）。

支持以下两种节点输入方式：
1) 已有节点向量 data.x（直接使用）
2) 离散 id: data.type 和 data.token -> 内部 nn.Embedding 映射得到节点向量

主要类：
- GINEncoder: 图编码器（支持 embedding 层或直接使用 data.x）
- GAE_GIN_lightning: LightningModule 封装训练 / 验证 / 对比损失等

如何使用（简要）：
1. 如果 preprocess 已在 .pt 中存了 data.x（节点向量），在初始化 GAE_GIN_lightning 时传入 in_channels=数据维度。
2. 如果只有 type/token id，则在初始化 GINEncoder 时传入 type_vocab_size, token_vocab_size, 并把 in_channels=None（或忽略）。
3. 准备 PyG DataLoader（每个 batch 的 Data 必须包含 edge_index, batch, connected_node_mask）
4. 用 pl.Trainer 去训练 model。

好处：
- 无监督训练，不依赖标签，能学到结构化代码图的通用表示（可用于后续下游任务）
- 支持节点级别与图级别联合对比，提升表示质量
"""

from typing import Optional
import math

import torch
import torch.nn.functional as F
from torch import nn
import lightning.pytorch as pl
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp


class FF(nn.Module):
    """Simple residual MLP used as decoder/transformer."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.block = Sequential(
            Linear(input_dim, input_dim),
            ReLU(),
            Linear(input_dim, input_dim),
            ReLU(),
            Linear(input_dim, input_dim),
            ReLU()
        )
        self.linear_shortcut = Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class GINEncoder(nn.Module):
    """
    GIN-based encoder.

    支持两种输入模式：
    - 如果传入 in_channels (int > 0) 且 data.x 存在，则直接使用 data.x 作为节点输入。
    - 否则，若传入 type_vocab_size 与 token_vocab_size，则使用内部 nn.Embedding 将 data.type 和 data.token 映射为向量并拼接为节点输入。

    参数:
    - in_channels: 当 preprocess 已经提供 node feature 时的维度（或 None）
    - type_vocab_size: 若要基于 type id 训练则传入词表大小（int）
    - token_vocab_size: 若要基于 token id 训练则传入词表大小（int）
    - type_emb_dim, token_emb_dim: 嵌入维度（若使用 embedding）
    - hidden_dim: GIN 每层输出维度
    - num_gc_layers: GIN 层数
    """
    def __init__(self,
                 in_channels: Optional[int] = None,
                 type_vocab_size: Optional[int] = None,
                 token_vocab_size: Optional[int] = None,
                 type_emb_dim: int = 64,
                 token_emb_dim: int = 64,
                 hidden_dim: int = 128,
                 num_gc_layers: int = 2):
        super().__init__()
        self.num_gc_layers = num_gc_layers
        self.hidden_dim = hidden_dim

        # If in_channels provided, we will expect data.x as input.
        self.in_channels = in_channels

        # embedding layers for type/token (optional)
        self.use_embeddings = False
        if in_channels is None:
            if type_vocab_size is None or token_vocab_size is None:
                raise ValueError("如果 in_channels=None，则必须提供 type_vocab_size 和 token_vocab_size")
            self.use_embeddings = True
            self.type_embedding = nn.Embedding(type_vocab_size, type_emb_dim, padding_idx=0)
            self.token_embedding = nn.Embedding(token_vocab_size, token_emb_dim, padding_idx=0)
            initial_node_dim = type_emb_dim + token_emb_dim
            # We'll project initial_node_dim -> hidden_dim for GIN first layer convenience
            self.input_proj = Linear(initial_node_dim, hidden_dim)
            self.effective_in_channels = hidden_dim
        else:
            self.use_embeddings = False
            self.effective_in_channels = in_channels

        # Build GINConv layers
        self.convs = nn.ModuleList()
        self.batch_norm = nn.ModuleList()

        for i in range(self.num_gc_layers):
            if i == 0:
                nn_mlp = Sequential(Linear(self.effective_in_channels, self.hidden_dim),
                                    ReLU(),
                                    Linear(self.hidden_dim, self.hidden_dim))
            else:
                nn_mlp = Sequential(Linear(self.hidden_dim, self.hidden_dim),
                                    ReLU(),
                                    Linear(self.hidden_dim, self.hidden_dim))
            conv = GINConv(nn_mlp)
            bn = nn.BatchNorm1d(self.hidden_dim)
            self.convs.append(conv)
            self.batch_norm.append(bn)

    @property
    def embedding_dim(self) -> int:
        """最终的图级 embedding 维度 = hidden_dim * num_gc_layers（每层拼接）"""
        return self.hidden_dim * self.num_gc_layers

    def forward(self, x, edge_index, batch, connected_node_mask, type_ids=None, token_ids=None):
        """
        前向：
        - x: 若 preprocess 已提供 node feature，则传入 data.x
        - 否则传入 None，并提供 type_ids 和 token_ids（LongTensor）
        - edge_index, batch: PyG 标准
        - connected_node_mask: Bool/Byte Tensor，shape [num_nodes]，指示哪些节点为有效节点
        返回：
        - global_vector: [num_graphs, hidden_dim * num_gc_layers]
        - local_vector: [num_valid_nodes, hidden_dim * num_gc_layers] （按 connected_node_mask 过滤）
        """
        # Build initial node features if needed
        if self.use_embeddings:
            # expect type_ids and token_ids to be LongTensors on same device
            if type_ids is None or token_ids is None:
                raise ValueError("使用 embedding 模式时，需要提供 type_ids 和 token_ids")
            type_emb = self.type_embedding(type_ids)  # [N, type_emb_dim]
            token_emb = self.token_embedding(token_ids)  # [N, token_emb_dim]
            x = torch.cat([type_emb, token_emb], dim=1)  # [N, initial_node_dim]
            x = self.input_proj(x)  # project to hidden_dim for first layer
        else:
            # Expect x is provided (float tensor)
            if x is None:
                raise ValueError("in_channels provided 时，forward 需要 data.x 作为 x 输入")

        xs = []
        # run GIN layers, collect layer outputs
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.batch_norm[i](x)
            # filter by connected_node_mask for local representations
            if connected_node_mask is not None:
                xs.append(x[connected_node_mask])
            else:
                xs.append(x)

        # global pooling on filtered nodes
        if connected_node_mask is not None:
            # we need to pool per-graph using batch but only include connected nodes
            # build masked batch indices for connected nodes
            masked_batch = batch[connected_node_mask]
            xpool = [global_add_pool(x[connected_node_mask], masked_batch) for x in xs]
        else:
            xpool = [global_add_pool(x, batch) for x in xs]

        global_vector = torch.cat(xpool, dim=1)   # [num_graphs, hidden_dim * num_layers]
        local_vector = torch.cat(xs, dim=1)       # [num_valid_nodes, hidden_dim * num_layers]
        return global_vector, local_vector


class GAE_GIN_lightning(pl.LightningModule):
    """
    Lightning 封装的无监督训练模块（InfoGraph 风格的节点-图对比）

    初始化参数说明：
    - in_channels: 若对 data.x 使用，传入节点特征维度；若使用 type/token id，请传入 None 并在 encoder 中提供 vocab sizes
    - out_channels: 未被直接使用（保留以兼容外部接口）
    - batch_size: used for logging
    - encoder_kwargs: dict, 传递给 GINEncoder 的其他参数（type_vocab_size, token_vocab_size 等）
    """
    def __init__(self, in_channels: Optional[int], out_channels: int, batch_size: int, encoder_kwargs: dict = None):
        super().__init__()
        encoder_kwargs = encoder_kwargs or {}
        # instantiate encoder
        # If in_channels is not None, pass it through
        self.encoder = GINEncoder(in_channels=in_channels, **encoder_kwargs) if in_channels is None else GINEncoder(in_channels=in_channels, **encoder_kwargs)
        self.embedding_dimension = self.encoder.embedding_dim
        self.local_decoder = FF(self.embedding_dimension)
        self.global_decoder = FF(self.embedding_dimension)
        self.batch_size = batch_size
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-5)

    def forward(self, data, mode="predict"):
        """
        data: a PyG Batch with fields:
            - x (optional): node features [N, in_channels]
            - type (optional): LongTensor [N]
            - token (optional): LongTensor [N]
            - edge_index
            - batch
            - connected_node_mask: Bool tensor [N]
        mode: "train" or "predict"
        """
        device = self.device
        # extract fields
        x = getattr(data, 'x', None)
        type_ids = getattr(data, 'type', None)
        token_ids = getattr(data, 'token', None)
        edge_index = data.edge_index
        batch = data.batch
        connected_node_mask = getattr(data, 'connected_node_mask', None)

        if type_ids is not None:
            type_ids = type_ids.to(device)
        if token_ids is not None:
            token_ids = token_ids.to(device)
        if x is not None:
            x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        if connected_node_mask is not None:
            connected_node_mask = connected_node_mask.to(device)

        global_vector, local_vector = self.encoder(x, edge_index, batch, connected_node_mask, type_ids, token_ids)
        global_vector = self.global_decoder(global_vector)
        if mode == "train":
            local_vector = self.local_decoder(local_vector)
            return global_vector, local_vector
        elif mode == "predict":
            return global_vector
        else:
            raise ValueError(f"mode {mode} not supported")

    def __contrastive_loss(self, data):
        """
        使用 InfoGraph 风格的对比式重构损失：
        - 对每个节点 local_vector，计算它与所有图的 global_vector 的相似度矩阵
        - 正样本：节点所属图的 global_vector（positive_mask）
        - 负样本：节点不所属图的 global_vector（negative_mask）
        - 损失按 InfoNCE/对比期望式计算（与示例一致）
        """
        # global/local vectors
        global_vector, local_vector = self.forward(data, mode="train")

        num_graphs = global_vector.size(0)
        num_nodes = local_vector.size(0)
        device = global_vector.device

        # build masks
        positive_mask = torch.zeros((num_nodes, num_graphs), device=device)
        negative_mask = torch.ones((num_nodes, num_graphs), device=device)
        for node_idx, graph_idx in enumerate(data.batch):
            # data.batch is size [total_nodes], but in this function data should be the batch containing only valid nodes
            positive_mask[node_idx, graph_idx] = 1.0
            negative_mask[node_idx, graph_idx] = 0.0

        # similarity matrix [num_nodes, num_graphs]
        reconstruct_similarity = torch.mm(local_vector, global_vector.t())  # inner product

        def get_positive_expectation(p_samples, average=True):
            Ep = math.log(2.) - F.softplus(-p_samples)
            return Ep.mean() if average else Ep

        def get_negative_expectation(q_samples, average=True):
            En = F.softplus(-q_samples) + q_samples - math.log(2.)
            return En.mean() if average else En

        loss_positive = get_positive_expectation(reconstruct_similarity * positive_mask, average=False).sum()
        loss_negative = get_negative_expectation(reconstruct_similarity * negative_mask, average=False).sum()
        # normalize
        loss_positive = loss_positive / num_nodes
        loss_negative = loss_negative / (num_nodes * max(1, (num_graphs - 1)))
        return loss_negative - loss_positive

    def training_step(self, data, batch_idx):
        """
        注意：这里示例接受 data 为 dict of graph_type -> batch
        这是照搬你之前示例的训练 loop，如果你在 DataLoader 中直接返回单一 batch，可以相应修改。
        """
        losses = []
        # data may be dict of batches, or single batch
        if isinstance(data, dict):
            for graph_type, batch in data.items():
                loss = self.__contrastive_loss(batch)
                losses.append(loss)
        else:
            loss = self.__contrastive_loss(data)
            losses.append(loss)

        loss = torch.mean(torch.stack(losses))
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, data, batch_idx):
        losses = []
        if isinstance(data, dict):
            for graph_type, batch in data.items():
                loss = self.__contrastive_loss(batch)
                losses.append(loss)
        else:
            loss = self.__contrastive_loss(data)
            losses.append(loss)
        loss = torch.mean(torch.stack(losses))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
