"""
无监督图表示学习模型（InfoGraph-like / contrastive）
使用 GINConv 作为编码器，训练目标是使图的局部节点表示与图的全局表示一致（对比学习）。

仅支持一种节点输入方式：
1) 已有节点向量 data.x（直接使用预处理好的节点特征）

主要类：
- GINEncoder: 图编码器（仅支持直接使用 data.x）
- GAE_GIN_lightning: LightningModule 封装训练 / 验证 / 对比损失等

如何使用（简要）：
1. 确保预处理已在 .pt 中存储了 data.x（节点向量）
2. 初始化 GAE_GIN_lightning 时传入 in_channels=节点特征维度
3. 准备 PyG DataLoader（每个 batch 的 Data 必须包含 x, edge_index, batch, connected_node_mask）
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
    GIN-based encoder（仅支持直接使用预处理节点特征）

    仅支持一种输入模式：
    - 必须传入 in_channels (int > 0)，且输入数据需包含 data.x（预处理好的节点特征）

    参数:
    - in_channels: 预处理节点特征的维度
    - hidden_dim: GIN 每层输出维度
    - num_gc_layers: GIN 层数
    """

    def __init__(self,
                 in_channels: int,
                 hidden_dim: int = 128,
                 num_gc_layers: int = 2):
        super().__init__()
        self.num_gc_layers = num_gc_layers
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.effective_in_channels = in_channels  # GIN第一层输入维度

        # Build GINConv layers
        self.convs = nn.ModuleList()
        self.batch_norm = nn.ModuleList()

        for i in range(self.num_gc_layers):
            if i == 0:
                # 第一层输入维度为预处理特征维度
                nn_mlp = Sequential(Linear(self.effective_in_channels, self.hidden_dim),
                                    ReLU(),
                                    Linear(self.hidden_dim, self.hidden_dim))
            else:
                # 后续层输入输出均为hidden_dim
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

    def forward(self, x, edge_index, batch, connected_node_mask):
        """
        前向传播：
        - x: 预处理好的节点特征（data.x）
        - edge_index, batch: PyG 标准参数
        - connected_node_mask: Bool/Byte Tensor，标记有效节点
        返回：
        - global_vector: [num_graphs, hidden_dim * num_gc_layers]
        - local_vector: [num_valid_nodes, hidden_dim * num_gc_layers]
        """
        # 校验输入
        if x is None:
            raise ValueError("必须传入预处理好的节点特征 x（即 data.x）")

        xs = []
        # 执行GIN层计算，收集各层输出
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.batch_norm[i](x)
            # 按掩码过滤有效节点
            if connected_node_mask is not None:
                xs.append(x[connected_node_mask])
            else:
                xs.append(x)

        # 全局池化（仅使用有效节点）
        if connected_node_mask is not None:
            masked_batch = batch[connected_node_mask]
            xpool = [global_add_pool(x_layer, masked_batch) for x_layer in xs]
        else:
            xpool = [global_add_pool(x_layer, batch) for x_layer in xs]

        # 拼接各层结果
        global_vector = torch.cat(xpool, dim=1)  # 图级向量
        local_vector = torch.cat(xs, dim=1)  # 节点级向量
        return global_vector, local_vector


class GAE_GIN_lightning(pl.LightningModule):
    """
    Lightning 封装的无监督训练模块（InfoGraph 风格的节点-图对比）

    初始化参数说明：
    - in_channels: 节点特征维度（必须提供）
    - out_channels: 保留以兼容外部接口（未直接使用）
    - batch_size: 用于日志记录
    - encoder_kwargs: 传递给 GINEncoder 的其他参数（hidden_dim, num_gc_layers等）
    """

    def __init__(self, in_channels: int, out_channels: int, batch_size: int, encoder_kwargs: dict = None):
        super().__init__()
        encoder_kwargs = encoder_kwargs or {}
        # 实例化编码器（仅支持模式1）
        self.encoder = GINEncoder(in_channels=in_channels, **encoder_kwargs)
        self.embedding_dimension = self.encoder.embedding_dim
        self.local_decoder = FF(self.embedding_dimension)
        self.global_decoder = FF(self.embedding_dimension)
        self.batch_size = batch_size
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-5)

    def forward(self, data, mode="predict"):
        """
        data: PyG Batch，需包含：
            - x: 节点特征 [N, in_channels]
            - edge_index: 边连接关系
            - batch: 节点所属图索引
            - connected_node_mask: 有效节点掩码 [N]
        mode: "train" 或 "predict"
        """
        device = self.device
        # 提取必要字段
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        connected_node_mask = getattr(data, 'connected_node_mask', None)
        if connected_node_mask is not None:
            connected_node_mask = connected_node_mask.to(device)

        # 编码得到全局和局部向量
        global_vector, local_vector = self.encoder(x, edge_index, batch, connected_node_mask)
        global_vector = self.global_decoder(global_vector)

        if mode == "train":
            local_vector = self.local_decoder(local_vector)
            return global_vector, local_vector
        elif mode == "predict":
            return global_vector
        else:
            raise ValueError(f"不支持的模式: {mode}")

    def __contrastive_loss(self, data):
        """
        InfoGraph 风格的对比式重构损失：
        - 计算节点局部向量与图全局向量的相似度
        - 正样本：节点所属图的全局向量
        - 负样本：其他图的全局向量
        """
        global_vector, local_vector = self.forward(data, mode="train")

        num_graphs = global_vector.size(0)
        num_nodes = local_vector.size(0)
        device = global_vector.device

        # 构建正负样本掩码
        positive_mask = torch.zeros((num_nodes, num_graphs), device=device)
        negative_mask = torch.ones((num_nodes, num_graphs), device=device)
        for node_idx, graph_idx in enumerate(data.batch):
            positive_mask[node_idx, graph_idx] = 1.0  # 正样本掩码
            negative_mask[node_idx, graph_idx] = 0.0  # 负样本掩码

        # 相似度矩阵（内积）
        reconstruct_similarity = torch.mm(local_vector, global_vector.t())

        def get_positive_expectation(p_samples, average=True):
            Ep = math.log(2.) - F.softplus(-p_samples)
            return Ep.mean() if average else Ep

        def get_negative_expectation(q_samples, average=True):
            En = F.softplus(-q_samples) + q_samples - math.log(2.)
            return En.mean() if average else En

        # 计算损失
        loss_positive = get_positive_expectation(reconstruct_similarity * positive_mask, average=False).sum()
        loss_negative = get_negative_expectation(reconstruct_similarity * negative_mask, average=False).sum()

        # 归一化损失
        loss_positive = loss_positive / num_nodes
        loss_negative = loss_negative / (num_nodes * max(1, (num_graphs - 1)))
        return loss_negative - loss_positive

    def training_step(self, data, batch_idx):
        """训练步骤：计算对比损失并记录日志"""
        if isinstance(data, dict):
            # 处理字典类型的批量数据
            losses = [self.__contrastive_loss(batch) for _, batch in data.items()]
        else:
            losses = [self.__contrastive_loss(data)]

        loss = torch.mean(torch.stack(losses))
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, data, batch_idx):
        """验证步骤：计算对比损失并记录日志"""
        if isinstance(data, dict):
            losses = [self.__contrastive_loss(batch) for _, batch in data.items()]
        else:
            losses = [self.__contrastive_loss(data)]

        loss = torch.mean(torch.stack(losses))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
