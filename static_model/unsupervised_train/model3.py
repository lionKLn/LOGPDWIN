"""

不使用lightning
无监督图表示学习模型（InfoGraph-like / contrastive）
使用 GINConv 作为编码器，训练目标是使图的局部节点表示与图的全局表示一致（对比学习）。

仅支持一种节点输入方式：
1) 已有节点向量 data.x（直接使用预处理好的节点特征）

主要类：
- GINEncoder: 图编码器（仅支持直接使用 data.x）
- GAE_GIN: 纯 PyTorch 封装训练 / 验证 / 对比损失等（替代原 Lightning 类）

如何使用（简要）：
1. 确保预处理已在 .pt 中存储了 data.x（节点向量）
2. 初始化 GAE_GIN 时传入 in_channels=节点特征维度、设备（如 'npu:0' 或 'cuda:0'）
3. 准备 PyG DataLoader（每个 batch 的 Data 必须包含 x, edge_index, batch, connected_node_mask）
4. 调用 train_loop 方法启动训练，调用 validate 方法进行验证
"""

from typing import Optional, Dict, List
import math
import time
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Batch
from torch.utils.data import DataLoader


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


class GAE_GIN(nn.Module):
    """
    纯 PyTorch 封装的无监督训练模块（替代原 Lightning 类）
    保留 InfoGraph 风格的节点-图对比损失逻辑

    初始化参数说明：
    - in_channels: 节点特征维度（必须提供）
    - out_channels: 保留以兼容外部接口（未直接使用）
    - device: 训练设备（如 'npu:0'、'cuda:0' 或 'cpu'）
    - encoder_kwargs: 传递给 GINEncoder 的其他参数（hidden_dim, num_gc_layers等）
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 device: str,
                 encoder_kwargs: dict = None):
        super().__init__()
        encoder_kwargs = encoder_kwargs or {}
        # 实例化编码器（仅支持模式1）
        self.encoder = GINEncoder(in_channels=in_channels, **encoder_kwargs)
        self.embedding_dimension = self.encoder.embedding_dim
        self.local_decoder = FF(self.embedding_dimension)
        self.global_decoder = FF(self.embedding_dimension)
        self.device = device  # 手动管理设备
        # 移动模型到指定设备
        self.to(self.device)

    def _prepare_data(self, data: Batch) -> Batch:
        """将数据移动到指定设备（替代 Lightning 的自动设备分配）"""
        # 提取必要字段并移动设备
        data.x = data.x.to(self.device)
        data.edge_index = data.edge_index.to(self.device)
        data.batch = data.batch.to(self.device)
        # 处理可选的 connected_node_mask
        if hasattr(data, 'connected_node_mask') and data.connected_node_mask is not None:
            data.connected_node_mask = data.connected_node_mask.to(self.device)
        return data

    def forward(self, data: Batch, mode: str = "predict"):
        """
        前向传播（替代 Lightning 的 forward）
        data: PyG Batch，需包含 x, edge_index, batch, [connected_node_mask]
        mode: "train"（返回全局+局部向量）或 "predict"（仅返回全局向量）
        """
        # 先准备数据（移动设备）
        data = self._prepare_data(data)

        # 编码得到全局和局部向量
        global_vector, local_vector = self.encoder(
            x=data.x,
            edge_index=data.edge_index,
            batch=data.batch,
            connected_node_mask=getattr(data, 'connected_node_mask', None)
        )
        global_vector = self.global_decoder(global_vector)

        if mode == "train":
            local_vector = self.local_decoder(local_vector)
            return global_vector, local_vector
        elif mode == "predict":
            return global_vector
        else:
            raise ValueError(f"不支持的模式: {mode}")

    def contrastive_loss(self, data: Batch) -> torch.Tensor:
        """
        InfoGraph 风格的对比式重构损失（替代原 __contrastive_loss）
        计算节点局部向量与图全局向量的相似度损失
        """
        global_vector, local_vector = self.forward(data, mode="train")

        num_graphs = global_vector.size(0)
        num_nodes = local_vector.size(0)

        # 构建正负样本掩码
        positive_mask = torch.zeros((num_nodes, num_graphs), device=self.device)
        negative_mask = torch.ones((num_nodes, num_graphs), device=self.device)
        for node_idx, graph_idx in enumerate(data.batch):
            positive_mask[node_idx, graph_idx] = 1.0  # 正样本：节点所属图
            negative_mask[node_idx, graph_idx] = 0.0  # 负样本：排除自身图

        # 相似度矩阵（内积）
        reconstruct_similarity = torch.mm(local_vector, global_vector.t())

        def get_positive_expectation(p_samples: torch.Tensor, average: bool = True) -> torch.Tensor:
            Ep = math.log(2.) - F.softplus(-p_samples)
            return Ep.mean() if average else Ep

        def get_negative_expectation(q_samples: torch.Tensor, average: bool = True) -> torch.Tensor:
            En = F.softplus(-q_samples) + q_samples - math.log(2.)
            return En.mean() if average else En

        # 计算损失
        loss_positive = get_positive_expectation(reconstruct_similarity * positive_mask, average=False).sum()
        loss_negative = get_negative_expectation(reconstruct_similarity * negative_mask, average=False).sum()

        # 归一化损失（与原逻辑一致）
        loss_positive = loss_positive / num_nodes
        loss_negative = loss_negative / (num_nodes * max(1, (num_graphs - 1)))
        return loss_negative - loss_positive

    def train_step(self, data: Batch, optimizer: torch.optim.Optimizer) -> float:
        """
        单批次训练步骤（替代 Lightning 的 training_step）
        返回当前批次的损失值
        """
        self.train()  # 开启训练模式（启用 dropout、batch norm 训练模式）
        optimizer.zero_grad()  # 清空梯度

        # 计算损失（处理字典/普通 Batch 两种数据格式）
        if isinstance(data, Dict):
            losses = [self.contrastive_loss(batch) for _, batch in data.items()]
        else:
            losses = [self.contrastive_loss(data)]

        loss = torch.mean(torch.stack(losses))
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

        return loss.item()  # 返回标量损失值

    def validate_step(self, data: Batch) -> float:
        """
        单批次验证步骤（替代 Lightning 的 validation_step）
        返回当前批次的损失值
        """
        self.eval()  # 开启评估模式（禁用 dropout、固定 batch norm 统计量）
        with torch.no_grad():  # 禁用梯度计算，节省内存和时间
            # 计算损失（与训练步骤逻辑一致）
            if isinstance(data, Dict):
                losses = [self.contrastive_loss(batch) for _, batch in data.items()]
            else:
                losses = [self.contrastive_loss(data)]

            loss = torch.mean(torch.stack(losses))

        return loss.item()  # 返回标量损失值

    def train_loop(self,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   epochs: int,
                   lr: float = 5e-5) -> None:
        """
        完整训练循环（替代 Lightning 的 Trainer）
        包含训练、验证、日志打印功能
        """
        # 初始化优化器（与原 Lightning configure_optimizers 一致）
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # 训练循环
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            train_total_loss = 0.0

            # 训练阶段
            for batch_idx, batch in enumerate(train_loader):
                batch_loss = self.train_step(batch, optimizer)
                train_total_loss += batch_loss

            # 计算训练集平均损失
            train_avg_loss = train_total_loss / len(train_loader)

            # 验证阶段
            val_total_loss = 0.0
            for batch_idx, batch in enumerate(val_loader):
                batch_loss = self.validate_step(batch)
                val_total_loss += batch_loss

            # 计算验证集平均损失
            val_avg_loss = val_total_loss / len(val_loader)

            # 打印日志（模拟 Lightning 的 prog_bar 日志）
            epoch_time = time.time() - start_time
            print(f"[Epoch {epoch:03d}/{epochs}] "
                  f"Train Loss: {train_avg_loss:.6f} | "
                  f"Val Loss: {val_avg_loss:.6f} | "
                  f"Time: {epoch_time:.2f}s")

    def predict_embedding(self, data_loader: DataLoader) -> List[torch.Tensor]:
        """
        预测图嵌入（基于训练好的模型提取图级表示）
        返回所有图的全局嵌入列表
        """
        self.eval()
        embeddings = []
        with torch.no_grad():
            for batch in data_loader:
                batch = self._prepare_data(batch)
                global_emb = self.forward(batch, mode="predict")
                embeddings.append(global_emb.cpu())  # 移到 CPU 便于后续处理
        return torch.cat(embeddings, dim=0)  # 拼接所有批次的嵌入