#dataloader
from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.typing import TensorFrame, torch_frame
import copy


class CPGCollater:
    """
    CPGCollater 类用于在 DataLoader 中将多个图样本合并成一个批次（Batch）。
    支持同时返回单一类型图（如 pdg）或全部图类型（ast、cfg、ddg、pdg）。

    参数:
        dataset: 图数据集对象或图数据列表
        graph_type: 图类型，可选 ['ast', 'cfg', 'ddg', 'pdg', 'all']
    """

    def __init__(
            self,
            dataset: Union[Dataset, Sequence[BaseData]],
            graph_type: str = 'pdg',
    ):
        self.dataset = dataset
        self.graph_type = graph_type
        # 限制 graph_type 必须在指定集合内
        if graph_type not in ['ast', 'cfg', 'ddg', 'pdg', 'all']:
            raise ValueError("graph_type must be one of ['ast', 'cfg', 'ddg', 'pdg', 'all']")

    def __call__(self, batch: List[Any]) -> Any:
        """
        将单个样本组成的 batch 列表合并成一个 Batch 对象。
        如果 graph_type='all'，则分别生成4种不同的图。
        """
        if self.graph_type != 'all':
            # 单一类型图处理
            batch = Batch.from_data_list(batch)
            return {
                self.graph_type: self._filter_graph(batch, self.graph_type)
            }
        else:
            # 多类型图情况，依次复制并过滤出4种图类型
            base_batch = Batch.from_data_list(batch)
            return {
                'ast': self._filter_graph(copy.deepcopy(base_batch), 'ast'),
                'cfg': self._filter_graph(copy.deepcopy(base_batch), 'cfg'),
                'ddg': self._filter_graph(copy.deepcopy(base_batch), 'ddg'),
                'pdg': self._filter_graph(copy.deepcopy(base_batch), 'pdg'),
            }

    def _filter_graph(self, batch: Batch, graph_type: str) -> Batch:
        """
        根据图类型（ast, cfg, ddg, pdg）筛选边与节点。
        """
        # 将不同类型边的掩码（bool向量）堆叠成边属性矩阵 [num_edges, 3]
        edge_attr = torch.stack([
            batch.ast,  # AST 类型边
            batch.cfg,  # CFG 类型边
            batch.ddg  # DDG 类型边
        ], dim=1).float()

        # 根据不同图类型选择保留的边
        if graph_type == 'ast':
            keep_mask = batch.ast
        elif graph_type == 'cfg':
            # cfg图保留 ast 和 cfg 边
            keep_mask = batch.ast | batch.cfg
        elif graph_type == 'ddg':
            # ddg图保留 ast 和 ddg 边
            keep_mask = batch.ast | batch.ddg
        elif graph_type == 'pdg':
            # pdg图保留 ast + cfg + ddg
            keep_mask = batch.ast | batch.cfg | batch.ddg
        else:
            raise ValueError("Invalid graph_type")

        # 根据 keep_mask 筛选边索引和边属性
        batch.edge_index = batch.edge_index[:, keep_mask]
        batch.edge_attr = edge_attr[keep_mask]

        # 标记当前图中哪些节点与至少一条边相连
        connected_nodes = torch.unique(batch.edge_index)
        connected_node_mask = torch.zeros(batch.num_nodes, dtype=torch.bool, device=batch.x.device)
        connected_node_mask[connected_nodes] = True
        batch.connected_node_mask = connected_node_mask

        return batch


class CPGDataLoader(torch.utils.data.DataLoader):
    r"""
    CPGDataLoader 类：继承自 torch.utils.data.DataLoader，
    专门用于加载图数据（继承自 torch_geometric.data.Dataset）。

    它会自动使用 CPGCollater 作为 collate_fn 将图对象合并成一个 Batch。

    参数:
        dataset (Dataset): 图数据集对象
        batch_size (int): 每个 batch 的样本数，默认 1
        shuffle (bool): 是否在每个 epoch 打乱数据
        graph_type (str): 图类型，可选 ['ast','cfg','ddg','pdg','all']
        follow_batch (List[str]): PyG 用于异构图的批次划分键（一般用不到）
        exclude_keys (List[str]): 排除不需要打包的属性键
        **kwargs: 传递给 PyTorch DataLoader 的其它参数
    """

    def __init__(
            self,
            dataset: Union[Dataset, Sequence[BaseData]],
            batch_size: int = 1,
            shuffle: bool = False,
            graph_type: str = 'pdg',
            follow_batch: Optional[List[str]] = None,
            exclude_keys: Optional[List[str]] = None,
            **kwargs,
    ):
        # 避免与 PyTorch Lightning 内部冲突
        kwargs.pop('collate_fn', None)

        # 兼容旧版本 Lightning
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        # graph_type 参数检查
        if graph_type not in ['ast', 'cfg', 'ddg', 'pdg', 'all']:
            raise ValueError("graph_type must be one of ['ast', 'cfg', 'ddg', 'pdg', 'all']")
        self.graph_type = graph_type

        # 调用父类 DataLoader 初始化
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=CPGCollater(dataset, graph_type),  # 使用自定义的批处理逻辑
            **kwargs,
        )
