# unsupervised_train/data_loader.py
import os
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class GraphDataset(torch.utils.data.Dataset):
    """
    从指定目录加载通过 preprocess 生成的图数据 (.pt 文件)
    每个样本是一个 torch_geometric.data.Data 对象
    """
    def __init__(self, graph_dir):
        self.graph_dir = graph_dir
        self.graph_files = [
            os.path.join(graph_dir, f)
            for f in os.listdir(graph_dir)
            if f.endswith(".pt")
        ]
        if not self.graph_files:
            raise FileNotFoundError(f"目录 {graph_dir} 下未找到任何 .pt 图文件")

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        # 关键修改：添加 map_location=cpu，强制将数据加载到CPU，规避HPU设备问题
        graph = torch.load(self.graph_files[idx], map_location=torch.device('cpu'))
        if not hasattr(graph, "connected_node_mask"):
            # 添加 connected_node_mask，用于局部对比学习，默认标记所有节点为有效
            graph.connected_node_mask = torch.ones(graph.x.size(0), dtype=torch.bool)
        return graph


def build_dataloaders(graph_dir, batch_size=8, num_workers=0, shuffle=True):
    """
    构建 PyG DataLoader
    """
    dataset = GraphDataset(graph_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return loader


if __name__ == "__main__":
    # ✅ 测试加载器是否能正确工作
    graph_dir = "../data/graph_dataset"
    loader = build_dataloaders(graph_dir, batch_size=4)

    for batch in loader:
        print(batch)
        print("Batch size:", batch.num_graphs)
        print("Node feature shape:", batch.x.shape)
        break
