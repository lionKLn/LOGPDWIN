import os
import os.path as osp
import torch
from torch_geometric.data import Dataset


class GraphDataset(Dataset):
    """
    加载 ../data/graph_dataset 下的所有 .pt 图数据文件。
    每个文件应是 torch.save() 存储的 PyG Data 对象。
    """
    def __init__(self, root_dir="../data/graph_dataset", transform=None, pre_transform=None):
        super().__init__(root=root_dir, transform=transform, pre_transform=pre_transform)
        self.root_dir = osp.abspath(root_dir)

        if not osp.exists(self.root_dir):
            raise FileNotFoundError(f"❌ 路径不存在: {self.root_dir}")

        # 收集所有 .pt 文件路径
        self.graph_paths = [
            osp.join(self.root_dir, f)
            for f in os.listdir(self.root_dir)
            if f.endswith(".pt")
        ]
        if not self.graph_paths:
            raise RuntimeError(f"⚠️ 没有找到任何 .pt 文件，请检查目录: {self.root_dir}")

        print(f"✅ 成功加载 {len(self.graph_paths)} 个图样本。")

    def len(self):
        """返回数据集大小"""
        return len(self.graph_paths)

    def get(self, idx):
        """返回第 idx 个样本（PyG Data 对象）"""
        graph_path = self.graph_paths[idx]
        # 关键修改：强制将数据加载到CPU，避免设备依赖问题
        data = torch.load(graph_path, map_location=torch.device('cpu'))
        return data


if __name__ == "__main__":
    dataset = GraphDataset("../data/graph_dataset")
    print(f"数据集大小: {len(dataset)}")

    # 查看一个样本结构
    data = dataset.get(0)
    print(data)
