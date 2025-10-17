import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import os
import os.path as osp

# 导入你的GAE_GIN模型类（确保与训练时的定义一致）
from model import GAE_GIN

# ---------------------- 配置参数 ----------------------
# 1. 模型路径
MODEL_PATH = "logs/pdg/2025-05-20_14-30-00/best_pdg.pt"
# 2. 待编码的代码图数据目录（存放.pt文件）
DATA_DIR = "../data/graph_data_to_encode"  # 替换为你的数据目录
# 3. 设备（与训练时一致，或用CPU）
DEVICE = torch.device("npu:4" if torch.npu.is_available() else "cpu")
# 4. 批次大小（根据设备内存调整）
BATCH_SIZE = 512


# ---------------------- 数据加载 ----------------------
class CodeGraphDataset(Dataset):
    """加载待编码的.pt格式代码图数据"""
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.graph_paths = [
            osp.join(root_dir, f) for f in os.listdir(root_dir)
            if f.endswith(".pt")
        ]
        if not self.graph_paths:
            raise ValueError(f"未找到.pt文件，请检查目录：{root_dir}")
        print(f"找到 {len(self.graph_paths)} 个待编码的代码图文件")

    def len(self):
        return len(self.graph_paths)

    def get(self, idx):
        # 加载.pt文件中的PyG Data对象，强制加载到CPU（避免设备冲突）
        data = torch.load(self.graph_paths[idx], map_location="cpu")
        # 确保数据包含模型需要的字段（x, edge_index, ast, cfg, ddg）
        required_fields = ['x', 'edge_index', 'ast', 'cfg', 'ddg']
        for field in required_fields:
            if not hasattr(data, field):
                raise ValueError(f"数据缺失必需字段：{field}，文件：{self.graph_paths[idx]}")
        return data


# ---------------------- 编码函数 ----------------------
def encode_code_graphs(model, dataloader, device):
    """
    对数据加载器中的代码图进行编码，返回图级嵌入特征
    """
    model.eval()  # 切换到评估模式
    all_embeddings = []  # 存储所有图的嵌入

    with torch.no_grad():  # 禁用梯度计算，节省内存
        for batch in dataloader:
            # 将批次数据移至目标设备
            batch = batch.to(device)
            # 调用模型的预测接口，获取图级嵌入（全局向量）
            # 注意：根据你的GAE_GIN实现，可能是forward(mode="predict")或predict_embedding
            graph_embeddings = model.forward(batch, mode="predict")
            # 将嵌入移至CPU并保存
            all_embeddings.append(graph_embeddings.cpu())

    # 拼接所有批次的嵌入，返回形状为 [总图数, 嵌入维度] 的张量
    return torch.cat(all_embeddings, dim=0)


# ---------------------- 主函数 ----------------------
def main():
    # 1. 初始化模型并加载预训练参数
    print("加载模型...")
    # 模型参数需与训练时一致（in_channels=768，与CodeBERT特征维度匹配）
    model = GAE_GIN(
        in_channels=768,
        out_channels=768,
        device=DEVICE
    ).to(DEVICE)
    # 加载预训练参数
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"模型加载完成，设备：{DEVICE}")

    # 2. 加载待编码的数据
    print("\n加载待编码的数据...")
    dataset = CodeGraphDataset(DATA_DIR)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 编码时无需打乱，保持顺序
        num_workers=4   # 多进程加载，根据CPU核心数调整
    )

    # 3. 执行编码
    print("\n开始编码...")
    embeddings = encode_code_graphs(model, dataloader, DEVICE)
    print(f"编码完成，嵌入形状：{embeddings.shape}")  # 应为 [N, 256]（假设嵌入维度256）

    # 4. 保存编码结果（可选）
    save_path = "code_graph_embeddings.pt"
    torch.save(embeddings, save_path)
    print(f"嵌入特征已保存至：{save_path}")


if __name__ == "__main__":
    main()