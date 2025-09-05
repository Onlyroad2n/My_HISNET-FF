# extract_features.py
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import argparse
import sys
from tqdm import tqdm
import numpy as np

# 添加 utils 目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 复用您项目中已有的工具函数
from utils.data_utils import load_genus_dataset, load_species_dataset
from utils.model_utils import build_model

def extract_features(model: nn.Module, data_loader: DataLoader, device: str) -> np.ndarray:
    """
    核心特征提取函数，模型的输出即特征。
    """
    model.eval()
    all_features = []
    
    with torch.no_grad():
        # 在tqdm的描述中加入数据集信息，更清晰
        split_name = "train" if "train" in data_loader.dataset.root.lower() else "test"
        for images, _ in tqdm(data_loader, desc=f"正在提取 '{split_name}' 集特征"):
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)

def process_and_save_split(model: nn.Module, dataset: Dataset, split_name: str, 
                           target_dir: str, batch_size: int, device: str):
    """
    辅助函数：处理单个数据集（train或test）的完整流程。
    修改点：现在接收一个最终的目标目录 target_dir，并使用简化的文件名。
    """
    print("-" * 30)
    print(f"开始处理 '{split_name}' 数据集，共 {len(dataset)} 张图片...")
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    extracted_features = extract_features(model, data_loader, device)
    labels = np.array(dataset.targets)
    
    # 使用简化的、标准的文件名
    feature_save_path = os.path.join(target_dir, f"{split_name}_features.npy")
    label_save_path = os.path.join(target_dir, f"{split_name}_labels.npy")
    
    np.save(feature_save_path, extracted_features)
    np.save(label_save_path, labels)
    
    print(f"✓ '{split_name}' 数据集特征提取完成。")
    print(f"  - 特征维度: {extracted_features.shape}")
    print(f"  - 特征已保存至: {feature_save_path}")
    print(f"  - 标签已保存至: {label_save_path}")


def main(args):
    """主执行函数"""
    
    # --- 1. 动态构建结构化的输出目录 (核心修改点) ---
    if args.mode == 'genus':
        target_output_dir = os.path.join(args.output_dir, 'genus', args.data_type)
    elif args.mode == 'species':
        if not args.target_genus:
            raise ValueError("--mode 'species' 需要指定 --target_genus。")
        target_output_dir = os.path.join(args.output_dir, 'species', args.target_genus, args.data_type)
    
    print(f"所有特征将被保存在结构化目录中: {target_output_dir}")
    os.makedirs(target_output_dir, exist_ok=True)

    # --- 2. 统一数据加载 ---
    DATASET_ROOT = f'./datasets/{args.data_type}/2genus_4species'
    
    print(f"正在从 {DATASET_ROOT} 加载数据集...")
    if args.mode == 'genus':
        train_data, test_data = load_genus_dataset(DATASET_ROOT, args.input_size)
    elif args.mode == 'species':
        train_data, test_data = load_species_dataset(DATASET_ROOT, args.target_genus, args.input_size)
    
    # --- 3. 模型加载与修改 ---
    num_classes = len(train_data.classes) 
    model = build_model(args.model_name, num_classes, use_pretrained=False).to(args.device)
    model.load_state_dict(torch.load(args.weights_path, map_location=args.device))
    print(f"成功加载权重: {args.weights_path}")

    if args.model_name.startswith('efficientnet'):
        model.classifier[-1] = nn.Identity()
    elif args.model_name.startswith('resnet'):
        model.fc = nn.Identity()
    else:
        raise NotImplementedError(f"特征提取逻辑未对 {args.model_name} 实现。")
    
    # --- 4. 自动化处理流程 ---
    # 调用辅助函数时，传入新构建的 target_output_dir
    process_and_save_split(model, train_data, 'train', target_output_dir, 
                           args.batch_size, args.device)
                           
    process_and_save_split(model, test_data, 'test', target_output_dir, 
                           args.batch_size, args.device)
    
    print("\n" + "="*20 + " 全部完成 " + "="*20)
    print("已成功提取并保存训练集和测试集的特征到指定结构化目录。")
    print("="*65)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从训练好的模型中自动提取特征并保存到结构化目录。")
    # 命令行参数保持不变
    parser.add_argument('--weights_path', type=str, required=True, help="训练好的 .pth 模型权重路径。")
    parser.add_argument('--output_dir', type=str, default="./features", help="保存所有特征的根目录。")
    parser.add_argument('--mode', type=str, required=True, choices=['genus', 'species'], help='设置模式: "genus" 或 "species".')
    parser.add_argument('--data_type', type=str, required=True, choices=['head', 'teeth'], help='数据类型: "head" 或 "teeth".')
    parser.add_argument('--target_genus', type=str, default=None, help='当模式为 "species" 时，需要指定目标属。')
    parser.add_argument('--model_name', type=str, default='efficientnet_b7', help="模型架构名称。")
    parser.add_argument('--input_size', type=int, default=600, help="模型的输入图像分辨率。")
    parser.add_argument('--batch_size', type=int, default=16, help="特征提取时的批处理大小。")
    parser.add_argument('--device', type=str, default='cuda:0', help="计算设备。")

    args = parser.parse_args()
    main(args)