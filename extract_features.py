# extract_features.py
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import argparse
import sys
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 从配置文件引入路径
from config import DATASET_ROOT, WEIGHTS_ROOT, FEATURES_ROOT, DEFAULT_WEIGHT_FILE

from utils.data_utils import load_genus_dataset, load_species_dataset
from utils.model_utils import build_model

def extract_features(model: nn.Module, data_loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    all_features = []
    with torch.no_grad():
        split_name = "train" if "train" in data_loader.dataset.root.lower() else "test"
        for images, _ in tqdm(data_loader, desc=f"正在提取 '{split_name}' 集特征"):
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu().numpy())
    return np.concatenate(all_features, axis=0)

def process_and_save_split(model: nn.Module, dataset: Dataset, split_name: str, 
                           target_dir: str, batch_size: int, device: str):
    print("-" * 30)
    print(f"开始处理 '{split_name}' 数据集，共 {len(dataset)} 张图片...")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    extracted_features = extract_features(model, data_loader, device)
    labels = np.array(dataset.targets)

    feature_save_path = os.path.join(target_dir, f"{split_name}_features.npy")
    label_save_path = os.path.join(target_dir, f"{split_name}_labels.npy")

    np.save(feature_save_path, extracted_features)
    np.save(label_save_path, labels)

    print(f"✓ '{split_name}' 特征已保存到: {feature_save_path}")
    print(f"✓ 标签已保存到: {label_save_path}")

def main(args):
    # ====== 1. 构建特征保存路径 ======
    if args.mode == 'genus':
        target_output_dir = os.path.join(FEATURES_ROOT, 'genus', args.data_type)
    elif args.mode == 'species':
        if not args.target_genus:
            raise ValueError("--mode 'species' 需要指定 --target_genus")
        target_output_dir = os.path.join(FEATURES_ROOT, 'species', args.target_genus, args.data_type)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"特征将保存到: {target_output_dir}")

    # ====== 2. 数据路径（与训练一致）======
    dataset_path = os.path.join(DATASET_ROOT, args.data_type, args.dataset_name)
    print(f"加载数据集: {dataset_path}")

    if args.mode == 'genus':
        train_data, test_data = load_genus_dataset(dataset_path, args.input_size)
    elif args.mode == 'species':
        train_data, test_data = load_species_dataset(dataset_path, args.target_genus, args.input_size)

    # 如果用户没有传 --weights_path，则使用默认路径
    if args.weights_path:
        weights_path = args.weights_path
    else:
        # 组合默认权重文件路径 -> 例如 /data/weights/pretrain/genus/teeth/best_network.pth
        if args.mode == "genus":
            weights_path = os.path.join(WEIGHTS_ROOT, "pretrain", "genus", args.data_type, DEFAULT_WEIGHT_FILE)
        elif args.mode == "species":
            weights_path = os.path.join(WEIGHTS_ROOT, "pretrain", "species", args.target_genus, args.data_type, DEFAULT_WEIGHT_FILE)
    print(f"使用权重文件: {weights_path}")

    # ====== 4. 加载模型并改为特征提取模式 ======
    num_classes = len(train_data.classes)
    model = build_model(args.model_name, num_classes, use_pretrained=False).to(args.device)
    model.load_state_dict(torch.load(weights_path, map_location=args.device))

    if args.model_name.startswith('efficientnet'):
        model.classifier[-1] = nn.Identity()
    elif args.model_name.startswith('resnet'):
        model.fc = nn.Identity()
    else:
        raise NotImplementedError(f"特征提取逻辑未对 {args.model_name} 实现")

    # ====== 5. 提取并保存特征 ======
    process_and_save_split(model, train_data, 'train', target_output_dir, args.batch_size, args.device)
    process_and_save_split(model, test_data, 'test', target_output_dir, args.batch_size, args.device)

    print("="*20, "全部完成", "="*20)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从训练好的模型中自动提取特征并保存")
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集目录名，例如 2genus_4species')
    parser.add_argument('--mode', type=str, required=True, choices=['genus', 'species'])
    parser.add_argument('--data_type', type=str, required=True, choices=['head', 'teeth'])
    parser.add_argument('--target_genus', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='efficientnet_b7')
    parser.add_argument('--input_size', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--weights_path', type=str, default=None,
                    help="训练好的 .pth 模型权重路径。如果不指定，将从 config.WEIGHTS_ROOT 下的默认路径读取")
    args = parser.parse_args()
    main(args)
