# train.py
import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.utils import tensorboard
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_utils import load_genus_dataset, load_species_dataset
from utils.model_utils import build_model
from utils.train_utils import train_model

# ======= 参数区 (固定参数/超参数) =======
LEARNING_RATE = 0.01
STEP_SIZE = 10
DEVICE = 'cuda'
PATIENCE = 50
# =======================================

# 常见 EfficientNet 模型及其标准输入分辨率
MODEL_RESOLUTIONS = {
    'efficientnet_b0': 224,
    'efficientnet_b1': 240,
    'efficientnet_b2': 260,
    'efficientnet_b3': 300,
    'efficientnet_b4': 380,
    'efficientnet_b5': 456,
    'efficientnet_b6': 528,
    'efficientnet_b7': 600,
}

if __name__ == "__main__":
    # 1. 设置命令行参数解析器
    parser = argparse.ArgumentParser(description='Unified and configurable training script for image classification.')
    # --- 模式与数据控制 ---
    parser.add_argument('--mode', type=str, required=True, choices=['genus', 'species'], help='Set training mode: "genus" or "species".')
    parser.add_argument('--data_type', type=str, default='head', choices=['head', 'teeth'], help='Specify data type: "head" or "teeth". Default: "head".')
    parser.add_argument('--target_genus', type=str, default=None, help='Target genus for "species" mode.')
    # --- 模型与分辨率控制 ---
    parser.add_argument('--model_name', type=str, default='efficientnet_b7', choices=list(MODEL_RESOLUTIONS.keys()), help='Name of the model architecture to use.')
    parser.add_argument('--input_size', type=int, default=None, help='Image input resolution. If not set, it will be automatically inferred from the model name.')
    # --- 训练控制 ---
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading.')
    parser.add_argument('--batch_size', type=int, default=4,help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--device', type=str, default=DEVICE, help='Device to use for training (e.g., "cuda", "cuda:0", "cuda:1", "cpu").')
    args = parser.parse_args()

    # 2. 动态配置与智能推断
    # 智能推断输入分辨率
    if args.input_size is None:
        input_size = MODEL_RESOLUTIONS.get(args.model_name)
        print(f"提示: 未指定 --input_size, 根据模型 '{args.model_name}' 自动设置为 {input_size}x{input_size}。")
    else:
        input_size = args.input_size
        print(f"提示: 使用用户指定的分辨率 {input_size}x{input_size}。")

    # 配置路径
    DATASET_ROOT = f'./datasets/{args.data_type}/2genus_4species'
    
    if args.mode == 'genus':
        print(f">> 模式: 属 (Genus) | 数据: {args.data_type} | 模型: {args.model_name}")
        if args.target_genus: print("警告: 'genus' 模式下 --target_genus 参数将被忽略。")
        SAVE_DIR = f'./weights/genus/{args.data_type}'
        os.makedirs(SAVE_DIR, exist_ok=True)
        train_data, test_data = load_genus_dataset(DATASET_ROOT, input_size)

    elif args.mode == 'species':
        if not args.target_genus: parser.error("--mode 'species' requires --target_genus.")
        print(f">> 模式: 种 (Species) | 数据: {args.data_type} | 属: {args.target_genus} | 模型: {args.model_name}")
        SAVE_DIR = f'./weights/species/{args.target_genus}/{args.data_type}'
        os.makedirs(SAVE_DIR, exist_ok=True)
        train_data, test_data = load_species_dataset(DATASET_ROOT, args.target_genus, input_size)
    
    # 3. 加载数据
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # ... (类别信息打印部分保持不变) ...
    print("=" * 40)
    print(f"类别数量: {len(train_data.classes)}")
    print(f"类别映射: {train_data.class_to_idx}")
    print(f"训练集总样本数: {len(train_data)}")
    print(f"测试集总样本数: {len(test_data)}")
    print("=" * 40)

    # 4. 创建模型 (使用新的参数)
    model = build_model(args.model_name, num_classes=len(train_data.classes), use_pretrained=True).to(DEVICE)

    # 5. 定义优化器、损失、调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.8)

    # 6. TensorBoard
    writer = tensorboard.SummaryWriter(os.path.join(SAVE_DIR, 'logs', args.model_name)) # 在log路径中加入模型名

    # 7. 训练
    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler,
                device=DEVICE, num_epochs=args.epochs, save_dir=SAVE_DIR,
                class_names=train_data.classes, patience=PATIENCE, writer=writer)

    writer.close()
    print("训练完成！")