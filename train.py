# train.py
import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.utils import tensorboard
import sys
import argparse  # 导入 argparse 模块

# 将 utils 目录添加到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_utils import load_genus_dataset, load_species_dataset
from utils.model_utils import build_model
from utils.train_utils import train_model

# ======= 参数区 (固定参数) =======
# 可变参数已移至命令行
INPUT_SIZE = 600
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
STEP_SIZE = 10
DEVICE = 'cuda:0'
MODEL_NAME = 'efficientnet_b7'
PATIENCE = 50
# ======================

if __name__ == "__main__":
    # 1. 设置命令行参数解析器
    parser = argparse.ArgumentParser(description='Unified training script for Genus and Species classification.')
    parser.add_argument('--mode', type=str, required=True, choices=['genus', 'species'],
                        help='Set the training mode: "genus" or "species".')
    parser.add_argument('--data_type', type=str, default='head', choices=['head', 'teeth'],
                        help='Specify the data type: "head" or "teeth". Default is "head".')
    parser.add_argument('--target_genus', type=str, default=None,
                        help='Specify the target genus to train for. Required and only used when --mode is "species".')
    args = parser.parse_args()

    # 2. 根据命令行参数动态配置路径
    DATASET_ROOT = f'./datasets/{args.data_type}/2genus_4species'
    
    if args.mode == 'genus':
        print(f">> 模式: 属 (Genus) 分类 | 数据类型: {args.data_type}")
        if args.target_genus:
            print("警告: 在 'genus' 模式下，--target_genus 参数将被忽略。")

        SAVE_DIR = f'./weights/genus/{args.data_type}'
        os.makedirs(SAVE_DIR, exist_ok=True)
        train_data, test_data = load_genus_dataset(DATASET_ROOT, INPUT_SIZE)
        print("=" * 40)
        print("正在加载属级别数据集...")

    elif args.mode == 'species':
        if not args.target_genus:
            parser.error("--mode 'species' requires --target_genus to be specified.")

        print(f">> 模式: 种 (Species) 分类 | 数据类型: {args.data_type} | 目标属: {args.target_genus}")
        SAVE_DIR = f'./weights/species/{args.target_genus}/{args.data_type}'
        os.makedirs(SAVE_DIR, exist_ok=True)
        train_data, test_data = load_species_dataset(DATASET_ROOT, args.target_genus, INPUT_SIZE)
        print("=" * 40)
        print(f"正在加载 [{args.target_genus}] 属下的种级别数据集...")
    
    # 3. 加载数据
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    # === 类别信息打印 ===
    print(f"类别数量: {len(train_data.classes)}")
    print(f"类别映射: {train_data.class_to_idx}")
    for cls_name, idx in train_data.class_to_idx.items():
        count = sum(1 for t in train_data.targets if t == idx)
        print(f"  {cls_name}: {count} 张图片")
    print(f"训练集总样本数: {len(train_data)}")
    print(f"测试集总样本数: {len(test_data)}")
    print("=" * 40)

    # 4. 创建模型
    model = build_model(MODEL_NAME, num_classes=len(train_data.classes), pretrained=True).to(DEVICE)

    # 5. 定义优化器、损失、调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.8)

    # 6. TensorBoard
    writer = tensorboard.SummaryWriter(os.path.join(SAVE_DIR, 'logs'))

    # 7. 训练
    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler,
                device=DEVICE, num_epochs=NUM_EPOCHS, save_dir=SAVE_DIR,
                class_names=train_data.classes, patience=PATIENCE, writer=writer)

    writer.close()
    print("训练完成！")