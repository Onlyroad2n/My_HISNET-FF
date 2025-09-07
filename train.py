import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.utils import tensorboard
from datetime import datetime
import argparse
from config import DATASET_ROOT, WEIGHTS_ROOT   
from utils.data_utils import load_genus_dataset, load_species_dataset
from utils.model_utils import build_model
from utils.train_utils import train_model
  

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
    parser = argparse.ArgumentParser(description='Unified and configurable training script for image classification.')
    # --- 模式与数据控制 ---
    parser.add_argument('--mode', type=str, required=True, choices=['genus', 'species'])
    parser.add_argument('--data_type', type=str, default='cranium', choices=['cranium', 'teeth'])
    parser.add_argument('--dataset_name', type=str, required=True, help='Specify dataset folder name under DATASET_ROOT')
    parser.add_argument('--target_genus', type=str, default=None)
    # --- 模型与分辨率控制 ---
    parser.add_argument('--model_name', type=str, default='efficientnet_b7', choices=list(MODEL_RESOLUTIONS.keys()))
    parser.add_argument('--input_size', type=int, default=None)
    # --- 计算资源与数据加载 ---
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    # --- 训练控制 ---
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()


    if args.input_size is None:
        input_size = MODEL_RESOLUTIONS.get(args.model_name)
        print(f"提示: 未指定 --input_size, 根据模型 '{args.model_name}' 自动设置为 {input_size}x{input_size}。")
    else:
        input_size = args.input_size
        print(f"提示: 使用用户指定的分辨率 {input_size}x{input_size}。")

    dataset_path = os.path.join(DATASET_ROOT, args.data_type, args.dataset_name)
    save_root = os.path.join(WEIGHTS_ROOT, 'pretrain')

    if args.mode == 'genus':
        print(f">> 模式: 属 (Genus) | 数据: {args.data_type}/{args.dataset_name} | 模型: {args.model_name}")
        if args.target_genus: 
            print("警告: 'genus' 模式下 --target_genus 参数将被忽略。")
        SAVE_DIR = os.path.join(save_root, 'genus', args.data_type)
        os.makedirs(SAVE_DIR, exist_ok=True)
        train_data, test_data = load_genus_dataset(dataset_path, input_size)

    elif args.mode == 'species':
        if not args.target_genus: 
            parser.error("--mode 'species' requires --target_genus.")
        print(f">> 模式: 种 (Species) | 数据: {args.data_type}/{args.dataset_name} | 属: {args.target_genus} | 模型: {args.model_name}")
        SAVE_DIR = os.path.join(save_root, 'species', args.target_genus, args.data_type)
        os.makedirs(SAVE_DIR, exist_ok=True)
        train_data, test_data = load_species_dataset(dataset_path, args.target_genus, input_size)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("=" * 40)
    print(f"类别数量: {len(train_data.classes)}")
    print(f"类别映射: {train_data.class_to_idx}")
    print(f"训练集总样本数: {len(train_data)}")
    print(f"测试集总样本数: {len(test_data)}")
    print("=" * 40)

    print(f"保存类别信息到: {os.path.join(SAVE_DIR, 'classes.txt')}")
    with open(os.path.join(SAVE_DIR, 'classes.txt'), 'w', encoding='utf-8') as f:
        for c in train_data.classes:
            f.write(f"{c}\n")

    DEVICE = args.device if torch.cuda.is_available() else 'cpu'
    # 创建模型
    model = build_model(args.model_name, num_classes=len(train_data.classes), use_pretrained=True).to(DEVICE)

    # 优化器、损失、调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.8)

    # TensorBoard
    timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(SAVE_DIR, 'logs', args.model_name, timestamp)
    writer = tensorboard.SummaryWriter(log_dir)

    # 训练
    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler,
                device=DEVICE, num_epochs=args.epochs, save_dir=SAVE_DIR,
                patience=args.patience, writer=writer)

    writer.close()
    print("训练完成！")
