import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import argparse
import numpy as np
import shutil
from utils.fusion_utils import fuse_features, FeatureDataset
from utils.fusion_models import SimpleMLP
from utils.train_utils import EarlyStopping
from config import FEATURES_ROOT, WEIGHTS_ROOT

def main(args):
    if args.mode == 'genus':
        teeth_dir = os.path.join(FEATURES_ROOT, 'genus', 'teeth')
        cranium_dir  = os.path.join(FEATURES_ROOT, 'genus', 'cranium')
        save_weight_dir = os.path.join(WEIGHTS_ROOT, 'fusion', 'genus')
        source_class_file = os.path.join(WEIGHTS_ROOT, 'pretrain', 'genus', 'cranium', 'classes.txt')

    elif args.mode == 'species':
        if not args.target_genus:
            parser.error("--mode 'species' 需要指定 --target_genus。")
        teeth_dir = os.path.join(FEATURES_ROOT, 'species', args.target_genus, 'teeth')
        cranium_dir  = os.path.join(FEATURES_ROOT, 'species', args.target_genus, 'cranium')
        save_weight_dir = os.path.join(WEIGHTS_ROOT, 'fusion', 'species', args.target_genus)
        source_class_file = os.path.join(WEIGHTS_ROOT, 'pretrain', 'species', args.target_genus, 'cranium', 'classes.txt')

    
    print(f"模式: '{args.mode}', 目标属: '{args.target_genus or 'N/A'}'")
    print(f"读取 Cranium 特征自: {cranium_dir}")
    print(f"读取 Teeth 特征自: {teeth_dir}")
    print(f"融合模型将保存至: {save_weight_dir}")
    os.makedirs(save_weight_dir, exist_ok=True)

    try:
        dest_class_file = os.path.join(save_weight_dir, 'classes.txt')
        shutil.copyfile(source_class_file, dest_class_file)
        print(f"类别文件已成功保存至: {dest_class_file}")
    except FileNotFoundError:
        print(f"\n错误: 未能找到源类别文件: {source_class_file}")
        print("请确保您已经为 'cranium' 数据类型成功运行了 train.py。")
        return

    try:
        x_train, y_train, x_test, y_test = fuse_features(
            teeth_dir, cranium_dir,
            teeth_ratio=args.teeth_ratio,
            cranium_ratio=args.cranium_ratio,
            save_scaler_dir=save_weight_dir
        )
    except FileNotFoundError as e:
        print(f"\n错误: 特征文件未找到！ {e}")
        print("请确保您已经为 'cranium' 和 'teeth' 两种数据类型都成功运行了 extract_features.py。")
        return
        
    print(f"融合后训练特征维度: {x_train.shape}")
    print(f"融合后测试特征维度: {x_test.shape}")
    num_classes = len(np.unique(y_train))
    train_dataset = FeatureDataset(x_train, y_train)
    test_dataset  = FeatureDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    input_dim = train_dataset.data.size(1)
    model = SimpleMLP(input_dim, num_classes, hidden_dim=args.hidden_dim, dropout=args.dropout).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (outputs.argmax(1) == y).sum().item()
        train_acc = correct / len(train_loader.dataset)
        avg_train_loss = total_loss / len(train_loader.dataset)
        model.eval()
        correct_test = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(args.device), y.to(args.device)
                outputs = model(x)
                correct_test += (outputs.argmax(1) == y).sum().item()
        test_acc = correct_test / len(test_loader.dataset)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Acc: {test_acc:.4f}")
        stopper(test_acc, epoch + 1, model, save_weight_dir)
        if stopper.early_stop:
            print("验证集准确率已停止提升，提前停止训练。")
            break
    
    best_epoch = stopper.best_epoch
    best_acc = stopper.best_score
    final_model_name = 'best_fusion_model.pth'
    temp_model_path = os.path.join(save_weight_dir, 'best_network.pth')
    final_model_path = os.path.join(save_weight_dir, final_model_name)
    if os.path.exists(temp_model_path):
        os.rename(temp_model_path, final_model_path)

    summary_file_path = os.path.join(save_weight_dir, 'best_acc.txt')
    try:
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Best Validation Accuracy: {best_acc:.4f}\n")
            f.write(f"Best Epoch: {best_epoch}\n")
    except Exception as e:
        print(f"错误：无法将总结信息写入文件 {summary_file_path}。原因: {e}")

    summary_message = (
        f"{'='*50}\n"
        f"训练完毕!\n"
        f"最优模型已保存至: {final_model_path}\n"
        f"总结信息已写入: {summary_file_path}\n"
        f"{'-'*50}\n"
        f"最优轮次 (Best Epoch): {best_epoch}\n"
        f"最高验证准确率 (Best Val Acc): {best_acc:.4f}\n"
        f"{'='*50}"
    )
    print(summary_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练一个MLP模型来融合深度特征。")
    parser.add_argument('--mode', type=str, required=True, choices=['genus', 'species'], help='设置模式: "genus" 或 "species".')
    parser.add_argument('--target_genus', type=str, default=None, help='当模式为 "species" 时，需要指定目标属。')
    parser.add_argument('--teeth_ratio', type=float, default=1.0, help='牙齿特征的归一化最大值。')
    parser.add_argument('--cranium_ratio', type=float, default=1.0, help='头骨特征的归一化最大值。')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数。')
    parser.add_argument('--batch_size', type=int, default=64, help='批处理大小。')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率。')
    parser.add_argument('--patience', type=int, default=20, help='早停的耐心轮数。')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='MLP隐藏层维度。')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout比率。')
    parser.add_argument('--device', type=str, default='cuda:0', help='计算设备。')
    args = parser.parse_args()
    main(args)