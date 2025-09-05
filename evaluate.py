import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import os

# 导入两个预测器引擎
from utils.predictor_utils import HierarchicalPredictor, FusionHierarchicalPredictor

def find_image_pairs(head_root, teeth_root):
    """
    (修改后) 辅助函数：在两个目录中寻找文件名相同的成对图片。
    这个版本现在不区分扩展名大小写，且支持更多格式。
    """
    head_root, teeth_root = Path(head_root), Path(teeth_root)
    image_pairs = []
    
    # 定义所有要搜索的图片格式（包括大小写）
    image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    
    # 以head目录为基准，遍历所有可能的图片格式
    head_images = []
    for ext in image_extensions:
        head_images.extend(head_root.rglob(ext))

    if not head_images:
        print("警告: 在头骨测试目录中未找到任何支持的图片文件。")
        return []

    for head_path in tqdm(head_images, desc="正在匹配成对图片"):
        # 构建对应的teeth图片路径
        relative_path = head_path.relative_to(head_root)
        teeth_path = teeth_root / relative_path
        
        # 只需要检查对应的文件是否存在即可，无需关心其扩展名大小写
        if teeth_path.exists():
            image_pairs.append((str(head_path), str(teeth_path)))
            
    return image_pairs

def run_evaluation(args):
    """主评估函数，支持双模式。"""
    
    y_true, y_pred = [], []

    if args.fusion:
        # ==================== 融合评估模式 ====================
        print("\n>> 模式: 融合层级评估 (Fusion Hierarchical Evaluation)")
        if not args.head_test_dir or not args.teeth_test_dir:
            parser.error("--fusion 模式需要同时提供 --head_test_dir 和 --teeth_test_dir。")

        try:
            predictor = FusionHierarchicalPredictor(model_name=args.model_name)
        except FileNotFoundError as e:
            print(f"错误: {e}\n请确保已为所有层级成功训练了融合模型。")
            return
        
        image_pairs = find_image_pairs(args.head_test_dir, args.teeth_test_dir)
        if not image_pairs:
            print("错误: 未找到任何成对的头骨/牙齿图片。")
            return

        for head_path, teeth_path in tqdm(image_pairs, desc="正在评估融合模型"):
            true_label = Path(head_path).parent.name
            results = predictor.predict(head_path, teeth_path)
            y_true.append(true_label)
            y_pred.append(results['final_prediction'])

    else:
        # ==================== 单模态评估模式 (原有逻辑) ====================
        print("\n>> 模式: 单模态层级评估 (Single-Modal Hierarchical Evaluation)")
        if not args.test_dir:
            parser.error("单模态模式需要提供 --test_dir。")
        
        try:
            predictor = HierarchicalPredictor(model_name=args.model_name, data_type=args.data_type)
        except FileNotFoundError as e:
            print(f"错误: {e}\n请确保已为该 data_type 成功训练了单模态模型。")
            return
        
        # (同样在此处应用健壮的图片搜索逻辑)
        test_path = Path(args.test_dir)
        image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(test_path.rglob(ext))

        if not image_paths:
            print(f"错误: 在 '{args.test_dir}' 中没有找到任何图片文件。")
            return
        
        for image_path in tqdm(image_paths, desc=f"正在评估 {args.data_type} 模型"):
            true_label = image_path.parent.name
            results = predictor.predict(str(image_path))
            y_true.append(true_label)
            y_pred.append(results['final_prediction'])

    # ==================== 统一的评估报告生成 (完全复用) ====================
    if not y_true:
        print("没有可供评估的样本，程序退出。")
        return

    print("\n" + "="*20 + " 评估结果 " + "="*20)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    all_classes = sorted(list(set(y_true + y_pred)))
    report = classification_report(y_true, y_pred, labels=all_classes, zero_division=0)
    print("\nClassification Report:\n")
    print(report)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    cm_df = pd.DataFrame(cm, index=all_classes, columns=all_classes)
    print(cm_df)
    print("="*55)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='统一评估脚本，支持单模态和融合模式。')
    # ... 命令行参数部分保持不变 ...
    parser.add_argument('--fusion', action='store_true', help='启用融合评估模式。如果启用，则必须提供 head 和 teeth 的测试集路径。')
    parser.add_argument('--test_dir', type=str, help='(单模态) 测试集根目录。')
    parser.add_argument('--data_type', type=str, default='head', choices=['head', 'teeth'], help='(单模态) 要评估的数据类型。')
    parser.add_argument('--head_test_dir', type=str, help='(融合模式) 头骨测试集根目录。')
    parser.add_argument('--teeth_test_dir', type=str, help='(融合模式) 牙齿测试集根目录。')
    parser.add_argument('--model_name', type=str, default='efficientnet_b7', help='用于特征提取的骨干网络名称。')
    
    args = parser.parse_args()
    run_evaluation(args)