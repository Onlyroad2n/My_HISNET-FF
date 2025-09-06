import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import os

# 导入两个预测器引擎
from utils.predictor_utils import HierarchicalPredictor, FusionHierarchicalPredictor
from config import DATASET_ROOT as CFG_DATASET_ROOT, WEIGHTS_ROOT as CFG_WEIGHTS_ROOT, FEATURES_ROOT as CFG_FEATURES_ROOT

def find_image_pairs(head_root, teeth_root):
    """
    在两个目录中寻找文件名相同的成对图片
    """
    head_root, teeth_root = Path(head_root), Path(teeth_root)
    image_pairs = []
    image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']

    head_images = []
    for ext in image_extensions:
        head_images.extend(head_root.rglob(ext))

    if not head_images:
        print("警告: 在头骨测试目录中未找到任何支持的图片文件。")
        return []

    for head_path in tqdm(head_images, desc="正在匹配成对图片"):
        relative_path = head_path.relative_to(head_root)
        teeth_path = teeth_root / relative_path
        if teeth_path.exists():
            image_pairs.append((str(head_path), str(teeth_path)))

    return image_pairs

def run_evaluation(args):
    """主评估函数，支持单模态与融合模式"""
    y_true, y_pred = [], []

    dataset_root = args.dataset_root or CFG_DATASET_ROOT
    weights_root = args.weights_root or CFG_WEIGHTS_ROOT
    features_root = args.features_root or CFG_FEATURES_ROOT

    if args.fusion:
        print("\n>> 模式: 融合层级评估")
        if not args.dataset_name:
            parser.error("--fusion 模式需要提供 --dataset_name（测试集目录名）")

        head_test_dir = os.path.join(dataset_root, 'head', args.dataset_name, 'test')
        teeth_test_dir = os.path.join(dataset_root, 'teeth', args.dataset_name, 'test')

        if not os.path.exists(head_test_dir) or not os.path.exists(teeth_test_dir):
            print(f"错误: 未找到测试集目录\n Head: {head_test_dir}\n Teeth: {teeth_test_dir}")
            return

        try:
            predictor = FusionHierarchicalPredictor(model_name=args.model_name,
                                                    features_root=features_root,
                                                    weights_root=weights_root,
                                                    input_size=args.input_size)
        except FileNotFoundError as e:
            print(f"错误: {e}")
            return

        image_pairs = find_image_pairs(head_test_dir, teeth_test_dir)
        if not image_pairs:
            print("错误: 未找到成对的头骨/牙齿测试图片。")
            return

        for head_path, teeth_path in tqdm(image_pairs, desc="评估融合模型"):
            true_label = Path(head_path).parent.name
            results = predictor.predict(head_path, teeth_path)
            y_true.append(true_label)
            y_pred.append(results['final_prediction'])

    else:
        print("\n>> 模式: 单模态层级评估")
        if not args.dataset_name:
            parser.error("--单模态 模式需要提供 --dataset_name")
        if not args.data_type:
            parser.error("--单模态 模式需要提供 --data_type")

        test_dir = os.path.join(dataset_root, args.data_type, args.dataset_name, 'test')
        if not os.path.exists(test_dir):
            print(f"错误: 测试集路径不存在: {test_dir}")
            return

        try:
            predictor = HierarchicalPredictor(model_name=args.model_name,
                                              data_type=args.data_type,
                                              dataset_name=args.dataset_name,
                                              weights_root=weights_root,
                                              input_size=args.input_size)
        except FileNotFoundError as e:
            print(f"错误: {e}")
            return

        image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(Path(test_dir).rglob(ext))

        if not image_paths:
            print(f"错误: 在 {test_dir} 中未找到有效图片。")
            return

        for image_path in tqdm(image_paths, desc=f"评估 {args.data_type} 模型"):
            true_label = image_path.parent.name
            results = predictor.predict(str(image_path))
            y_true.append(true_label)
            y_pred.append(results['final_prediction'])

    # 生成评估报告
    if not y_true:
        print("没有可供评估的样本。")
        return

    print("\n" + "="*20 + " 评估结果 " + "="*20)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    all_classes = sorted(list(set(y_true + y_pred)))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=all_classes, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    print(pd.DataFrame(cm, index=all_classes, columns=all_classes))
    print("="*55)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='统一评估脚本，支持单模态和融合模式')
    parser.add_argument('--fusion', action='store_true', help='使用融合评估模式')
    parser.add_argument('--dataset_name', type=str, required=False, help='数据集目录名（如 2genus_4species）')
    parser.add_argument('--data_type', type=str, choices=['head', 'teeth'], help='单模态模式需要的数据类型')
    parser.add_argument('--model_name', type=str, default='efficientnet_b7', help='模型架构')
    parser.add_argument('--input_size', type=int, default=600, help='输入图像的大小')
    parser.add_argument('--dataset_root', type=str, default=None, help='(可选) 覆盖 config.py 中 DATASET_ROOT')
    parser.add_argument('--weights_root', type=str, default=None, help='(可选) 覆盖 config.py 中 WEIGHTS_ROOT')
    parser.add_argument('--features_root', type=str, default=None, help='(可选) 覆盖 config.py 中 FEATURES_ROOT')


    args = parser.parse_args()
    run_evaluation(args)
