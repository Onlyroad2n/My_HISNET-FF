# evaluate.py
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# 导入新的预测器
from utils.predictor_utils import HierarchicalPredictor

def run_evaluation(test_dir, data_type, model_name):
    """主评估函数。"""
    # 1. 初始化预测器
    try:
        predictor = HierarchicalPredictor(model_name=model_name, data_type=data_type)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保你已经为该 data_type 成功训练了属分类器。")
        return

    # 2. 查找所有测试图片并收集标签
    print(f"正在从 '{test_dir}' 递归搜索图片...")
    test_path = Path(test_dir)
    image_paths = []
    # 定义一个更全面的后缀列表
    supported_extensions = ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG']
    
    for ext in supported_extensions:
        image_paths.extend(test_path.rglob(ext))

    if not image_paths:
        print(f"错误: 在 '{test_dir}' 中没有找到任何支持的图片文件。")
        print(f"支持的格式为: {supported_extensions}")
        return
        
    print(f"成功找到 {len(image_paths)} 张图片。")
    if not image_paths:
        print(f"错误: 在 '{test_dir}' 中没有找到任何图片文件。")
        return

    # 从目录结构中获取所有唯一的真实类别
    ground_truth_classes = sorted(list(set(p.parent.name for p in image_paths)))
    print(f"在测试集中找到 {len(ground_truth_classes)} 个类别。")
    print("-" * 30)

    y_true, y_pred = [], []
    # 3. 遍历图片并进行预测
    for image_path in tqdm(image_paths, desc="Evaluating"):
        true_label = image_path.parent.name
        results = predictor.predict(str(image_path))
        pred_label = results['final_prediction']
        
        y_true.append(true_label)
        y_pred.append(pred_label)

    # 4. 计算并打印评估报告
    print("\n" + "="*20 + " 评估结果 " + "="*20)
    
    # 总准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # 分类报告 (精确率, 召回率, F1)
    # 使用 all_classes 确保报告中包含所有类别，即使某个类别在预测中从未出现
    all_classes = sorted(list(set(y_true + y_pred)))
    report = classification_report(y_true, y_pred, labels=all_classes, zero_division=0)
    print("\nClassification Report:\n")
    print(report)

    # 混淆矩阵
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    cm_df = pd.DataFrame(cm, index=all_classes, columns=all_classes)
    print(cm_df)
    print("="*55)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a hierarchical classification model.')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to the root directory of the test set.')
    parser.add_argument('--data_type', type=str, default='head', choices=['head', 'teeth'], help='Data type to evaluate.')
    parser.add_argument('--model_name', type=str, default='efficientnet_b7', help='Name of the model architecture used for training.')
    
    args = parser.parse_args()

    run_evaluation(args.test_dir, args.data_type, args.model_name)