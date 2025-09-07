import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from config import (DATASET_ROOT as CFG_DATASET_ROOT,
                    FEATURES_ROOT as CFG_FEATURES_ROOT,
                    WEIGHTS_ROOT as CFG_WEIGHTS_ROOT)
from utils.predictor_utils import (FusionHierarchicalPredictor,
                                     HierarchicalPredictor)


def find_image_pairs(cranium_root: str, teeth_root: str):
    """Finds pairs of images with the same relative path in two root directories."""
    cranium_root, teeth_root = Path(cranium_root), Path(teeth_root)
    image_pairs = []
    image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']

    cranium_images = [p for ext in image_extensions for p in cranium_root.rglob(ext)]

    if not cranium_images:
        print("Warning: No supported image files found in the cranium test directory.")
        return []

    for cranium_path in tqdm(cranium_images, desc="Matching image pairs"):
        relative_path = cranium_path.relative_to(cranium_root)
        teeth_path = teeth_root / relative_path
        if teeth_path.exists():
            image_pairs.append((str(cranium_path), str(teeth_path)))

    return image_pairs


def run_evaluation(args):
    """Main evaluation function, supporting both single-modality and fusion modes."""
    y_true, y_pred = [], []

    # 1. Set up configuration paths, allowing overrides from command line
    dataset_root = args.dataset_root or CFG_DATASET_ROOT
    weights_root = args.weights_root or CFG_WEIGHTS_ROOT
    features_root = args.features_root or CFG_FEATURES_ROOT

    # 2. Execute evaluation based on the selected mode (fusion or single)
    if args.fusion:
        print("\n>> Mode: Fusion-based Hierarchical Evaluation")
        if not args.dataset_name:
            parser.error("--fusion mode requires the --dataset_name argument.")

        cranium_test_dir = os.path.join(dataset_root, 'cranium', args.dataset_name, 'test')
        teeth_test_dir = os.path.join(dataset_root, 'teeth', args.dataset_name, 'test')

        if not os.path.isdir(cranium_test_dir) or not os.path.isdir(teeth_test_dir):
            print(f"Error: Test set directory not found.\n Cranium path: {cranium_test_dir}\n Teeth path:   {teeth_test_dir}")
            return

        try:
            predictor = FusionHierarchicalPredictor(model_name=args.model_name,
                                                    weights_root=weights_root,
                                                    input_size=args.input_size)
        except FileNotFoundError as e:
            print(f"Error initializing predictor: {e}\nPlease ensure all required models and scalers are trained and available.")
            return

        image_pairs = find_image_pairs(cranium_test_dir, teeth_test_dir)
        if not image_pairs:
            print("Error: No matching image pairs found for cranium/teeth evaluation.")
            return

        for cranium_path, teeth_path in tqdm(image_pairs, desc="Evaluating fusion model"):
            true_label = Path(cranium_path).parent.name
            results = predictor.predict(cranium_path, teeth_path)
            y_true.append(true_label)
            y_pred.append(results['final_prediction'])

    else: # Single-modality mode
        print("\n>> Mode: Single-Modality Hierarchical Evaluation")
        if not args.dataset_name or not args.data_type:
            parser.error("Single-modality mode requires both --dataset_name and --data_type.")

        test_dir = os.path.join(dataset_root, args.data_type, args.dataset_name, 'test')
        if not os.path.isdir(test_dir):
            print(f"Error: Test set path does not exist: {test_dir}")
            return

        try:
            predictor = HierarchicalPredictor(model_name=args.model_name,
                                              data_type=args.data_type,
                                              weights_root=weights_root,
                                              input_size=args.input_size)
        except FileNotFoundError as e:
            print(f"Error initializing predictor: {e}\nPlease ensure the required models are trained.")
            return

        image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        image_paths = [p for ext in image_extensions for p in Path(test_dir).rglob(ext)]

        if not image_paths:
            print(f"Error: No valid images found in {test_dir}.")
            return

        for image_path in tqdm(image_paths, desc=f"Evaluating {args.data_type} model"):
            true_label = image_path.parent.name
            results = predictor.predict(str(image_path))
            y_true.append(true_label)
            y_pred.append(results['final_prediction'])

    # 3. Generate and display evaluation results
    if not y_true:
        print("Evaluation could not be completed as no samples were processed.")
        return

    print("\n" + "="*20 + " Evaluation Results " + "="*20)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    all_classes = sorted(list(set(y_true + y_pred)))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=all_classes, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    cm_df = pd.DataFrame(cm, index=all_classes, columns=all_classes)
    print(cm_df)
    print("="*58)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Unified evaluation script for single-modality and fusion-based hierarchical models.'
    )
    parser.add_argument('--fusion', action='store_true', 
                        help='Enable fusion-based evaluation mode using both cranium and teeth images.')
    parser.add_argument('--dataset_name', type=str, required=True, 
                        help='Name of the dataset directory to test on (e.g., 2genus_4species).')
    parser.add_argument('--data_type', type=str, choices=['cranium', 'teeth'], 
                        help='Data type, required for single-modality mode.')
    parser.add_argument('--model_name', type=str, default='efficientnet_b7', 
                        help='Model architecture name.')
    parser.add_argument('--input_size', type=int, default=600, 
                        help='Input image size/resolution.')
    parser.add_argument('--dataset_root', type=str, default=None, 
                        help='(Optional) Override DATASET_ROOT from config.py.')
    parser.add_argument('--weights_root', type=str, default=None, 
                        help='(Optional) Override WEIGHTS_ROOT from config.py.')
    parser.add_argument('--features_root', type=str, default=None, 
                        help='(Optional) Override FEATURES_ROOT from config.py for fusion mode.')

    args = parser.parse_args()
    run_evaluation(args)