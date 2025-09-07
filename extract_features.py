import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import DATASET_ROOT, WEIGHTS_ROOT, FEATURES_ROOT, DEFAULT_WEIGHT_FILE
from utils.data_utils import load_genus_dataset, load_species_dataset
from utils.model_utils import build_model


def extract_features(model: nn.Module, data_loader: DataLoader, device: str) -> np.ndarray:
    """Extracts features from a dataset using the provided model."""
    model.eval()
    all_features = []
    with torch.no_grad():
        split_name = "train" if "train" in data_loader.dataset.root.lower() else "test"
        for images, _ in tqdm(data_loader, desc=f"Extracting features from '{split_name}' set"):
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu().numpy())
    return np.concatenate(all_features, axis=0)


def process_and_save_split(model: nn.Module, dataset: Dataset, split_name: str, 
                           target_dir: str, batch_size: int, device: str):
    """Processes a dataset split to extract and save features and labels."""
    print("-" * 30)
    print(f"Processing '{split_name}' dataset with {len(dataset)} images...")
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    extracted_features = extract_features(model, data_loader, device)
    labels = np.array(dataset.targets)

    feature_save_path = os.path.join(target_dir, f"{split_name}_features.npy")
    label_save_path = os.path.join(target_dir, f"{split_name}_labels.npy")

    np.save(feature_save_path, extracted_features)
    np.save(label_save_path, labels)

    print(f"'{split_name}' features saved to: {feature_save_path}")
    print(f"Labels saved to: {label_save_path}")


def main(args):
    # 1. Configure output directory for features
    if args.mode == 'genus':
        target_output_dir = os.path.join(FEATURES_ROOT, 'genus', args.data_type)
    elif args.mode == 'species':
        if not args.target_genus:
            raise ValueError("--mode 'species' requires the --target_genus argument.")
        target_output_dir = os.path.join(FEATURES_ROOT, 'species', args.target_genus, args.data_type)
    
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"Features will be saved to: {target_output_dir}")

    # 2. Load dataset
    dataset_path = os.path.join(DATASET_ROOT, args.data_type, args.dataset_name)
    print(f"Loading dataset from: {dataset_path}")

    if args.mode == 'genus':
        train_data, test_data = load_genus_dataset(dataset_path, args.input_size)
    elif args.mode == 'species':
        train_data, test_data = load_species_dataset(dataset_path, args.target_genus, args.input_size)

    # 3. Determine the model weights path
    if args.weights_path:
        weights_path = args.weights_path
    else:
        # Construct the default weights path, e.g., /data/weights/pretrain/genus/teeth/best_network.pth
        if args.mode == "genus":
            weights_path = os.path.join(WEIGHTS_ROOT, "pretrain", "genus", args.data_type, DEFAULT_WEIGHT_FILE)
        elif args.mode == "species":
            weights_path = os.path.join(WEIGHTS_ROOT, "pretrain", "species", args.target_genus, args.data_type, DEFAULT_WEIGHT_FILE)
    print(f"Using weights from: {weights_path}")

    # 4. Load the model and prepare it for feature extraction
    num_classes = len(train_data.classes)
    model = build_model(args.model_name, num_classes, use_pretrained=False).to(args.device)
    model.load_state_dict(torch.load(weights_path, map_location=args.device))

    # Remove the final classification layer to get feature embeddings
    if args.model_name.startswith('efficientnet'):
        model.classifier[-1] = nn.Identity()
    elif 'resnet' in args.model_name:
        model.fc = nn.Identity()
    else:
        raise NotImplementedError(f"Feature extraction logic not implemented for {args.model_name}")

    # 5. Extract and save features for train and test splits
    process_and_save_split(model, train_data, 'train', target_output_dir, args.batch_size, args.device)
    process_and_save_split(model, test_data, 'test', target_output_dir, args.batch_size, args.device)

    print("\n" + "="*20, "Extraction Complete", "="*20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract and save features from a trained image classification model.")
    
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset directory (e.g., 2genus_4species).')
    parser.add_argument('--mode', type=str, required=True, choices=['genus', 'species'], help='Classification level.')
    parser.add_argument('--data_type', type=str, required=True, choices=['cranium', 'teeth'], help='Data category.')
    parser.add_argument('--target_genus', type=str, default=None, help='Target genus, required for species mode.')
    
    parser.add_argument('--model_name', type=str, default='efficientnet_b7', help='Name of the model architecture.')
    parser.add_argument('--input_size', type=int, default=600, help='Input image resolution.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for feature extraction.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., "cuda:0" or "cpu").')
    
    parser.add_argument('--weights_path', type=str, default=None,
                        help="Path to the trained .pth model weights. If not specified, a default path will be constructed.")
    
    args = parser.parse_args()
    main(args)