import os
import argparse
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils import tensorboard

from config import DATASET_ROOT, WEIGHTS_ROOT
from utils.data_utils import load_genus_dataset, load_species_dataset
from utils.model_utils import build_model
from utils.train_utils import train_model


# Standard input resolutions for common EfficientNet models
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
    
    # --- Mode and Data Control ---
    parser.add_argument('--mode', type=str, required=True, choices=['genus', 'species'],
                        help='Set training mode: "genus" or "species" level classification.')
    parser.add_argument('--data_type', type=str, default='cranium', choices=['cranium', 'teeth'],
                        help='Specify the type of data to use.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Specify the dataset folder name under DATASET_ROOT.')
    parser.add_argument('--target_genus', type=str, default=None,
                        help='Required for "species" mode. Specifies the target genus to train on.')

    # --- Model and Resolution Control ---
    parser.add_argument('--model_name', type=str, default='efficientnet_b7', choices=list(MODEL_RESOLUTIONS.keys()),
                        help='The name of the model architecture to use.')
    parser.add_argument('--input_size', type=int, default=None,
                        help='Input image size. If None, it is inferred from the model name.')

    # --- Compute Resource and Data Loading ---
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of samples per batch.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device to run the training on, e.g., "cuda:0" or "cpu".')

    # --- Training Hyperparameters ---
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Total number of training epochs.')
    parser.add_argument('--step_size', type=int, default=10,
                        help='Step size for the learning rate scheduler.')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for early stopping.')
    
    args = parser.parse_args()

    # Determine input size
    if args.input_size is None:
        input_size = MODEL_RESOLUTIONS.get(args.model_name)
        print(f"Info: --input_size not specified. Automatically set to {input_size}x{input_size} based on model '{args.model_name}'.")
    else:
        input_size = args.input_size
        print(f"Info: Using user-specified resolution {input_size}x{input_size}.")

    # Configure paths
    dataset_path = os.path.join(DATASET_ROOT, args.data_type, args.dataset_name)
    save_root = os.path.join(WEIGHTS_ROOT, 'pretrain')

    # Load data based on mode
    if args.mode == 'genus':
        print(f">> Mode: Genus | Data: {args.data_type}/{args.dataset_name} | Model: {args.model_name}")
        if args.target_genus: 
            print("Warning: --target_genus will be ignored in 'genus' mode.")
        SAVE_DIR = os.path.join(save_root, 'genus', args.data_type)
        train_data, test_data = load_genus_dataset(dataset_path, input_size)

    elif args.mode == 'species':
        if not args.target_genus: 
            parser.error("--mode 'species' requires --target_genus.")
        print(f">> Mode: Species | Data: {args.data_type}/{args.dataset_name} | Genus: {args.target_genus} | Model: {args.model_name}")
        SAVE_DIR = os.path.join(save_root, 'species', args.target_genus, args.data_type)
        train_data, test_data = load_species_dataset(dataset_path, args.target_genus, input_size)

    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Print dataset summary
    print("=" * 40)
    print(f"Number of classes: {len(train_data.classes)}")
    print(f"Class to index mapping: {train_data.class_to_idx}")
    print(f"Total training samples: {len(train_data)}")
    print(f"Total testing samples: {len(test_data)}")
    print("=" * 40)

    # Save class names to a file
    classes_file_path = os.path.join(SAVE_DIR, 'classes.txt')
    print(f"Saving class information to: {classes_file_path}")
    with open(classes_file_path, 'w', encoding='utf-8') as f:
        for c in train_data.classes:
            f.write(f"{c}\n")

    # Setup device
    DEVICE = args.device if torch.cuda.is_available() else 'cpu'
    
    # Initialize model
    model = build_model(args.model_name, num_classes=len(train_data.classes), use_pretrained=True).to(DEVICE)

    # Optimizer, Loss Function, and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.8)

    # TensorBoard setup
    timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(SAVE_DIR, 'logs', args.model_name, timestamp)
    writer = tensorboard.SummaryWriter(log_dir)

    # Start training
    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler,
                device=DEVICE, num_epochs=args.epochs, save_dir=SAVE_DIR,
                patience=args.patience, writer=writer)

    writer.close()
    print("Training complete!")