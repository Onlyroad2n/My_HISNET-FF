import os
import argparse
import shutil
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import FEATURES_ROOT, WEIGHTS_ROOT
from utils.fusion_models import SimpleMLP
from utils.fusion_utils import fuse_features, FeatureDataset
from utils.train_utils import EarlyStopping


def main(args):
    # 1. Configure paths based on the selected mode
    if args.mode == 'genus':
        teeth_dir = os.path.join(FEATURES_ROOT, 'genus', 'teeth')
        cranium_dir = os.path.join(FEATURES_ROOT, 'genus', 'cranium')
        save_weight_dir = os.path.join(WEIGHTS_ROOT, 'fusion', 'genus')
        source_class_file = os.path.join(WEIGHTS_ROOT, 'pretrain', 'genus', 'cranium', 'classes.txt')
    elif args.mode == 'species':
        if not args.target_genus:
            parser.error("--mode 'species' requires the --target_genus argument.")
        teeth_dir = os.path.join(FEATURES_ROOT, 'species', args.target_genus, 'teeth')
        cranium_dir = os.path.join(FEATURES_ROOT, 'species', args.target_genus, 'cranium')
        save_weight_dir = os.path.join(WEIGHTS_ROOT, 'fusion', 'species', args.target_genus)
        source_class_file = os.path.join(WEIGHTS_ROOT, 'pretrain', 'species', args.target_genus, 'cranium', 'classes.txt')

    print(f"Mode: '{args.mode}', Target Genus: '{args.target_genus or 'N/A'}'")
    print(f"Reading Cranium features from: {cranium_dir}")
    print(f"Reading Teeth features from: {teeth_dir}")
    print(f"Fusion model will be saved to: {save_weight_dir}")
    os.makedirs(save_weight_dir, exist_ok=True)

    # 2. Copy the class file from the pretrain directory
    try:
        dest_class_file = os.path.join(save_weight_dir, 'classes.txt')
        shutil.copyfile(source_class_file, dest_class_file)
        print(f"Class file successfully copied to: {dest_class_file}")
    except FileNotFoundError:
        print(f"\nError: Source class file not found at: {source_class_file}")
        print("Please ensure you have successfully run train.py for the 'cranium' data type.")
        return

    # 3. Fuse features from both data types
    try:
        x_train, y_train, x_test, y_test = fuse_features(
            teeth_dir, cranium_dir,
            teeth_ratio=args.teeth_ratio,
            cranium_ratio=args.cranium_ratio,
            save_scaler_dir=save_weight_dir
        )
    except FileNotFoundError as e:
        print(f"\nError: Feature file not found! {e}")
        print("Please ensure you have run extract_features.py for both 'cranium' and 'teeth' data types.")
        return
        
    print(f"Fused training features shape: {x_train.shape}")
    print(f"Fused testing features shape: {x_test.shape}")

    # 4. Set up datasets, model, and training components
    train_dataset = FeatureDataset(x_train, y_train)
    test_dataset = FeatureDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    input_dim = train_dataset.data.size(1)
    num_classes = len(np.unique(y_train))
    model = SimpleMLP(input_dim, num_classes, hidden_dim=args.hidden_dim, dropout=args.dropout).to(args.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopping(patience=args.patience, verbose=True)

    # 5. Training loop
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
        
        avg_train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)
        
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
            print("Validation accuracy did not improve. Stopping training early.")
            break
    
    # 6. Finalize and save results
    best_epoch = stopper.best_epoch
    best_acc = stopper.best_score if stopper.best_score is not None else 0.0
    
    temp_model_path = os.path.join(save_weight_dir, 'best_network.pth')
    final_model_path = os.path.join(save_weight_dir, 'best_fusion_model.pth')
    if os.path.exists(temp_model_path):
        os.rename(temp_model_path, final_model_path)

    summary_file_path = os.path.join(save_weight_dir, 'training_summary.txt')
    try:
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Best Validation Accuracy: {best_acc:.4f}\n")
            f.write(f"Achieved at Epoch: {best_epoch}\n")
    except IOError as e:
        print(f"Error: Could not write summary to {summary_file_path}. Reason: {e}")

    summary_message = (
        f"\n{'='*50}\n"
        f"Training Complete!\n"
        f"Best model saved to: {final_model_path}\n"
        f"Summary written to: {summary_file_path}\n"
        f"{'-'*50}\n"
        f"Best Epoch: {best_epoch}\n"
        f"Best Validation Accuracy: {best_acc:.4f}\n"
        f"{'='*50}"
    )
    print(summary_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP model to fuse deep learning features.")
    
    parser.add_argument('--mode', type=str, required=True, choices=['genus', 'species'], 
                        help='Set the training mode: "genus" or "species".')
    parser.add_argument('--target_genus', type=str, default=None, 
                        help='Required when mode is "species".')
    
    parser.add_argument('--teeth_ratio', type=float, default=1.0, 
                        help='Normalization maximum value for teeth features.')
    parser.add_argument('--cranium_ratio', type=float, default=1.0, 
                        help='Normalization maximum value for cranium features.')
    
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping.')
    
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Dimension of the MLP hidden layer.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Computation device (e.g., "cuda:0" or "cpu").')
    
    args = parser.parse_args()
    main(args)