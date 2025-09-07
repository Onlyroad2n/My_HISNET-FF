import os
from contextlib import nullcontext

import torch
import torch.cuda.amp as amp
from tqdm import tqdm


class EarlyStopping:
    """Stops training when a monitored metric has stopped improving."""
    def __init__(self, patience=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation accuracy improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation accuracy improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, acc, epoch, model, save_dir):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model, save_dir)
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model, save_dir)
            self.counter = 0

    def save_checkpoint(self, model, save_dir):
        """Saves model when validation accuracy increases."""
        save_path = os.path.join(save_dir, 'best_network.pth')
        torch.save(model.state_dict(), save_path)
        if self.verbose:
            print(f"Validation accuracy improved. Saving model to: {save_path}")


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler,
                device, num_epochs, save_dir, patience=20, writer=None):
    """
    Main function to train and validate a model.
    """
    use_amp = 'cuda' in device and torch.cuda.is_available()
    scaler = amp.GradScaler(enabled=use_amp)
    autocast_context = amp.autocast() if use_amp else nullcontext()

    stopper = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss, train_correct = 0.0, 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for imgs, labels in train_pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast_context:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()

            train_loss += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast_context:
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        avg_val_loss = val_loss / len(test_loader.dataset)
        val_acc = val_correct / len(test_loader.dataset)

        scheduler.step()

        print(f"[{epoch+1:03d}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/validation", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/validation", val_acc, epoch)
            writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)

        # Early stopping check
        stopper(val_acc, epoch + 1, model, save_dir)
        if stopper.early_stop:
            print("Early stopping triggered.")
            break

    # --- Final Summary ---
    best_acc = stopper.best_score if stopper.best_score is not None else 0.0
    best_epoch = stopper.best_epoch

    summary_file_path = os.path.join(save_dir, 'training_summary.txt')
    try:
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Best Validation Accuracy: {best_acc:.4f}\n")
            f.write(f"Achieved at Epoch: {best_epoch}\n")
    except IOError as e:
        print(f"Error: Failed to write summary to {summary_file_path}. Reason: {e}")

    summary_message = (
        f"\n{'='*50}\n"
        f"Training Finished.\n"
        f"Best model saved to: {os.path.join(save_dir, 'best_network.pth')}\n"
        f"Summary written to: {summary_file_path}\n"
        f"Best Epoch: {best_epoch}\n"
        f"Best Validation Accuracy: {best_acc:.4f}\n"
        f"{'='*50}"
    )
    print(summary_message)