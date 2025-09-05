import os
import torch
from tqdm import tqdm
import torch.cuda.amp as amp

class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, acc, model, save_path):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, save_path)
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, save_path)
            self.counter = 0

    def save_checkpoint(self, model, save_path):
        torch.save(model.state_dict(), os.path.join(save_path, 'best_network.pth'))
        if self.verbose:
            print(f"✓ 权重已更新: {save_path}/best_network.pth")

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler,
                device, num_epochs, save_dir, class_names, patience=20, writer=None):

    scaler = amp.GradScaler()  # 混合精度梯度缩放
    stopper = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):
        # ===== 训练阶段 =====
        model.train()
        train_loss, train_correct = 0.0, 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train") as tbar:
            for imgs, labels in tbar:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()

                with amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * labels.size(0)
                train_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = train_correct / len(train_loader.dataset)
        avg_train_loss = train_loss / len(train_loader.dataset)

        # ===== 验证阶段 =====
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(test_loader.dataset)
        avg_val_loss = val_loss / len(test_loader.dataset)

        scheduler.step()

        # ===== 日志输出 =====
        print(f"[{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")

        if writer is not None:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.add_scalar("Acc/val", val_acc, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        # ===== 早停判断 =====
        stopper(val_acc, model, save_dir)
        if stopper.early_stop:
            print("✋ 提前停止训练")
            break

    # 保存类别标签
    with open(os.path.join(save_dir, 'classes.txt'), 'w') as f:
        for c in class_names:
            f.write(f"{c}\n")
