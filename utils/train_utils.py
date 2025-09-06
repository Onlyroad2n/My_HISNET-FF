import os
import torch
from tqdm import tqdm
import torch.cuda.amp as amp
from contextlib import nullcontext  

class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0  

    def __call__(self, acc, epoch, model, save_path):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch 
            self.save_checkpoint(model, save_path)
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model, save_path)
            self.counter = 0

    def save_checkpoint(self, model, save_path):
        torch.save(model.state_dict(), os.path.join(save_path, 'best_network.pth'))
        if self.verbose:
            print(f"权重已更新: {save_path}/best_network.pth")


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler,
                device, num_epochs, save_dir, patience=20, writer=None):

    use_cuda = 'cuda' in device and torch.cuda.is_available()
    scaler = amp.GradScaler() if use_cuda else None
    autocast_context = amp.autocast() if use_cuda else nullcontext()

    stopper = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):
        # 训练阶段 
        model.train()
        train_loss, train_correct = 0.0, 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train") as tbar:
            for imgs, labels in tbar:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()

                with autocast_context:
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                if scaler:  # CUDA 混合精度
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:       # CPU 或没有混合精度
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item() * labels.size(0)
                train_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = train_correct / len(train_loader.dataset)
        avg_train_loss = train_loss / len(train_loader.dataset)

        # 验证阶段
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

        val_acc = val_correct / len(test_loader.dataset)
        avg_val_loss = val_loss / len(test_loader.dataset)

        scheduler.step()

        print(f"[{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")

        if writer is not None:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.add_scalar("Acc/val", val_acc, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        stopper(val_acc, epoch + 1, model, save_dir)
        if stopper.early_stop:
            print("提前停止训练")
            break

    best_acc = stopper.best_score if stopper.best_score is not None else 0.0
    best_epoch = stopper.best_epoch

    summary_file_path = os.path.join(save_dir, 'best_acc.txt')
    try:
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Best Validation Accuracy: {best_acc:.4f}\n")
            f.write(f"Best Epoch: {best_epoch}\n")
    except Exception as e:
        print(f"错误：无法将总结信息写入文件 {summary_file_path}。原因: {e}")

    summary_message = (
        f"{'='*50}\n"
        f"最优模型已保存至: {save_dir}/best_network.pth\n"
        f"总结信息已写入: {summary_file_path}\n"
        f"最优轮次 (Best Epoch): {best_epoch}\n"
        f"最高验证准确率 (Best Val Acc): {best_acc:.4f}\n"
        f"{'='*50}"
    )
    print(summary_message)
