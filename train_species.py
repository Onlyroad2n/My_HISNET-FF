import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.utils import tensorboard
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_utils import load_species_dataset
from utils.model_utils import build_model
from utils.train_utils import train_model

# ======= 参数区 =======
data_type = 'head' #head or teeth
DATASET_ROOT = './datasets/'+data_type+'/2genus_4species'
TARGET_GENUS = 'Euroscaptor'
INPUT_SIZE = 600
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
STEP_SIZE = 10
DEVICE = 'cuda:0'
MODEL_NAME = 'efficientnet_b7'
SAVE_DIR = f'./weights/species/{TARGET_GENUS}/'+data_type
PATIENCE = 50
# ======================

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. 加载数据
    train_data, test_data = load_species_dataset(DATASET_ROOT, TARGET_GENUS, INPUT_SIZE)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    # === 类别信息打印 ===
    print(f"[{TARGET_GENUS}] 模式下，共 {len(train_data.classes)} 个物种类别")
    print(f"类别映射: {train_data.class_to_idx}")
    for cls_name, idx in train_data.class_to_idx.items():
        count = sum(1 for t in train_data.targets if t == idx)
        print(f"  {cls_name}: {count} 张图片")
    print(f"训练集总样本数: {len(train_data)}")
    print(f"测试集总样本数: {len(test_data)}")
    print("=" * 40)

    # 2. 创建模型
    model = build_model(MODEL_NAME, num_classes=len(train_data.classes), pretrained=True).to(DEVICE)

    # 3. 定义优化器、损失、调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.8)

    # 4. TensorBoard
    writer = tensorboard.SummaryWriter(os.path.join(SAVE_DIR, 'logs'))

    # 5. 训练
    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler,
                device=DEVICE, num_epochs=NUM_EPOCHS, save_dir=SAVE_DIR,
                class_names=train_data.classes, patience=PATIENCE, writer=writer)

    writer.close()
