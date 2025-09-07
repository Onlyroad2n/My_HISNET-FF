import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import torch
import joblib

def load_features_from_npy(feature_dir: str):
    # 加载 .npy 文件
    x_train = np.load(os.path.join(feature_dir, 'train_features.npy'))
    y_train = np.load(os.path.join(feature_dir, 'train_labels.npy'))
    x_test = np.load(os.path.join(feature_dir, 'test_features.npy'))
    y_test = np.load(os.path.join(feature_dir, 'test_labels.npy'))
    
    return x_train, y_train, x_test, y_test

def fuse_features(teeth_dir: str, cranium_dir: str, teeth_ratio=1.0, cranium_ratio=1.0, save_scaler_dir=None):
    # 从 .npy 文件加载特征
    tx_train, ty_train, tx_test, ty_test = load_features_from_npy(teeth_dir)
    hx_train, hy_train, hx_test, hy_test = load_features_from_npy(cranium_dir)

    assert np.array_equal(ty_train, hy_train), "训练集标签不匹配！请检查特征提取过程。"
    assert np.array_equal(ty_test, hy_test), "测试集标签不匹配！请检查特征提取过程。"

    # Teeth归一化
    scaler_teeth = MinMaxScaler(feature_range=(0, teeth_ratio))
    tx_train_scaled = scaler_teeth.fit_transform(tx_train)
    tx_test_scaled  = scaler_teeth.transform(tx_test)

    # Cranium归一化
    scaler_cranium = MinMaxScaler(feature_range=(0, cranium_ratio))
    hx_train_scaled = scaler_cranium.fit_transform(hx_train)
    hx_test_scaled  = scaler_cranium.transform(hx_test)

    # 可选地保存scaler
    if save_scaler_dir:
        os.makedirs(save_scaler_dir, exist_ok=True)
        joblib.dump(scaler_teeth, os.path.join(save_scaler_dir, 'scaler_teeth.pkl'))
        joblib.dump(scaler_cranium, os.path.join(save_scaler_dir, 'scaler_cranium.pkl'))
        print(f"特征缩放器 (Scaler) 已保存至: {save_scaler_dir}")

    # 特征拼接 (CONCAT)
    x_train_fused = np.concatenate([tx_train_scaled, hx_train_scaled], axis=1)
    x_test_fused  = np.concatenate([tx_test_scaled, hx_test_scaled], axis=1)

    return x_train_fused, ty_train, x_test_fused, ty_test


class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.data = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]