import os
import joblib
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


def load_features_from_npy(feature_dir: str):
    """Loads train/test features and labels from .npy files within a directory."""
    x_train = np.load(os.path.join(feature_dir, 'train_features.npy'))
    y_train = np.load(os.path.join(feature_dir, 'train_labels.npy'))
    x_test = np.load(os.path.join(feature_dir, 'test_features.npy'))
    y_test = np.load(os.path.join(feature_dir, 'test_labels.npy'))
    
    return x_train, y_train, x_test, y_test


def fuse_features(teeth_dir: str, cranium_dir: str, teeth_ratio=1.0, cranium_ratio=1.0, save_scaler_dir=None):
    """Loads, scales, and concatenates features from teeth and cranium sources."""
    # Load features from the respective .npy files
    tx_train, ty_train, tx_test, ty_test = load_features_from_npy(teeth_dir)
    hx_train, hy_train, hx_test, hy_test = load_features_from_npy(cranium_dir)

    # Verify that labels match between the two datasets
    assert np.array_equal(ty_train, hy_train), "Training labels do not match! Please check the feature extraction process."
    assert np.array_equal(ty_test, hy_test), "Testing labels do not match! Please check the feature extraction process."

    # Scale teeth features
    scaler_teeth = MinMaxScaler(feature_range=(0, teeth_ratio))
    tx_train_scaled = scaler_teeth.fit_transform(tx_train)
    tx_test_scaled = scaler_teeth.transform(tx_test)

    # Scale cranium features
    scaler_cranium = MinMaxScaler(feature_range=(0, cranium_ratio))
    hx_train_scaled = scaler_cranium.fit_transform(hx_train)
    hx_test_scaled = scaler_cranium.transform(hx_test)

    # Optionally save the fitted scalers for later use (e.g., in inference)
    if save_scaler_dir:
        os.makedirs(save_scaler_dir, exist_ok=True)
        joblib.dump(scaler_teeth, os.path.join(save_scaler_dir, 'scaler_teeth.pkl'))
        joblib.dump(scaler_cranium, os.path.join(save_scaler_dir, 'scaler_cranium.pkl'))
        print(f"Feature scalers have been saved to: {save_scaler_dir}")

    # Concatenate the scaled features
    x_train_fused = np.concatenate([tx_train_scaled, hx_train_scaled], axis=1)
    x_test_fused = np.concatenate([tx_test_scaled, hx_test_scaled], axis=1)

    return x_train_fused, ty_train, x_test_fused, ty_test


class FeatureDataset(Dataset):
    """A custom PyTorch Dataset for features and labels loaded from NumPy arrays."""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.data = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]