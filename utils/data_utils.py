import os
from torchvision import datasets, transforms


def get_data_transforms(input_size=600):
    """Creates standard transformations for image resizing, tensor conversion, and normalization."""
    # Note: Consider adding data augmentation to the training transform for better performance.
    data_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return data_transform, data_transform


def load_genus_dataset(dataset_root, input_size=600):
    """Loads a dataset where the direct subdirectories of train/ and test/ are classes."""
    train_transform, test_transform = get_data_transforms(input_size)
    
    train_data = datasets.ImageFolder(
        root=os.path.join(dataset_root, "train"),
        transform=train_transform
    )
    test_data = datasets.ImageFolder(
        root=os.path.join(dataset_root, "test"),
        transform=test_transform
    )
    
    return train_data, test_data


def load_species_dataset(dataset_root, target_genus, input_size=600):
    """Loads a species-level dataset from within a specific 'target_genus' subdirectory."""
    train_transform, test_transform = get_data_transforms(input_size)
    
    genus_train_path = os.path.join(dataset_root, "train", target_genus)
    genus_test_path = os.path.join(dataset_root, "test", target_genus)
    
    train_data = datasets.ImageFolder(
        root=genus_train_path,
        transform=train_transform
    )
    test_data = datasets.ImageFolder(
        root=genus_test_path,
        transform=test_transform
    )
    
    return train_data, test_data