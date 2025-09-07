import torch.nn as nn
import torchvision.models as models
from torchvision.models import get_model_weights


def build_model(model_name, num_classes, use_pretrained=True):
    """Builds a model from torchvision and replaces its classifier head."""
    
    # Use the modern 'weights' API to get pretrained weights or set to None
    weights = get_model_weights(model_name).DEFAULT if use_pretrained else None
    
    # Dynamically get the model constructor from the models library
    try:
        model_fn = getattr(models, model_name)
    except AttributeError:
        raise ValueError(f"Model '{model_name}' not found in torchvision.models.")

    # Instantiate the model with the specified weights
    model = model_fn(weights=weights)

    # Identify and replace the final classification layer
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        # Handles models like EfficientNet, which have a Sequential classifier
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):
        # Handles models like ResNet, which have a single 'fc' layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise TypeError(
            f"Classifier head for '{model_name}' not recognized. Please update model_utils.py."
        )

    return model