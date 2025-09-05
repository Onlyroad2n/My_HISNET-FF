import torchvision.models as models
import torch.nn as nn

def build_model(model_name, num_classes, pretrained=True):
    """构建骨干网络并替换分类头"""
    model_fn = getattr(models, model_name)
    model = model_fn(pretrained=pretrained)

    if hasattr(model, 'classifier'):  # EfficientNet系列
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):  # ResNet系列
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")

    return model
