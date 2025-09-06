import torchvision.models as models
from torchvision.models import get_model_weights
import torch.nn as nn

def build_model(model_name, num_classes, use_pretrained=True):
    """
    构建骨干网络并替换分类头.
    使用现代 torchvision API (weights=...).
    """
    if use_pretrained:
        weights = get_model_weights(model_name).DEFAULT
    else:
        weights = None
    
    # 动态获取模型构造函数
    try:
        model_fn = getattr(models, model_name)
    except AttributeError:
        raise ValueError(f"模型 '{model_name}' 在 torchvision.models 中不存在。")

    # 使用 'weights' 参数实例化模型
    model = model_fn(weights=weights)

    # 替换分类头，逻辑保持不变
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential): # EfficientNet系列
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):  # ResNet, VGG等系列
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise TypeError(f"无法自动识别模型 '{model_name}' 的分类头。请检查 model_utils.py。")

    return model