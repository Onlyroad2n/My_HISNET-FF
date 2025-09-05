# utils/predictor_utils.py
import os
import torch
from PIL import Image
from torchvision import transforms

# 导入你项目中的其他工具模块
from .data_utils import get_data_transforms
from .model_utils import build_model

class HierarchicalPredictor:
    """
    一个封装了二级层级预测所需全部逻辑的类。
    初始化时加载所有必需的模型。
    """
    def __init__(self, model_name='efficientnet_b7', data_type='head', device='cuda:0'):
        """
        初始化预测器。
        - model_name: 使用的模型架构名称。
        - data_type: 要加载的数据类型 ('head' or 'teeth')。
        - device: 计算设备。
        """
        print("Initializing Hierarchical Predictor...")
        self.model_name = model_name
        self.data_type = data_type
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 1. 获取图像预处理变换
        _, self.transform = get_data_transforms()
        
        # 2. 加载主分类器（属分类器）
        self.genus_model, self.genus_classes = self._load_genus_model()
        print(f"✓ Genus model loaded. Found {len(self.genus_classes)} classes: {self.genus_classes}")

        # 3. 自动发现并加载所有次级分类器（种分类器）
        self.species_models = self._load_all_species_models()
        print(f"✓ Species models loaded for {len(self.species_models)} genera: {list(self.species_models.keys())}")
        print("-" * 30)

    def _load_model(self, model_path, class_path):
        """通用模型加载函数。"""
        if not os.path.exists(model_path) or not os.path.exists(class_path):
            return None, None

        # 从 classes.txt 读取类别
        with open(class_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        num_classes = len(classes)

        # 构建模型并加载权重
        model = build_model(self.model_name, num_classes=num_classes, use_pretrained=False).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model, classes

    def _load_genus_model(self):
        """加载属分类器。"""
        model_path = f'./weights/genus/{self.data_type}/best_network.pth'
        class_path = f'./weights/genus/{self.data_type}/classes.txt'
        model, classes = self._load_model(model_path, class_path)
        if model is None:
            raise FileNotFoundError(f"Genus model or class file not found at: ./weights/genus/{self.data_type}/")
        return model, classes

    def _load_all_species_models(self):
        """扫描weights目录，加载所有可用的种分类器。"""
        species_models_dict = {}
        species_root_dir = './weights/species/'
        if not os.path.exists(species_root_dir):
            return species_models_dict

        # 遍历所有潜在的属目录
        for genus_name in os.listdir(species_root_dir):
            genus_dir = os.path.join(species_root_dir, genus_name)
            if os.path.isdir(genus_dir):
                model_path = os.path.join(genus_dir, self.data_type, 'best_network.pth')
                class_path = os.path.join(genus_dir, self.data_type, 'classes.txt')
                
                model, classes = self._load_model(model_path, class_path)
                if model and classes:
                    species_models_dict[genus_name] = (model, classes)
        return species_models_dict

    @torch.no_grad()
    def predict(self, image_path: str):
        """
        对单张图片进行完整的二级层级预测。
        """
        # 1. 图像预处理
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 2. 第一级：属预测
        genus_output = self.genus_model(image_tensor)
        genus_prob = torch.softmax(genus_output, dim=1)
        genus_confidence, genus_pred_idx = torch.max(genus_prob, 1)
        predicted_genus = self.genus_classes[genus_pred_idx.item()]

        # 3. 第二级：种预测（条件执行）
        if predicted_genus in self.species_models:
            species_model, species_classes = self.species_models[predicted_genus]
            species_output = species_model(image_tensor)
            species_prob = torch.softmax(species_output, dim=1)
            species_confidence, species_pred_idx = torch.max(species_prob, 1)
            predicted_species = species_classes[species_pred_idx.item()]
            
            final_prediction = predicted_species
            final_confidence = species_confidence.item()
        else:
            # 如果没有专门的种分类器，最终预测结果就是属名
            final_prediction = predicted_genus
            final_confidence = genus_confidence.item()

        return {
            "predicted_genus": predicted_genus,
            "genus_confidence": genus_confidence.item(),
            "final_prediction": final_prediction,
            "final_confidence": final_confidence
        }