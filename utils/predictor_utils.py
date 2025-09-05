# utils/predictor_utils.py
import os
import torch
from PIL import Image
from torchvision import transforms

# 导入你项目中的其他工具模块
from .data_utils import get_data_transforms
from .model_utils import build_model

import joblib
from .fusion_models import SimpleMLP 
import torch.nn as nn
import numpy as np

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

class FusionHierarchicalPredictor:
    """
    (新增) 一个封装了融合特征+二级层级预测所需全部逻辑的类。
    """
    def __init__(self, model_name='efficientnet_b7', device='cuda:0'):
        print("Initializing Fusion Hierarchical Predictor...")
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        _, self.transform = get_data_transforms()

        # 1. 加载属级别的所有资源 (特征提取器, scaler, MLP)
        self.genus_resources = self._load_fusion_resources('genus')
        if not self.genus_resources:
            raise FileNotFoundError("未能加载属级别(genus)的融合所需资源。请先完成训练。")
        print(f"✓ Genus-level fusion resources loaded.")

        # 2. 自动发现并加载所有种级别的资源
        self.species_resources = {}
        species_list = self._discover_species()
        for genus_name in species_list:
            resources = self._load_fusion_resources('species', genus_name)
            if resources:
                self.species_resources[genus_name] = resources
        print(f"✓ Species-level fusion resources loaded for {len(self.species_resources)} genera: {list(self.species_resources.keys())}")
        print("-" * 30)

    def _discover_species(self):
        """扫描目录，发现所有训练过融合模型的属。"""
        weights_fusion_dir = './weights_fusion/species/'
        if not os.path.exists(weights_fusion_dir): return []
        return [d for d in os.listdir(weights_fusion_dir) if os.path.isdir(os.path.join(weights_fusion_dir, d))]

    def _load_fusion_resources(self, mode, target_genus=None):
        """一个通用的辅助函数，用于加载一个层级所需的全套资源。"""
        if mode == 'genus':
            base_weights_path = './weights/genus'
            base_fusion_path = './weights_fusion/genus'
        else:
            base_weights_path = f'./weights/species/{target_genus}'
            base_fusion_path = f'./weights_fusion/species/{target_genus}'

        try:
            # 加载类别信息以确定模型输出维度
            with open(os.path.join(base_weights_path, 'head/classes.txt'), 'r') as f:
                num_classes_backbone = len([line.strip() for line in f.readlines()])
            with open(os.path.join(base_fusion_path, 'classes.txt'), 'r') as f:
                classes = [line.strip() for line in f.readlines()]
                num_classes_mlp = len(classes)

            # 1. 加载Head特征提取器
            head_model = build_model(self.model_name, num_classes=num_classes_backbone, use_pretrained=False).to(self.device)
            head_model.load_state_dict(torch.load(os.path.join(base_weights_path, 'head/best_network.pth'), map_location=self.device))
            head_model.classifier[-1] = nn.Identity()
            head_model.eval()

            # 2. 加载Teeth特征提取器
            teeth_model = build_model(self.model_name, num_classes=num_classes_backbone, use_pretrained=False).to(self.device)
            teeth_model.load_state_dict(torch.load(os.path.join(base_weights_path, 'teeth/best_network.pth'), map_location=self.device))
            teeth_model.classifier[-1] = nn.Identity()
            teeth_model.eval()

            # 3. 加载Scalers
            scaler_head = joblib.load(os.path.join(base_fusion_path, 'scaler_head.pkl'))
            scaler_teeth = joblib.load(os.path.join(base_fusion_path, 'scaler_teeth.pkl'))

            # 4. 加载融合MLP模型
            # 动态推断MLP的输入维度
            dummy_input_head = torch.randn(1, 2560) # b7的特征维度
            dummy_input_teeth = torch.randn(1, 2560)
            input_dim = np.concatenate([scaler_head.transform(dummy_input_head), scaler_teeth.transform(dummy_input_teeth)], axis=1).shape[1]
            
            fusion_mlp = SimpleMLP(input_dim=input_dim, num_class=num_classes_mlp).to(self.device)
            fusion_mlp.load_state_dict(torch.load(os.path.join(base_fusion_path, 'best_fusion_model.pth'), map_location=self.device))
            fusion_mlp.eval()

            return {
                "head_extractor": head_model, "teeth_extractor": teeth_model,
                "scaler_head": scaler_head, "scaler_teeth": scaler_teeth,
                "fusion_mlp": fusion_mlp, "classes": classes
            }
        except FileNotFoundError as e:
            print(f"警告: 在加载 '{mode}/{target_genus or ''}' 的资源时失败: {e}")
            return None

    @torch.no_grad()
    def _predict_single_level(self, head_tensor, teeth_tensor, resources):
        """对单个层级进行一次完整的 融合->预测 流程。"""
        # 提取特征
        head_feat = resources['head_extractor'](head_tensor).cpu().numpy()
        teeth_feat = resources['teeth_extractor'](teeth_tensor).cpu().numpy()
        # 归一化
        head_feat_scaled = resources['scaler_head'].transform(head_feat)
        teeth_feat_scaled = resources['scaler_teeth'].transform(teeth_feat)
        # 融合
        fused_feat = np.concatenate([teeth_feat_scaled, head_feat_scaled], axis=1)
        fused_tensor = torch.from_numpy(fused_feat).float().to(self.device)
        # MLP预测
        output = resources['fusion_mlp'](fused_tensor)
        prob = torch.softmax(output, dim=1)
        confidence, pred_idx = torch.max(prob, 1)
        
        return resources['classes'][pred_idx.item()], confidence.item()

    def predict(self, head_image_path: str, teeth_image_path: str):
        """对一对图片进行完整的二级融合层级预测。"""
        head_image = Image.open(head_image_path).convert('RGB')
        teeth_image = Image.open(teeth_image_path).convert('RGB')
        head_tensor = self.transform(head_image).unsqueeze(0).to(self.device)
        teeth_tensor = self.transform(teeth_image).unsqueeze(0).to(self.device)

        # 第一级：属预测
        predicted_genus, genus_confidence = self._predict_single_level(
            head_tensor, teeth_tensor, self.genus_resources
        )

        # 第二级：种预测
        if predicted_genus in self.species_resources:
            predicted_species, species_confidence = self._predict_single_level(
                head_tensor, teeth_tensor, self.species_resources[predicted_genus]
            )
            final_prediction = predicted_species
            final_confidence = species_confidence
        else:
            final_prediction = predicted_genus
            final_confidence = genus_confidence

        return {
            "predicted_genus": predicted_genus,
            "genus_confidence": genus_confidence,
            "final_prediction": final_prediction,
            "final_confidence": final_confidence
        }