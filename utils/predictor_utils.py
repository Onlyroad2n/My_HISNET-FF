import os
import torch
from PIL import Image
import joblib
import numpy as np
import torch.nn as nn
from .data_utils import get_data_transforms
from .model_utils import build_model
from .fusion_models import SimpleMLP
from config import WEIGHTS_ROOT as CFG_WEIGHTS_ROOT, FEATURES_ROOT as CFG_FEATURES_ROOT

class HierarchicalPredictor:
    """
    单模态预测器（属+种两级）
    """
    def __init__(self, model_name='efficientnet_b7', data_type='cranium',
                 dataset_name=None, weights_root=None,input_size=600, device='cuda:0'):
        self.model_name = model_name
        self.data_type = data_type
        self.dataset_name = dataset_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.weights_root = weights_root or CFG_WEIGHTS_ROOT
        self.input_size = input_size

        _, self.transform = get_data_transforms()

        self.genus_model, self.genus_classes = self._load_genus_model()
        print(f" Genus model loaded: {self.genus_classes}")

        self.species_models = self._load_all_species_models()
        print(f"Species models loaded for genera: {list(self.species_models.keys())}")

    def _load_model(self, model_path, class_path):
        if not os.path.exists(model_path) or not os.path.exists(class_path):
            return None, None
        with open(class_path) as f:
            classes = [line.strip() for line in f]
        num_classes = len(classes)
        model = build_model(self.model_name, num_classes, use_pretrained=False).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model, classes

    def _load_genus_model(self):
        model_path = os.path.join(self.weights_root, 'pretrain', 'genus', self.data_type, 'best_network.pth')
        class_path = os.path.join(self.weights_root, 'pretrain', 'genus', self.data_type, 'classes.txt')
        model, classes = self._load_model(model_path, class_path)
        if not model:
            raise FileNotFoundError(f"Genus model files not found at {model_path}")
        return model, classes

    def _load_all_species_models(self):
        species_dir = os.path.join(self.weights_root, 'pretrain', 'species')
        models_dict = {}
        if not os.path.exists(species_dir):
            return models_dict
        for genus in os.listdir(species_dir):
            genus_dir = os.path.join(species_dir, genus)
            if os.path.isdir(genus_dir):
                mpath = os.path.join(genus_dir, self.data_type, 'best_network.pth')
                cpath = os.path.join(genus_dir, self.data_type, 'classes.txt')
                model, classes = self._load_model(mpath, cpath)
                if model:
                    models_dict[genus] = (model, classes)
        return models_dict

    @torch.no_grad()
    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        g_out = self.genus_model(tensor)
        g_prob = torch.softmax(g_out, dim=1)
        g_conf, g_idx = g_prob.max(1)
        genus_name = self.genus_classes[g_idx.item()]
        if genus_name in self.species_models:
            sp_model, sp_classes = self.species_models[genus_name]
            s_out = sp_model(tensor)
            s_prob = torch.softmax(s_out, dim=1)
            s_conf, s_idx = s_prob.max(1)
            sp_name = sp_classes[s_idx.item()]
            return {
                "predicted_genus": genus_name,
                "genus_confidence": g_conf.item(),
                "final_prediction": sp_name,
                "final_confidence": s_conf.item()
            }
        else:
            return {
                "predicted_genus": genus_name,
                "genus_confidence": g_conf.item(),
                "final_prediction": genus_name,
                "final_confidence": g_conf.item()
            }


class FusionHierarchicalPredictor:
    """
    融合预测器（头骨+牙齿）
    """
    def __init__(self, model_name='efficientnet_b7', features_root=None, weights_root=None, input_size=600, device='cuda:0'):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.features_root = features_root or CFG_FEATURES_ROOT
        self.weights_root = weights_root or CFG_WEIGHTS_ROOT
        self.input_size = input_size
        _, self.transform = get_data_transforms()

        self.genus_resources = self._load_fusion_resources('genus')
        if not self.genus_resources:
            raise FileNotFoundError("Genus fusion resources not found.")

        self.species_resources = {}
        for genus in self._discover_species():
            res = self._load_fusion_resources('species', genus)
            if res:
                self.species_resources[genus] = res

    def _discover_species(self):
        species_dir = os.path.join(self.weights_root, 'fusion', 'species')
        if not os.path.exists(species_dir):
            return []
        return [d for d in os.listdir(species_dir) if os.path.isdir(os.path.join(species_dir, d))]

    def _load_fusion_resources(self, mode, genus_name=None):
        if mode == 'genus':
            base_weights_path = os.path.join(self.weights_root, 'pretrain', 'genus')
            base_fusion_path  = os.path.join(self.weights_root, 'fusion', 'genus')
        else:
            base_weights_path = os.path.join(self.weights_root, 'pretrain', 'species', genus_name)
            base_fusion_path  = os.path.join(self.weights_root, 'fusion', 'species', genus_name)
        try:
            # 类别信息
            with open(os.path.join(base_weights_path, 'cranium', 'classes.txt')) as f:
                num_classes_backbone = len([l.strip() for l in f])
            with open(os.path.join(base_fusion_path, 'classes.txt')) as f:
                fusion_classes = [l.strip() for l in f]
            num_classes_mlp = len(fusion_classes)

            # Cranium 模型
            cranium_model = build_model(self.model_name, num_classes=num_classes_backbone, use_pretrained=False).to(self.device)
            if hasattr(cranium_model, 'classifier') and isinstance(cranium_model.classifier, nn.Sequential):
                feature_dim = cranium_model.classifier[-1].in_features
            elif hasattr(cranium_model, 'fc'):
                feature_dim = cranium_model.fc.in_features
            else:
                raise ValueError(f"无法识别模型 {self.model_name} 的特征层")
            cranium_model.load_state_dict(torch.load(os.path.join(base_weights_path, 'cranium', 'best_network.pth'),
                                                  map_location=self.device))
            if hasattr(cranium_model, 'classifier') and isinstance(cranium_model.classifier, nn.Sequential):
                cranium_model.classifier[-1] = nn.Identity()
            elif hasattr(cranium_model, 'fc'):
                cranium_model.fc = nn.Identity()
            cranium_model.eval()

            # Teeth 模型
            teeth_model = build_model(self.model_name, num_classes=num_classes_backbone, use_pretrained=False).to(self.device)
            teeth_model.load_state_dict(torch.load(os.path.join(base_weights_path, 'teeth', 'best_network.pth'),
                                                   map_location=self.device))
            if hasattr(teeth_model, 'classifier') and isinstance(teeth_model.classifier, nn.Sequential):
                teeth_model.classifier[-1] = nn.Identity()
            elif hasattr(teeth_model, 'fc'):
                teeth_model.fc = nn.Identity()
            teeth_model.eval()

            # Scalers
            scaler_cranium = joblib.load(os.path.join(base_fusion_path, 'scaler_cranium.pkl'))
            scaler_teeth = joblib.load(os.path.join(base_fusion_path, 'scaler_teeth.pkl'))

            # Fusion MLP
            input_dim = feature_dim * 2
            fusion_mlp = SimpleMLP(input_dim=input_dim, num_class=num_classes_mlp).to(self.device)
            fusion_mlp.load_state_dict(torch.load(os.path.join(base_fusion_path, 'best_fusion_model.pth'),
                                                  map_location=self.device))
            fusion_mlp.eval()

            return {
                "cranium_extractor": cranium_model,
                "teeth_extractor": teeth_model,
                "scaler_cranium": scaler_cranium,
                "scaler_teeth": scaler_teeth,
                "fusion_mlp": fusion_mlp,
                "classes": fusion_classes
            }

        except FileNotFoundError as e:
            print(f"警告: {mode}/{genus_name or ''} 资源缺失: {e}")
            return None

    def _predict_single_level(self, cranium_tensor, teeth_tensor, res):
        with torch.no_grad():
            cranium_feat = res['cranium_extractor'](cranium_tensor).detach().cpu().numpy()
            teeth_feat = res['teeth_extractor'](teeth_tensor).detach().cpu().numpy()
            cranium_feat_scaled = res['scaler_cranium'].transform(cranium_feat)
            teeth_feat_scaled = res['scaler_teeth'].transform(teeth_feat)
            fused_feat = np.concatenate([teeth_feat_scaled, cranium_feat_scaled], axis=1)
            fused_tensor = torch.from_numpy(fused_feat).float().to(self.device)
            out = res['fusion_mlp'](fused_tensor)
            prob = torch.softmax(out, dim=1)
            conf, idx = prob.max(1)
        return res['classes'][idx.item()], conf.item()

    def predict(self, cranium_image_path, teeth_image_path):
        cranium_img = Image.open(cranium_image_path).convert('RGB')
        teeth_img = Image.open(teeth_image_path).convert('RGB')
        cranium_t = self.transform(cranium_img).unsqueeze(0).to(self.device)
        teeth_t = self.transform(teeth_img).unsqueeze(0).to(self.device)
        g_name, g_conf = self._predict_single_level(cranium_t, teeth_t, self.genus_resources)
        if g_name in self.species_resources:
            s_name, s_conf = self._predict_single_level(cranium_t, teeth_t, self.species_resources[g_name])
            return {
                "predicted_genus": g_name, "genus_confidence": g_conf,
                "final_prediction": s_name, "final_confidence": s_conf
            }
        else:
            return {
                "predicted_genus": g_name, "genus_confidence": g_conf,
                "final_prediction": g_name, "final_confidence": g_conf
            }
