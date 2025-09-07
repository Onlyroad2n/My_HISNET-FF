import os
from typing import Dict

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .data_utils import get_data_transforms
from .fusion_models import SimpleMLP
from .model_utils import build_model
from config import WEIGHTS_ROOT as CFG_WEIGHTS_ROOT


class HierarchicalPredictor:
    """
    Performs hierarchical prediction (genus, then species) using a single image modality.
    """
    def __init__(self, model_name: str = 'efficientnet_b7', data_type: str = 'cranium',
                 weights_root: str = None, input_size: int = 600, device: str = 'cuda:0'):
        """Initializes the single-modality predictor by loading genus and species models."""
        self.model_name = model_name
        self.data_type = data_type
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.weights_root = weights_root or CFG_WEIGHTS_ROOT
        self.input_size = input_size

        _, self.transform = get_data_transforms(input_size=self.input_size)

        self.genus_model, self.genus_classes = self._load_genus_model()
        print(f"Successfully loaded Genus model for '{self.data_type}' with classes: {self.genus_classes}")

        self.species_models = self._load_all_species_models()
        print(f"Found {len(self.species_models)} species-level models for genera: {list(self.species_models.keys())}")

    def _load_model(self, model_path: str, class_path: str):
        """A helper function to load a model and its corresponding class list."""
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
        """Loads the primary genus-level classification model."""
        model_dir = os.path.join(self.weights_root, 'pretrain', 'genus', self.data_type)
        model_path = os.path.join(model_dir, 'best_network.pth')
        class_path = os.path.join(model_dir, 'classes.txt')
        model, classes = self._load_model(model_path, class_path)
        if not model:
            raise FileNotFoundError(f"Genus model or class file not found in '{model_dir}'. Please run train.py first.")
        return model, classes

    def _load_all_species_models(self):
        """Loads all available species-level models for different genera."""
        species_dir = os.path.join(self.weights_root, 'pretrain', 'species')
        models_dict = {}
        if not os.path.exists(species_dir):
            return models_dict
        for genus in os.listdir(species_dir):
            genus_dir = os.path.join(species_dir, genus)
            if os.path.isdir(genus_dir):
                model_dir = os.path.join(genus_dir, self.data_type)
                mpath = os.path.join(model_dir, 'best_network.pth')
                cpath = os.path.join(model_dir, 'classes.txt')
                model, classes = self._load_model(mpath, cpath)
                if model:
                    models_dict[genus] = (model, classes)
        return models_dict

    @torch.no_grad()
    def predict(self, image_path: str) -> Dict[str, any]:
        """
        Predicts the genus and species from a single image file.
        """
        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Genus-level prediction
        g_out = self.genus_model(tensor)
        g_prob = torch.softmax(g_out, dim=1)
        g_conf, g_idx = g_prob.max(1)
        genus_name = self.genus_classes[g_idx.item()]

        # Species-level prediction if a corresponding model exists
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
            # Fallback to genus prediction if no species model is found
            return {
                "predicted_genus": genus_name,
                "genus_confidence": g_conf.item(),
                "final_prediction": genus_name, # The most specific prediction is the genus itself
                "final_confidence": g_conf.item()
            }


class FusionHierarchicalPredictor:
    """Performs hierarchical prediction by fusing features from cranium and teeth images."""
    def __init__(self, model_name: str = 'efficientnet_b7', weights_root: str = None, 
                 input_size: int = 600, device: str = 'cuda:0'):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.weights_root = weights_root or CFG_WEIGHTS_ROOT
        self.input_size = input_size
        _, self.transform = get_data_transforms(input_size=self.input_size)

        self.genus_resources = self._load_fusion_resources('genus')
        if not self.genus_resources:
            raise FileNotFoundError("Genus-level fusion resources (models, scalers) not found.")
        print("Successfully loaded Genus-level fusion resources.")

        self.species_resources = {}
        for genus in self._discover_species_genera():
            res = self._load_fusion_resources('species', genus)
            if res:
                self.species_resources[genus] = res
        print(f"Found {len(self.species_resources)} species-level fusion resources for genera: {list(self.species_resources.keys())}")


    def _discover_species_genera(self):
        """Finds all genera that have trained species-level fusion models."""
        species_dir = os.path.join(self.weights_root, 'fusion', 'species')
        if not os.path.exists(species_dir):
            return []
        return [d for d in os.listdir(species_dir) if os.path.isdir(os.path.join(species_dir, d))]

    def _prepare_feature_extractor(self, base_path: str, num_classes: int) -> nn.Module:
        """Loads a pretrained backbone and modifies it to be a feature extractor."""
        model = build_model(self.model_name, num_classes=num_classes, use_pretrained=False).to(self.device)
        model.load_state_dict(torch.load(base_path, map_location=self.device))
        
        # Replace the classifier with an identity layer
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = nn.Identity()
        elif hasattr(model, 'fc'):
            model.fc = nn.Identity()
        else:
            raise TypeError(f"Cannot adapt model '{self.model_name}' for feature extraction.")
            
        model.eval()
        return model

    def _load_fusion_resources(self, mode: str, genus_name: str = None) -> Dict:
        """Loads all necessary components (models, scalers, MLP) for a fusion level."""
        if mode == 'genus':
            base_weights_path = os.path.join(self.weights_root, 'pretrain', 'genus')
            base_fusion_path = os.path.join(self.weights_root, 'fusion', 'genus')
        else:
            base_weights_path = os.path.join(self.weights_root, 'pretrain', 'species', genus_name)
            base_fusion_path = os.path.join(self.weights_root, 'fusion', 'species', genus_name)
        try:
            # Load class information to determine model output sizes
            with open(os.path.join(base_weights_path, 'cranium', 'classes.txt')) as f:
                backbone_num_classes = len([l.strip() for l in f])
            with open(os.path.join(base_fusion_path, 'classes.txt')) as f:
                fusion_classes = [l.strip() for l in f]
            mlp_num_classes = len(fusion_classes)

            # Load feature extractors
            cranium_extractor = self._prepare_feature_extractor(
                os.path.join(base_weights_path, 'cranium', 'best_network.pth'), backbone_num_classes)
            teeth_extractor = self._prepare_feature_extractor(
                os.path.join(base_weights_path, 'teeth', 'best_network.pth'), backbone_num_classes)
            
            # Get feature dimension from one of the models
            temp_model = build_model(self.model_name, num_classes=backbone_num_classes)
            if hasattr(temp_model, 'classifier') and isinstance(temp_model.classifier, nn.Sequential):
                feature_dim = temp_model.classifier[-1].in_features
            else:
                feature_dim = temp_model.fc.in_features

            # Load scalers
            scaler_cranium = joblib.load(os.path.join(base_fusion_path, 'scaler_cranium.pkl'))
            scaler_teeth = joblib.load(os.path.join(base_fusion_path, 'scaler_teeth.pkl'))

            # Load Fusion MLP
            fusion_mlp = SimpleMLP(input_dim=(feature_dim * 2), num_classes=mlp_num_classes).to(self.device)
            fusion_mlp.load_state_dict(torch.load(os.path.join(base_fusion_path, 'best_fusion_model.pth'),
                                                  map_location=self.device))
            fusion_mlp.eval()

            return {
                "cranium_extractor": cranium_extractor, "teeth_extractor": teeth_extractor,
                "scaler_cranium": scaler_cranium, "scaler_teeth": scaler_teeth,
                "fusion_mlp": fusion_mlp, "classes": fusion_classes
            }
        except FileNotFoundError as e:
            print(f"Warning: Could not load resources for {mode}/{genus_name or ''}. Missing file: {e.filename}")
            return None

    def _predict_single_level(self, cranium_tensor, teeth_tensor, resources):
        """Performs a prediction at a single hierarchical level (genus or species)."""
        with torch.no_grad():
            cranium_feat = resources['cranium_extractor'](cranium_tensor).cpu().numpy()
            teeth_feat = resources['teeth_extractor'](teeth_tensor).cpu().numpy()
            
            cranium_feat_scaled = resources['scaler_cranium'].transform(cranium_feat)
            teeth_feat_scaled = resources['scaler_teeth'].transform(teeth_feat)
            
            fused_feat = np.concatenate([teeth_feat_scaled, cranium_feat_scaled], axis=1)
            fused_tensor = torch.from_numpy(fused_feat).float().to(self.device)
            
            out = resources['fusion_mlp'](fused_tensor)
            prob = torch.softmax(out, dim=1)
            conf, idx = prob.max(1)
            
        return resources['classes'][idx.item()], conf.item()

    def predict(self, cranium_image_path: str, teeth_image_path: str) -> Dict[str, any]:
        """
        Predicts genus and species by fusing features from cranium and teeth images.
        """
        cranium_img = Image.open(cranium_image_path).convert('RGB')
        teeth_img = Image.open(teeth_image_path).convert('RGB')
        cranium_t = self.transform(cranium_img).unsqueeze(0).to(self.device)
        teeth_t = self.transform(teeth_img).unsqueeze(0).to(self.device)
        
        # Genus-level prediction
        genus_name, genus_conf = self._predict_single_level(cranium_t, teeth_t, self.genus_resources)
        
        # Species-level prediction if a corresponding model exists
        if genus_name in self.species_resources:
            species_name, species_conf = self._predict_single_level(cranium_t, teeth_t, self.species_resources[genus_name])
            return {
                "predicted_genus": genus_name, "genus_confidence": genus_conf,
                "final_prediction": species_name, "final_confidence": species_conf
            }
        else:
            return {
                "predicted_genus": genus_name, "genus_confidence": genus_conf,
                "final_prediction": genus_name, "final_confidence": genus_conf
            }