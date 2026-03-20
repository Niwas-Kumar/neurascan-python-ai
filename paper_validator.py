"""
Paper/Handwriting Validator Module - V2 (PyTorch + TensorFlow Support)

Two-stage validation for maximum accuracy:
1. ML Model: Trained document classifier (MobileNetV2-based)
   - Supports PyTorch (RVL-CDIP trained) OR TensorFlow
2. Rule-Based: Strict heuristic checks as backup

BOTH must pass for image to be accepted.
"""

import cv2
import numpy as np
from typing import Dict, Tuple
import os
import json

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# PyTorch model paths (V2 - RVL-CDIP trained)
PYTORCH_MODEL_PATH = os.path.join(MODEL_DIR, 'document_classifier_pytorch.pth')
PYTORCH_TRACED_PATH = os.path.join(MODEL_DIR, 'document_classifier_traced.pt')

# TensorFlow model paths (V1 - fallback)
TF_MODEL_KERAS = os.path.join(MODEL_DIR, 'document_classifier.keras')
TF_MODEL_H5 = os.path.join(MODEL_DIR, 'document_classifier.h5')

# Config
CONFIG_PATH = os.path.join(MODEL_DIR, 'document_classifier_config.json')

# Global model cache
_model = None
_config = None
_framework = None


def load_document_classifier():
    """Load the trained document classifier model (PyTorch preferred, TensorFlow fallback)."""
    global _model, _config, _framework

    if _model is not None:
        return _model, _config, _framework

    # Load config
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            _config = json.load(f)
    else:
        _config = {'IMG_SIZE': 224, 'threshold': 0.5}

    # Try PyTorch first (better model from RVL-CDIP)
    if os.path.exists(PYTORCH_TRACED_PATH) or os.path.exists(PYTORCH_MODEL_PATH):
        try:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if os.path.exists(PYTORCH_TRACED_PATH):
                _model = torch.jit.load(PYTORCH_TRACED_PATH, map_location=device)
                print(f"[DocumentClassifier] Loaded PyTorch TorchScript model")
            else:
                # Load from checkpoint
                from torchvision.models import mobilenet_v2
                import torch.nn as nn

                checkpoint = torch.load(PYTORCH_MODEL_PATH, map_location=device)

                # Recreate model architecture
                base = mobilenet_v2(weights=None)
                num_features = base.classifier[1].in_features
                base.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(num_features, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.2),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )
                base.load_state_dict(checkpoint['model_state_dict'])
                _model = base.to(device)
                _model.eval()

                if 'threshold' in checkpoint:
                    _config['threshold'] = checkpoint['threshold']

                print(f"[DocumentClassifier] Loaded PyTorch checkpoint model")

            _framework = 'pytorch'
            _config['device'] = str(device)
            return _model, _config, _framework

        except Exception as e:
            print(f"[DocumentClassifier] PyTorch load failed: {e}")
            _model = None

    # Fallback to TensorFlow
    if os.path.exists(TF_MODEL_KERAS) or os.path.exists(TF_MODEL_H5):
        try:
            import tensorflow as tf
            from tensorflow import keras

            if os.path.exists(TF_MODEL_KERAS):
                _model = keras.models.load_model(TF_MODEL_KERAS)
                print("[DocumentClassifier] Loaded TensorFlow .keras model")
            else:
                _model = keras.models.load_model(TF_MODEL_H5)
                print("[DocumentClassifier] Loaded TensorFlow .h5 model")

            _framework = 'tensorflow'
            return _model, _config, _framework

        except Exception as e:
            print(f"[DocumentClassifier] TensorFlow load failed: {e}")
            _model = None

    print("[DocumentClassifier] No model found, using rule-based only")
    return None, _config, None


class MLDocumentClassifier:
    """ML-based document/paper classifier supporting PyTorch and TensorFlow."""

    def __init__(self):
        self.model, self.config, self.framework = load_document_classifier()
        self.img_size = self.config.get('IMG_SIZE', 224)
        self.threshold = self.config.get('threshold', 0.5)

    def classify(self, image_path: str) -> Dict:
        """
        Classify if image is a valid test paper.

        Returns:
            {
                'is_valid': bool,
                'confidence': float (0-100),
                'raw_score': float (0-1),
                'framework': str
            }
        """
        if self.model is None:
            return {'is_valid': None, 'confidence': 0, 'raw_score': 0, 'available': False}

        try:
            if self.framework == 'pytorch':
                return self._classify_pytorch(image_path)
            elif self.framework == 'tensorflow':
                return self._classify_tensorflow(image_path)
            else:
                return {'is_valid': None, 'confidence': 0, 'raw_score': 0, 'available': False}

        except Exception as e:
            print(f"[MLDocumentClassifier] Error: {e}")
            return {'is_valid': None, 'confidence': 0, 'raw_score': 0, 'available': False, 'error': str(e)}

    def _classify_pytorch(self, image_path: str) -> Dict:
        """Classify using PyTorch model."""
        import torch
        from torchvision import transforms
        from PIL import Image

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Preprocessing (same as training)
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            raw_score = self.model(img_tensor).squeeze().cpu().item()

        # Score > threshold means valid (class 1 = valid_paper)
        is_valid = raw_score > self.threshold
        confidence = raw_score * 100 if is_valid else (1 - raw_score) * 100

        return {
            'is_valid': is_valid,
            'confidence': round(confidence, 1),
            'raw_score': round(raw_score, 4),
            'threshold': self.threshold,
            'framework': 'pytorch',
            'available': True
        }

    def _classify_tensorflow(self, image_path: str) -> Dict:
        """Classify using TensorFlow model."""
        from tensorflow import keras
        import numpy as np

        img = keras.preprocessing.image.load_img(
            image_path,
            target_size=(self.img_size, self.img_size)
        )
        img_array = keras.preprocessing.image.img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        raw_score = float(self.model.predict(img_batch, verbose=0)[0][0])

        # For TensorFlow model: score < threshold means document (class 0 = document)
        is_valid = raw_score < self.threshold
        confidence = (1 - raw_score) * 100 if is_valid else raw_score * 100

        return {
            'is_valid': is_valid,
            'confidence': round(confidence, 1),
            'raw_score': round(raw_score, 4),
            'threshold': self.threshold,
            'framework': 'tensorflow',
            'available': True
        }


class StrictRuleBasedValidator:
    """Rule-based validation using image analysis heuristics."""

    def __init__(self):
        self.MIN_WHITE_RATIO = 0.35
        self.MAX_SATURATION = 45
        self.MIN_TEXT_REGIONS = 3

    def validate(self, img: np.ndarray) -> Dict:
        """Run all rule-based checks."""
        checks = {
            'background_uniformity': self._check_background_uniformity(img),
            'text_line_structure': self._check_text_line_structure(img),
            'color_analysis': self._check_color_analysis(img),
            'document_structure': self._check_document_structure(img),
            'texture_complexity': self._check_texture_complexity(img),
        }

        passed_checks = sum(1 for c in checks.values() if c['passed'])
        is_valid = passed_checks >= 3
        confidence = (passed_checks / len(checks)) * 100

        return {
            'is_valid': is_valid,
            'confidence': round(confidence, 1),
            'passed_checks': passed_checks,
            'total_checks': len(checks),
            'details': checks
        }

    def _check_background_uniformity(self, img: np.ndarray) -> Dict:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flat = gray.flatten()
        threshold = np.percentile(flat, 60)
        background_pixels = flat[flat >= threshold]
        bg_variance = np.var(background_pixels)
        bg_mean = np.mean(background_pixels)
        white_ratio = np.sum(gray > 200) / gray.size

        passed = bg_mean > 170 and bg_variance < 1000 and white_ratio > self.MIN_WHITE_RATIO

        return {
            'passed': passed,
            'bg_mean': round(float(bg_mean), 1),
            'bg_variance': round(float(bg_variance), 1),
            'white_ratio': round(white_ratio, 3),
            'message': f'Background: mean={bg_mean:.0f}, var={bg_variance:.0f}'
        }

    def _check_text_line_structure(self, img: np.ndarray) -> Dict:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        horizontal_proj = np.sum(binary, axis=1)

        if horizontal_proj.max() > 0:
            horizontal_proj = horizontal_proj / horizontal_proj.max()

        above_threshold = horizontal_proj > 0.1
        transitions = np.diff(above_threshold.astype(int))
        line_starts = np.sum(transitions == 1)

        peak_indices = np.where(above_threshold)[0]
        has_structure = len(peak_indices) > 10 and np.sum(np.diff(peak_indices) > 5) >= 2

        passed = line_starts >= 2 and has_structure

        return {'passed': passed, 'detected_lines': int(line_starts), 'message': f'Text lines: {line_starts}'}

    def _check_color_analysis(self, img: np.ndarray) -> Dict:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        mean_saturation = np.mean(saturation)
        very_light = np.sum(value > 200) / value.size
        very_dark = np.sum(value < 80) / value.size
        mid_tones = np.sum((value >= 80) & (value <= 200)) / value.size

        passed = mean_saturation < self.MAX_SATURATION and very_light > 0.25 and very_dark > 0.005 and mid_tones < 0.6

        return {
            'passed': passed,
            'mean_saturation': round(float(mean_saturation), 1),
            'message': f'Sat: {mean_saturation:.0f}, Light: {very_light*100:.1f}%'
        }

    def _check_document_structure(self, img: np.ndarray) -> Dict:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        img_area = img.shape[0] * img.shape[1]

        text_like = sum(1 for i in range(1, num_labels)
                       if 0.00001 < stats[i, cv2.CC_STAT_AREA] / img_area < 0.02
                       and 0.1 < stats[i, cv2.CC_STAT_WIDTH] / max(stats[i, cv2.CC_STAT_HEIGHT], 1) < 10)

        return {'passed': text_like >= self.MIN_TEXT_REGIONS, 'text_components': text_like, 'message': f'Text components: {text_like}'}

    def _check_texture_complexity(self, img: np.ndarray) -> Dict:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = laplacian.var()
        intensity_range = gray.max() - gray.min()
        normalized_complexity = texture_variance / (intensity_range ** 2) if intensity_range > 0 else 0

        return {'passed': normalized_complexity < 0.6, 'complexity': round(normalized_complexity, 4), 'message': f'Texture: {normalized_complexity:.4f}'}


class CombinedPaperValidator:
    """Combined ML + Rule-based validator for maximum accuracy."""

    def __init__(self):
        self.ml_classifier = MLDocumentClassifier()
        self.rule_validator = StrictRuleBasedValidator()

    def validate(self, image_path: str) -> Dict:
        """Validate image using both ML and rule-based methods."""
        img = cv2.imread(image_path)
        if img is None:
            return {'is_valid': False, 'confidence': 0, 'reason': 'Could not read image file', 'stage_failed': 'load'}

        result = {'is_valid': False, 'confidence': 0, 'reason': '', 'ml_result': None, 'rule_result': None}

        # Stage 1: ML Classification
        ml_result = self.ml_classifier.classify(image_path)
        result['ml_result'] = ml_result

        if ml_result.get('available') and not ml_result.get('is_valid'):
            result['confidence'] = ml_result['confidence']
            result['reason'] = f"AI detected this is NOT a valid test paper (confidence: {ml_result['confidence']:.1f}%). Upload a clear photo of handwritten work on paper."
            result['stage_failed'] = 'ml_classifier'
            return result

        # Stage 2: Rule-Based Validation
        rule_result = self.rule_validator.validate(img)
        result['rule_result'] = rule_result

        if not rule_result['is_valid']:
            failed = [k for k, v in rule_result['details'].items() if not v['passed']]
            reasons = {
                'background_uniformity': 'Background is not uniform like paper',
                'text_line_structure': 'No text lines detected',
                'color_analysis': 'Colors too vibrant for paper/ink',
                'document_structure': 'No text content found',
                'texture_complexity': 'Image texture too complex',
            }
            result['confidence'] = rule_result['confidence']
            result['reason'] = reasons.get(failed[0], 'Not a valid document') if failed else 'Validation failed'
            result['stage_failed'] = 'rule_validation'
            return result

        # Both passed
        ml_conf = ml_result.get('confidence', 80) if ml_result.get('available') else 80
        result['is_valid'] = True
        result['confidence'] = round((ml_conf + rule_result['confidence']) / 2, 1)
        result['reason'] = 'Image validated as test paper with handwriting'
        return result


def validate_paper_image(image_path: str) -> Dict:
    """Main validation function."""
    validator = CombinedPaperValidator()
    result = validator.validate(image_path)
    return {
        'is_valid': result['is_valid'],
        'confidence': result['confidence'],
        'reason': result['reason'],
        'details': {'ml_classifier': result.get('ml_result', {}), 'rule_validation': result.get('rule_result', {})}
    }


def is_valid_test_paper(image_path: str) -> Tuple[bool, str]:
    """Simple validation returning (is_valid, reason)."""
    result = validate_paper_image(image_path)
    return result['is_valid'], result['reason']


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        result = validate_paper_image(sys.argv[1])
        print(f"\n{'='*50}")
        print(f"VALID: {result['is_valid']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Reason: {result['reason']}")
        print(f"{'='*50}")
