"""
Paper/Handwriting Validator Module

Validates that uploaded images contain actual handwriting on paper,
rejecting irrelevant images (photos of trees, objects, memes, etc.)

Detection Methods:
1. Document Detection - Checks for paper-like properties (light background, rectangular)
2. Text/Writing Presence - Detects handwritten characters using contour analysis
3. Edge Density Analysis - Handwriting has specific edge patterns
4. Color Distribution - Papers have distinct color histograms
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import os


class PaperValidator:
    """Validates images to ensure they contain handwriting on paper."""

    def __init__(self):
        # Thresholds (tuned for handwriting detection)
        self.MIN_WHITE_RATIO = 0.30  # At least 30% light background
        self.MAX_WHITE_RATIO = 0.98  # Not completely blank
        self.MIN_TEXT_CONTOURS = 5   # At least 5 character-like shapes
        self.MIN_EDGE_DENSITY = 0.01  # Some edges from writing
        self.MAX_EDGE_DENSITY = 0.40  # Not too cluttered (nature photos)
        self.MIN_ASPECT_CHARS = 3     # At least 3 valid character aspects

    def validate(self, image_path: str) -> Dict:
        """
        Validate if image contains handwriting on paper.

        Returns:
            {
                'is_valid': bool,
                'confidence': float (0-100),
                'reason': str,
                'details': dict
            }
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {
                'is_valid': False,
                'confidence': 0,
                'reason': 'Could not read image file',
                'details': {}
            }

        # Run all checks
        checks = {
            'background': self._check_background(img),
            'text_presence': self._check_text_presence(img),
            'edge_density': self._check_edge_density(img),
            'color_distribution': self._check_color_distribution(img),
            'character_shapes': self._check_character_shapes(img),
        }

        # Calculate overall validity and confidence
        passed_checks = sum(1 for c in checks.values() if c['passed'])
        total_checks = len(checks)
        confidence = (passed_checks / total_checks) * 100

        # Determine validity (need at least 3 of 5 checks to pass)
        is_valid = passed_checks >= 3

        # Generate reason
        if is_valid:
            reason = 'Image appears to contain handwriting on paper'
        else:
            failed = [k for k, v in checks.items() if not v['passed']]
            reasons_map = {
                'background': 'No paper-like background detected',
                'text_presence': 'No handwriting/text detected',
                'edge_density': 'Edge pattern does not match handwriting',
                'color_distribution': 'Color distribution does not match paper',
                'character_shapes': 'No character-like shapes found',
            }
            reason = reasons_map.get(failed[0], 'Image does not appear to be handwriting on paper')

        return {
            'is_valid': is_valid,
            'confidence': round(confidence, 1),
            'reason': reason,
            'details': checks
        }

    def _check_background(self, img: np.ndarray) -> Dict:
        """Check if image has paper-like light background."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate ratio of light pixels (paper is usually white/light)
        light_threshold = 200
        light_pixels = np.sum(gray > light_threshold)
        total_pixels = gray.size
        white_ratio = light_pixels / total_pixels

        # Also check for medium-light pixels (slightly off-white paper)
        medium_light = np.sum(gray > 150) / total_pixels

        passed = (self.MIN_WHITE_RATIO <= medium_light <= self.MAX_WHITE_RATIO)

        return {
            'passed': passed,
            'white_ratio': round(white_ratio, 3),
            'medium_light_ratio': round(medium_light, 3),
            'message': f'Light background: {medium_light*100:.1f}%'
        }

    def _check_text_presence(self, img: np.ndarray) -> Dict:
        """Detect presence of text/handwriting using contour analysis."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding (better for handwriting)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter for text-like contours (small-medium size, proper aspect ratio)
        img_area = img.shape[0] * img.shape[1]
        min_area = img_area * 0.00005  # Min 0.005% of image
        max_area = img_area * 0.05     # Max 5% of image (individual chars)

        text_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / h if h > 0 else 0
                # Characters typically have aspect ratio between 0.2 and 3
                if 0.15 < aspect < 4:
                    text_contours.append(cnt)

        passed = len(text_contours) >= self.MIN_TEXT_CONTOURS

        return {
            'passed': passed,
            'text_contour_count': len(text_contours),
            'total_contours': len(contours),
            'message': f'Text elements found: {len(text_contours)}'
        }

    def _check_edge_density(self, img: np.ndarray) -> Dict:
        """Check edge density - handwriting has specific edge patterns."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Calculate edge density
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        edge_density = edge_pixels / total_pixels

        # Handwriting has moderate edge density
        passed = self.MIN_EDGE_DENSITY < edge_density < self.MAX_EDGE_DENSITY

        return {
            'passed': passed,
            'edge_density': round(edge_density, 4),
            'message': f'Edge density: {edge_density*100:.2f}%'
        }

    def _check_color_distribution(self, img: np.ndarray) -> Dict:
        """Check if color distribution matches paper with writing."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()

        # Paper with writing should have:
        # - Peak in light values (paper background)
        # - Some dark values (ink/writing)

        light_region = hist[180:256].sum()  # Light pixels
        dark_region = hist[0:80].sum()       # Dark pixels (writing)
        mid_region = hist[80:180].sum()      # Mid tones

        # Good paper: significant light + some dark, not too much mid
        has_paper = light_region > 0.2      # At least 20% light (paper)
        has_ink = dark_region > 0.01        # At least 1% dark (ink)
        not_photo = mid_region < 0.7        # Photos have lots of mid-tones

        passed = has_paper and has_ink and not_photo

        return {
            'passed': passed,
            'light_ratio': round(light_region, 3),
            'dark_ratio': round(dark_region, 3),
            'mid_ratio': round(mid_region, 3),
            'message': f'Light: {light_region*100:.1f}%, Dark: {dark_region*100:.1f}%'
        }

    def _check_character_shapes(self, img: np.ndarray) -> Dict:
        """Detect character-like shapes using morphological analysis."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # Filter for character-like components
        img_area = img.shape[0] * img.shape[1]
        char_like = 0

        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]

            # Character-like properties
            if area < img_area * 0.0001:  # Too small (noise)
                continue
            if area > img_area * 0.1:     # Too large (not a char)
                continue

            aspect = width / height if height > 0 else 0
            if 0.1 < aspect < 5:  # Reasonable aspect ratio for chars
                char_like += 1

        passed = char_like >= self.MIN_ASPECT_CHARS

        return {
            'passed': passed,
            'character_like_shapes': char_like,
            'total_components': num_labels - 1,
            'message': f'Character-like shapes: {char_like}'
        }


def validate_paper_image(image_path: str) -> Dict:
    """
    Convenience function to validate a paper image.

    Args:
        image_path: Path to the image file

    Returns:
        Validation result dictionary
    """
    validator = PaperValidator()
    return validator.validate(image_path)


def is_valid_handwriting_paper(image_path: str) -> Tuple[bool, str]:
    """
    Simple validation check.

    Returns:
        (is_valid, reason)
    """
    result = validate_paper_image(image_path)
    return result['is_valid'], result['reason']


# Quick test if run directly
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        result = validate_paper_image(sys.argv[1])
        print(f"Valid: {result['is_valid']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Reason: {result['reason']}")
        print(f"Details: {result['details']}")
