#!/usr/bin/env python3
"""
Dataset Quality Analysis Tool for Fruit Sorting System
Analyzes image quality, diversity, and potential issues
"""

import cv2
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
from sklearn.cluster import KMeans
from scipy import stats

class DatasetQualityAnalyzer:
    """Analyze dataset quality and provide recommendations"""

    def __init__(self, dataset_path="raw_images"):
        self.dataset_path = Path(dataset_path)
        self.classes = ['fresh', 'spoiled']
        self.analysis_results = {}

    def analyze_image_quality(self, image_path):
        """Analyze single image quality metrics"""
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Basic metrics
        height, width = image.shape[:2]

        # Brightness (V channel in HSV)
        brightness = np.mean(hsv[:, :, 2])

        # Contrast (standard deviation of grayscale)
        contrast = gray.std()

        # Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Colorfulness (Hasler and Süsstrunk)
        rg = image[:, :, 0] - image[:, :, 1]
        yb = 0.5 * (image[:, :, 0] + image[:, :, 1]) - image[:, :, 2]
        colorfulness = np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)

        # Saturation (S channel in HSV)
        saturation = np.mean(hsv[:, :, 1])

        # Blur detection (using FFT)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1e-8)
        blur_score = np.var(magnitude_spectrum)

        return {
            'width': width,
            'height': height,
            'brightness': round(brightness, 2),
            'contrast': round(contrast, 2),
            'sharpness': round(sharpness, 2),
            'colorfulness': round(colorfulness, 2),
            'saturation': round(saturation, 2),
            'blur_score': round(blur_score, 2),
            'aspect_ratio': round(width / height, 3)
        }

    def analyze_class_distribution(self):
        """Analyze class distribution and balance"""
        distribution = {}

        for class_name in self.classes:
            class_path = self.dataset_path / class_name
            if class_path.exists():
                images = list(class_path.glob("*.jpg"))
                distribution[class_name] = len(images)
            else:
                distribution[class_name] = 0

        total_images = sum(distribution.values())
        balance_ratio = min(distribution.values()) / max(distribution.values()) if max(distribution.values()) > 0 else 0

        return {
            'distribution': distribution,
            'total_images': total_images,
            'balance_ratio': round(balance_ratio, 3),
            'minority_class': min(distribution, key=distribution.get),
            'majority_class': max(distribution, key=distribution.get)
        }

    def analyze_image_diversity(self):
        """Analyze image diversity using clustering"""
        features = []

        for class_name in self.classes:
            class_path = self.dataset_path / class_name
            if not class_path.exists():
                continue

            for image_path in class_path.glob("*.jpg"):
                quality = self.analyze_image_quality(image_path)
                if quality:
                    # Extract key features for clustering
                    features.append([
                        quality['brightness'],
                        quality['contrast'],
                        quality['sharpness'],
                        quality['colorfulness'],
                        quality['saturation']
                    ])

        if len(features) < 10:
            return {'error': 'Insufficient data for diversity analysis'}

        features = np.array(features)

        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)

        # K-means clustering to find diversity
        n_clusters = min(5, len(features) // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_normalized)

        # Calculate cluster distribution
        unique, counts = np.unique(clusters, return_counts=True)
        cluster_distribution = dict(zip(unique, counts))

        # Diversity score (normalized entropy)
        entropy = stats.entropy(counts)
        max_entropy = np.log(n_clusters)
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0

        return {
            'total_images_analyzed': len(features),
            'n_clusters': n_clusters,
            'cluster_distribution': cluster_distribution,
            'diversity_score': round(diversity_score, 3),
            'cluster_centers': kmeans.cluster_centers_.tolist()
        }

    def detect_outliers(self):
        """Detect outlier images that may need removal"""
        outliers = defaultdict(list)

        for class_name in self.classes:
            class_path = self.dataset_path / class_name
            if not class_path.exists():
                continue

            qualities = []
            image_paths = []

            for image_path in class_path.glob("*.jpg"):
                quality = self.analyze_image_quality(image_path)
                if quality:
                    qualities.append([
                        quality['brightness'],
                        quality['contrast'],
                        quality['sharpness']
                    ])
                    image_paths.append(image_path)

            if len(qualities) < 5:
                continue

            qualities = np.array(qualities)

            # Use IQR method to detect outliers
            for i, feature_name in enumerate(['brightness', 'contrast', 'sharpness']):
                Q1 = np.percentile(qualities[:, i], 25)
                Q3 = np.percentile(qualities[:, i], 75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_indices = np.where((qualities[:, i] < lower_bound) | (qualities[:, i] > upper_bound))[0]

                for idx in outlier_indices:
                    outliers[f'{class_name}_{feature_name}'].append({
                        'image': str(image_paths[idx].name),
                        'value': round(qualities[idx, i], 2),
                        'bounds': [round(lower_bound, 2), round(upper_bound, 2)]
                    })

        return dict(outliers)

    def generate_quality_report(self):
        """Generate comprehensive quality report"""
        print("🔍 Analyzing Dataset Quality")
        print("=" * 50)

        # Class distribution
        distribution = self.analyze_class_distribution()
        print("📊 Class Distribution:")
        for class_name, count in distribution['distribution'].items():
            print(f"   {class_name}: {count} images")
        print(f"   Balance ratio: {distribution['balance_ratio']}")
        print(f"   Total: {distribution['total_images']} images")

        # Quality thresholds and recommendations
        quality_thresholds = {
            'min_images_per_class': 200,
            'min_balance_ratio': 0.5,
            'min_brightness': 50,
            'max_brightness': 200,
            'min_contrast': 30,
            'min_sharpness': 100
        }

        recommendations = []

        if distribution['total_images'] < quality_thresholds['min_images_per_class'] * len(self.classes):
            recommendations.append(f"❌ Need more data: {quality_thresholds['min_images_per_class'] * len(self.classes)} images minimum")

        if distribution['balance_ratio'] < quality_thresholds['min_balance_ratio']:
            minority = distribution['minority_class']
            needed = distribution['distribution'][distribution['majority_class']] - distribution['distribution'][minority]
            recommendations.append(f"⚠️  Class imbalance: Collect {needed} more {minority} images")

        # Analyze diversity if enough data
        if distribution['total_images'] >= 50:
            diversity = self.analyze_image_diversity()
            if 'error' not in diversity:
                print("\n🌈 Image Diversity:")
                print(f"   Diversity score: {diversity['diversity_score']} (0-1, higher is better)")
                print(f"   Clusters found: {diversity['n_clusters']}")

                if diversity['diversity_score'] < 0.6:
                    recommendations.append("⚠️  Low image diversity: Capture images with different angles, lighting, and backgrounds")

        # Check for outliers
        outliers = self.detect_outliers()
        if outliers:
            total_outliers = sum(len(outlier_list) for outlier_list in outliers.values())
            print(f"\n🚨 Potential Outliers: {total_outliers} images")
            recommendations.append(f"🔍 Review {total_outliers} outlier images for quality issues")

        # Overall assessment
        print("\n📋 Quality Assessment:")
        if not recommendations:
            print("✅ Dataset quality looks good!")
        else:
            print("⚠️  Areas for improvement:")
            for rec in recommendations:
                print(f"   {rec}")

        # Save detailed report
        report = {
            'distribution': distribution,
            'diversity': diversity if 'diversity' in locals() else None,
            'outliers': outliers,
            'recommendations': recommendations,
            'quality_thresholds': quality_thresholds,
            'timestamp': str(np.datetime64('now'))
        }

        with open('dataset_quality_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print("\n📄 Detailed report saved to: dataset_quality_report.json")
        return report

def main():
    analyzer = DatasetQualityAnalyzer()
    analyzer.generate_quality_report()

if __name__ == "__main__":
    main()
