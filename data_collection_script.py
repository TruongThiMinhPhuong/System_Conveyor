#!/usr/bin/env python3
"""
Enhanced Data Collection Script for Fruit Sorting System
Collects diverse images (angles, lighting) with quality checks
Target: 200+ images per class
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
from pathlib import Path
import json
from hardware.camera import Camera

class EnhancedDataCollector:
    """Enhanced data collection with quality assurance"""

    def __init__(self, output_dir="raw_images", target_per_class=200):
        self.output_dir = Path(output_dir)
        self.target_per_class = target_per_class

        # Create directories
        self.fresh_dir = self.output_dir / "fresh"
        self.spoiled_dir = self.output_dir / "spoiled"

        self.fresh_dir.mkdir(parents=True, exist_ok=True)
        self.spoiled_dir.mkdir(parents=True, exist_ok=True)

        # Quality thresholds
        self.min_brightness = 50
        self.max_brightness = 200
        self.min_sharpness = 100
        self.min_contrast = 30

        # Camera settings
        self.camera = None
        self.resolution = (640, 480)

        # Statistics
        self.stats = {
            'fresh_collected': 0,
            'spoiled_collected': 0,
            'rejected': 0,
            'start_time': datetime.now().isoformat()
        }

    def initialize_camera(self):
        """Initialize camera with optimal settings"""
        self.camera = Camera(resolution=self.resolution)

        if not self.camera.initialize():
            raise RuntimeError("Cannot access camera")

        print("✅ Camera ready")

    def calculate_brightness(self, image):
        """Calculate average brightness of image"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return np.mean(hsv[:, :, 2])

    def calculate_sharpness(self, image):
        """Calculate image sharpness using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def calculate_contrast(self, image):
        """Calculate image contrast"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray.std()

    def check_image_quality(self, image):
        """Check if image meets quality standards"""
        brightness = self.calculate_brightness(image)
        sharpness = self.calculate_sharpness(image)
        contrast = self.calculate_contrast(image)

        checks = {
            'brightness': brightness >= self.min_brightness and brightness <= self.max_brightness,
            'sharpness': sharpness >= self.min_sharpness,
            'contrast': contrast >= self.min_contrast
        }

        quality_score = sum(checks.values()) / len(checks)

        return quality_score >= 0.8, {
            'brightness': round(brightness, 1),
            'sharpness': round(sharpness, 1),
            'contrast': round(contrast, 1),
            'score': round(quality_score, 2)
        }

    def capture_image(self):
        """Capture single image from camera"""
        frame = self.camera.capture_frame()
        if frame is None:
            return None

        # Convert RGB to BGR for OpenCV compatibility
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Flip if needed (depending on camera orientation)
        frame = cv2.flip(frame, 1)

        return frame

    def save_image(self, image, class_name, timestamp):
        """Save image with timestamp and quality info"""
        class_dir = self.fresh_dir if class_name == 'fresh' else self.spoiled_dir

        filename = f"{class_name}_{timestamp}.jpg"
        filepath = class_dir / filename

        # Save with high quality
        cv2.imwrite(str(filepath), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return filepath

    def get_collection_status(self):
        """Get current collection status"""
        fresh_count = len(list(self.fresh_dir.glob("*.jpg")))
        spoiled_count = len(list(self.spoiled_dir.glob("*.jpg")))

        return {
            'fresh': fresh_count,
            'spoiled': spoiled_count,
            'total': fresh_count + spoiled_count,
            'fresh_remaining': max(0, self.target_per_class - fresh_count),
            'spoiled_remaining': max(0, self.target_per_class - spoiled_count)
        }

    def collect_data(self):
        """Main data collection loop (headless version)"""
        print("🍎 Enhanced Fruit Data Collection (Headless Mode)")
        print("=" * 55)
        print(f"Target: {self.target_per_class} images per class")
        print("Quality checks: brightness, sharpness, contrast")
        print("Commands: 'c' to capture, 'q' to quit")
        print()

        try:
            self.initialize_camera()

            while True:
                status = self.get_collection_status()

                if status['fresh_remaining'] == 0 and status['spoiled_remaining'] == 0:
                    print("🎉 Target reached! Collection complete.")
                    break

                # Display current status
                print(f"\n📊 Status: Fresh: {status['fresh']}/{self.target_per_class} | Spoiled: {status['spoiled']}/{self.target_per_class}")

                # Capture image
                print("📷 Capturing image...")
                frame = self.capture_image()
                if frame is None:
                    print("❌ Failed to capture image")
                    continue

                # Quality check
                is_good, quality_info = self.check_image_quality(frame)

                print("🔍 Quality Analysis:")
                print(f"   Brightness: {quality_info['brightness']} (target: {self.min_brightness}-{self.max_brightness})")
                print(f"   Sharpness: {quality_info['sharpness']} (min: {self.min_sharpness})")
                print(f"   Contrast: {quality_info['contrast']} (min: {self.min_contrast})")
                print(f"   Overall Score: {quality_info['score']} {'✅ GOOD' if is_good else '❌ POOR'}")

                if not is_good:
                    print("⚠️  Poor quality image rejected - try adjusting lighting/camera position")
                    self.stats['rejected'] += 1

                    # Ask if user wants to save anyway
                    while True:
                        try:
                            choice = input("Save anyway? (y/n): ").strip().lower()
                            if choice == 'y':
                                break
                            elif choice == 'n':
                                continue
                            else:
                                print("Please enter 'y' or 'n'")
                        except KeyboardInterrupt:
                            return
                else:
                    print("✅ Image quality acceptable")

                # Ask for class
                print("\nSelect class:")
                print("1. Fresh (tươi)")
                print("2. Spoiled (hỏng)")
                print("q. Quit")

                while True:
                    try:
                        choice = input("Enter 1, 2, or q: ").strip()
                        if choice == '1':
                            class_name = 'fresh'
                            break
                        elif choice == '2':
                            class_name = 'spoiled'
                            break
                        elif choice == 'q':
                            return
                        else:
                            print("Please enter 1, 2, or q")
                    except KeyboardInterrupt:
                        return

                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filepath = self.save_image(frame, class_name, timestamp)

                if class_name == 'fresh':
                    self.stats['fresh_collected'] += 1
                else:
                    self.stats['spoiled_collected'] += 1

                print(f"💾 Saved {class_name} image: {filepath.name}")
                print(f"   Quality score: {quality_info['score']}")

                # Ask for next action
                while True:
                    try:
                        cmd = input("\nNext action - 'c' capture next, 'q' quit: ").strip().lower()
                        if cmd == 'c':
                            break
                        elif cmd == 'q':
                            return
                        else:
                            print("Please enter 'c' or 'q'")
                    except KeyboardInterrupt:
                        return

        finally:
            if self.camera:
                self.camera.close()

            # Save statistics
            self.stats['end_time'] = datetime.now().isoformat()
            with open(self.output_dir / 'collection_stats.json', 'w') as f:
                json.dump(self.stats, f, indent=2)

            print("\n📊 Collection Summary:")
            print(f"   Fresh images: {self.stats['fresh_collected']}")
            print(f"   Spoiled images: {self.stats['spoiled_collected']}")
            print(f"   Rejected: {self.stats['rejected']}")
            print(f"   Total: {self.stats['fresh_collected'] + self.stats['spoiled_collected']}")

def main():
    collector = EnhancedDataCollector(target_per_class=250)  # Extra buffer
    collector.collect_data()

if __name__ == "__main__":
    main()
