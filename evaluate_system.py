"""
Đánh Giá Độ Chính Xác Hệ Thống
Evaluation script cho Raspberry Pi
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from ai_models import YOLODetector, MobileNetClassifier, ImagePreprocessor
from utils import Config, PerformanceMonitor


class SystemEvaluator:
    """
    Đánh giá toàn diện độ chính xác của hệ thống
    """
    
    def __init__(self, output_dir="evaluation_results"):
        """
        Initialize evaluator
        
        Args:
            output_dir: Thư mục lưu kết quả
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load models
        self.detector = YOLODetector(
            model_path=Config.YOLO_MODEL_PATH,
            confidence_threshold=Config.YOLO_CONFIDENCE_THRESHOLD
        )
        self.classifier = MobileNetClassifier(
            model_path=Config.MOBILENET_MODEL_PATH
        )
        self.preprocessor = ImagePreprocessor(
            target_size=(Config.MOBILENET_INPUT_SIZE, Config.MOBILENET_INPUT_SIZE),
            blur_kernel=Config.BLUR_KERNEL_SIZE,
            fast_mode=Config.FAST_PREPROCESSING
        )
        
        # Performance monitor
        self.perf_monitor = PerformanceMonitor()
        
        # Statistics
        self.stats = defaultdict(int)
        self.results = []
        
    def load_models(self):
        """Load AI models"""
        print("📦 Loading models...")
        
        if not self.detector.load_model():
            raise Exception("Failed to load YOLO model")
        
        if not self.classifier.load_model():
            raise Exception("Failed to load MobileNet model")
        
        print("✅ Models loaded successfully!")
        
    def evaluate_single_image(self, image_path, ground_truth_label):
        """
        Đánh giá 1 ảnh
        
        Args:
            image_path: Đường dẫn ảnh
            ground_truth_label: Nhãn thực tế ('fresh' hoặc 'spoiled')
            
        Returns:
            Dict kết quả đánh giá
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return {'error': 'Cannot load image'}
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        result = {
            'image': str(image_path),
            'ground_truth': ground_truth_label,
            'timestamp': datetime.now().isoformat()
        }
        
        # Step 1: YOLO Detection
        start_time = time.time()
        detections = self.detector.detect(image_rgb, verbose=False)
        yolo_time = time.time() - start_time
        result['yolo_time_ms'] = yolo_time * 1000
        
        if not detections:
            result['detected'] = False
            result['correct'] = False
            self.stats['detection_failed'] += 1
            return result
        
        result['detected'] = True
        detection = max(detections, key=lambda x: x['confidence'])
        result['detection_confidence'] = detection['confidence']
        result['detection_class'] = detection['class_name']
        
        # Step 2: Preprocessing
        start_time = time.time()
        bbox = detection['bbox']
        preprocessed = self.preprocessor.preprocess_complete_pipeline(image_rgb, bbox)
        prep_time = time.time() - start_time
        result['preprocessing_time_ms'] = prep_time * 1000
        
        if preprocessed is None:
            result['classified'] = False
            result['correct'] = False
            self.stats['preprocessing_failed'] += 1
            return result
        
        # Step 3: Classification
        start_time = time.time()
        classification = self.classifier.classify_with_details(preprocessed)
        class_time = time.time() - start_time
        result['classification_time_ms'] = class_time * 1000
        
        result['classified'] = True
        result['predicted_class'] = classification['predicted_class']
        result['classification_confidence'] = classification['confidence']
        result['is_fresh'] = classification['is_fresh']
        
        # Total time
        result['total_time_ms'] = (yolo_time + prep_time + class_time) * 1000
        
        # Correctness check
        predicted = 'fresh' if classification['is_fresh'] else 'spoiled'
        result['correct'] = (predicted == ground_truth_label)
        
        # Update stats
        if result['correct']:
            self.stats['correct'] += 1
        else:
            self.stats['incorrect'] += 1
            
        self.stats['total'] += 1
        
        return result
    
    def evaluate_dataset(self, test_dir, batch_size=10, save_interval=50, max_images=None):
        """
        Đánh giá toàn bộ dataset test với batch processing và progress tracking

        Args:
            test_dir: Thư mục chứa ảnh test
                      test_dir/fresh/*.jpg
                      test_dir/spoiled/*.jpg
            batch_size: Số ảnh xử lý mỗi batch
            save_interval: Lưu kết quả sau mỗi N ảnh
            max_images: Giới hạn số ảnh xử lý (None = tất cả)
        """
        test_path = Path(test_dir)

        print(f"\n{'='*60}")
        print(f"Đánh Giá Dataset: {test_dir}")
        print(f"{'='*60}\n")

        # Collect images
        fresh_images = list((test_path / 'fresh').glob('*.[jp][pn][g]*'))
        spoiled_images = list((test_path / 'spoiled').glob('*.[jp][pn][g]*'))

        print(f"📊 Dataset:")
        print(f"   Fresh: {len(fresh_images)} images")
        print(f"   Spoiled: {len(spoiled_images)} images")
        print(f"   Total: {len(fresh_images) + len(spoiled_images)} images")
        print(f"   Batch size: {batch_size}")
        print(f"   Save interval: {save_interval}")

        # Combine all images with labels
        all_images = [(img, 'fresh') for img in fresh_images] + [(img, 'spoiled') for img in spoiled_images]

        # Limit images if specified
        if max_images and len(all_images) > max_images:
            print(f"   Limited to: {max_images} images (quick test mode)")
            all_images = all_images[:max_images]

        total_images = len(all_images)

        # Progress tracking
        processed = 0
        start_time = time.time()

        print(f"\n🚀 Starting evaluation of {total_images} images...")

        try:
            # Process in batches
            for i in range(0, total_images, batch_size):
                batch = all_images[i:i+batch_size]
                batch_start = time.time()

                print(f"\n📦 Processing batch {i//batch_size + 1}/{(total_images + batch_size - 1)//batch_size}")
                print(f"   Images {i+1}-{min(i+batch_size, total_images)} of {total_images}")

                for img_path, label in batch:
                    try:
                        result = self.evaluate_single_image(img_path, label)
                        self.results.append(result)
                        processed += 1

                        # Progress update every 5 images
                        if processed % 5 == 0:
                            elapsed = time.time() - start_time
                            rate = processed / elapsed if elapsed > 0 else 0
                            remaining = (total_images - processed) / rate if rate > 0 else 0
                            print(f"   📊 Progress: {processed}/{total_images} "
                                  f"({processed/total_images*100:.1f}%) - "
                                  f"{rate:.1f} img/s - ETA: {remaining:.0f}s")

                    except KeyboardInterrupt:
                        print(f"\n⚠️  KeyboardInterrupt detected at image {processed+1}")
                        raise
                    except Exception as e:
                        print(f"   ❌ Error processing {img_path}: {e}")
                        self.stats['processing_errors'] += 1
                        continue

                batch_time = time.time() - batch_start
                print(f"   ✅ Batch completed in {batch_time:.1f}s")

                # Save intermediate results
                if processed % save_interval == 0 and processed > 0:
                    print(f"   💾 Saving intermediate results...")
                    self.calculate_metrics()
                    self.save_results(intermediate=True)

            # Final calculation and save
            print(f"\n🎯 Evaluation completed! Processed {processed}/{total_images} images")
            self.calculate_metrics()
            self.save_results()

        except KeyboardInterrupt:
            print(f"\n⏹️  Evaluation interrupted at {processed}/{total_images} images")
            print("💾 Saving partial results...")
            self.calculate_metrics()
            self.save_results(intermediate=True, interrupted=True)
            raise
        
    def calculate_metrics(self):
        """Tính toán các metrics đánh giá"""
        
        # Filter successful classifications
        classified_results = [r for r in self.results if r.get('classified', False)]
        
        if not classified_results:
            print("\n⚠️ No successful classifications!")
            return
        
        # Confusion matrix components
        tp_fresh = sum(1 for r in classified_results 
                      if r['ground_truth'] == 'fresh' and r['is_fresh'])
        tn_spoiled = sum(1 for r in classified_results 
                        if r['ground_truth'] == 'spoiled' and not r['is_fresh'])
        fp_fresh = sum(1 for r in classified_results 
                      if r['ground_truth'] == 'spoiled' and r['is_fresh'])
        fn_spoiled = sum(1 for r in classified_results 
                        if r['ground_truth'] == 'fresh' and not r['is_fresh'])
        
        # Metrics
        total = len(classified_results)
        accuracy = (tp_fresh + tn_spoiled) / total if total > 0 else 0
        
        # Precision & Recall for Fresh
        precision_fresh = tp_fresh / (tp_fresh + fp_fresh) if (tp_fresh + fp_fresh) > 0 else 0
        recall_fresh = tp_fresh / (tp_fresh + fn_spoiled) if (tp_fresh + fn_spoiled) > 0 else 0
        f1_fresh = 2 * (precision_fresh * recall_fresh) / (precision_fresh + recall_fresh) \
                   if (precision_fresh + recall_fresh) > 0 else 0
        
        # Precision & Recall for Spoiled  
        precision_spoiled = tn_spoiled / (tn_spoiled + fn_spoiled) if (tn_spoiled + fn_spoiled) > 0 else 0
        recall_spoiled = tn_spoiled / (tn_spoiled + fp_fresh) if (tn_spoiled + fp_fresh) > 0 else 0
        f1_spoiled = 2 * (precision_spoiled * recall_spoiled) / (precision_spoiled + recall_spoiled) \
                     if (precision_spoiled + recall_spoiled) > 0 else 0
        
        # Average times
        avg_yolo_time = np.mean([r.get('yolo_time_ms', 0) for r in classified_results])
        avg_prep_time = np.mean([r.get('preprocessing_time_ms', 0) for r in classified_results])
        avg_class_time = np.mean([r.get('classification_time_ms', 0) for r in classified_results])
        avg_total_time = np.mean([r.get('total_time_ms', 0) for r in classified_results])
        
        # Average confidences
        avg_det_conf = np.mean([r.get('detection_confidence', 0) for r in classified_results])
        avg_class_conf = np.mean([r.get('classification_confidence', 0) for r in classified_results])
        
        # Store metrics
        self.metrics = {
            'confusion_matrix': {
                'true_positive_fresh': tp_fresh,
                'true_negative_spoiled': tn_spoiled,
                'false_positive_fresh': fp_fresh,
                'false_negative_spoiled': fn_spoiled
            },
            'accuracy': accuracy,
            'fresh': {
                'precision': precision_fresh,
                'recall': recall_fresh,
                'f1_score': f1_fresh
            },
            'spoiled': {
                'precision': precision_spoiled,
                'recall': recall_spoiled,
                'f1_score': f1_spoiled
            },
            'performance': {
                'avg_yolo_time_ms': avg_yolo_time,
                'avg_preprocessing_time_ms': avg_prep_time,
                'avg_classification_time_ms': avg_class_time,
                'avg_total_time_ms': avg_total_time,
                'estimated_fps': 1000 / avg_total_time if avg_total_time > 0 else 0
            },
            'confidences': {
                'avg_detection_confidence': avg_det_conf,
                'avg_classification_confidence': avg_class_conf
            }
        }
        
    def print_results(self):
        """In kết quả đánh giá"""
        
        if not hasattr(self, 'metrics'):
            print("\n⚠️ No metrics calculated yet!")
            return
        
        m = self.metrics
        
        print(f"\n{'='*60}")
        print("📊 KẾT QUẢ ĐÁNH GIÁ HỆ THỐNG")
        print(f"{'='*60}\n")
        
        # Overall stats
        print(f"📈 Tổng Quan:")
        print(f"   Tổng số ảnh: {self.stats['total']}")
        print(f"   Phát hiện thành công: {self.stats['total'] - self.stats.get('detection_failed', 0)}")
        print(f"   Phân loại chính xác: {self.stats['correct']}")
        print(f"   Phân loại sai: {self.stats['incorrect']}")
        
        # Accuracy metrics
        print(f"\n🎯 Độ Chính Xác:")
        print(f"   Overall Accuracy: {m['accuracy']:.2%}")
        
        print(f"\n🍏 Fresh Class:")
        print(f"   Precision: {m['fresh']['precision']:.2%}")
        print(f"   Recall: {m['fresh']['recall']:.2%}")
        print(f"   F1 Score: {m['fresh']['f1_score']:.2%}")
        
        print(f"\n🍎 Spoiled Class:")
        print(f"   Precision: {m['spoiled']['precision']:.2%}")
        print(f"   Recall: {m['spoiled']['recall']:.2%}")
        print(f"   F1 Score: {m['spoiled']['f1_score']:.2%}")
        
        # Confusion Matrix
        cm = m['confusion_matrix']
        print(f"\n📋 Confusion Matrix:")
        print(f"                 Predicted Fresh  |  Predicted Spoiled")
        print(f"   Actual Fresh:     {cm['true_positive_fresh']:3d}         |       {cm['false_negative_spoiled']:3d}")
        print(f"   Actual Spoiled:   {cm['false_positive_fresh']:3d}         |       {cm['true_negative_spoiled']:3d}")
        
        # Performance
        perf = m['performance']
        print(f"\n⚡ Hiệu Năng (Raspberry Pi):")
        print(f"   YOLO Detection: {perf['avg_yolo_time_ms']:.1f}ms")
        print(f"   Preprocessing: {perf['avg_preprocessing_time_ms']:.1f}ms")
        print(f"   Classification: {perf['avg_classification_time_ms']:.1f}ms")
        print(f"   Total: {perf['avg_total_time_ms']:.1f}ms")
        print(f"   Estimated FPS: {perf['estimated_fps']:.1f}")
        
        # Confidences
        conf = m['confidences']
        print(f"\n🔍 Độ Tin Cậy:")
        print(f"   Avg Detection Confidence: {conf['avg_detection_confidence']:.2%}")
        print(f"   Avg Classification Confidence: {conf['avg_classification_confidence']:.2%}")
        
        print(f"\n{'='*60}\n")
        
        # Assessment
        self.print_assessment()
        
    def print_assessment(self):
        """Đánh giá kết quả"""
        
        m = self.metrics
        accuracy = m['accuracy']
        f1_fresh = m['fresh']['f1_score']
        f1_spoiled = m['spoiled']['f1_score']
        fps = m['performance']['estimated_fps']
        
        print("🎓 ĐÁNH GIÁ:")
        print()
        
        # Accuracy assessment
        if accuracy >= 0.95:
            print("   ✅ Accuracy: XUẤT SẮC (≥95%)")
        elif accuracy >= 0.90:
            print("   ✅ Accuracy: TỐT (≥90%)")
        elif accuracy >= 0.85:
            print("   ⚠️  Accuracy: KHÁ (<90%, cần cải thiện)")
        else:
            print("   ❌ Accuracy: THẤP (<85%, cần train lại)")
        
        # F1 score assessment
        avg_f1 = (f1_fresh + f1_spoiled) / 2
        if avg_f1 >= 0.90:
            print("   ✅ F1 Score: TỐT (≥90%)")
        elif avg_f1 >= 0.85:
            print("   ⚠️  F1 Score: KHÁ (<90%)")
        else:
            print("   ❌ F1 Score: THẤP (<85%)")
        
        # Performance assessment
        if fps >= 10:
            print("   ✅ Performance: ĐỦ NHANH (≥10 FPS)")
        elif fps >= 8:
            print("   ⚠️  Performance: CHẤP NHẬN ĐƯỢC (≥8 FPS)")
        else:
            print("   ❌ Performance: QUÁ CHẬM (<8 FPS)")
        
        print()
        
        # Recommendations
        if accuracy < 0.90 or avg_f1 < 0.90:
            print("💡 KHUYẾN NGHỊ CẢI THIỆN:")
            print("   - Thu thập thêm dữ liệu (200+ ảnh/loại)")
            print("   - Đảm bảo ảnh đa dạng (góc độ, ánh sáng)")
            print("   - Train lại với epochs cao hơn")
            print("   - Kiểm tra quality dataset")
        
        if fps < 10:
            print("💡 KHUYẾN NGHỊ TỐI ƯU:")
            print("   - Giảm CAMERA_RESOLUTION xuống 320x320")
            print("   - Set FAST_PREPROCESSING = True")
            print("   - Kiểm tra XNNPACK delegate")
            print("   - Xem xét dùng Pi 5 hoặc Coral TPU")
    
    def save_results(self, intermediate=False, interrupted=False):
        """Lưu kết quả ra file"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_intermediate" if intermediate else ("_interrupted" if interrupted else "")

        # Save detailed results
        results_file = self.output_dir / f"evaluation_{timestamp}{suffix}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'intermediate': intermediate,
                    'interrupted': interrupted,
                    'total_images_processed': len(self.results)
                },
                'metrics': getattr(self, 'metrics', {}),
                'stats': dict(self.stats),
                'results': self.results
            }, f, indent=2, ensure_ascii=False)

        print(f"💾 Kết quả đã lưu: {results_file}")

        # Save summary report only for final results
        if not intermediate and hasattr(self, 'metrics'):
            report_file = self.output_dir / f"report_{timestamp}{suffix}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("BÁO CÁO ĐÁNH GIÁ HỆ THỐNG\n")
                f.write("="*60 + "\n\n")
                f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if intermediate:
                    f.write("TRẠNG THÁI: KẾT QUẢ TẠM THỜI\n\n")
                elif interrupted:
                    f.write("TRẠNG THÁI: ĐÃ DỪNG (INTERRUPTED)\n\n")
                else:
                    f.write("TRẠNG THÁI: HOÀN THÀNH\n\n")
                f.write(f"Số ảnh đã xử lý: {len(self.results)}\n")
                f.write(f"Accuracy: {self.metrics.get('accuracy', 0):.2%}\n")
                f.write(f"F1 Fresh: {self.metrics.get('fresh', {}).get('f1_score', 0):.2%}\n")
                f.write(f"F1 Spoiled: {self.metrics.get('spoiled', {}).get('f1_score', 0):.2%}\n")
                f.write(f"Avg FPS: {self.metrics.get('performance', {}).get('estimated_fps', 0):.1f}\n")

            print(f"📄 Báo cáo đã lưu: {report_file}")


def main():
    """Main evaluation function"""
    import argparse

    parser = argparse.ArgumentParser(description="Đánh giá độ chính xác hệ thống")
    parser.add_argument('--test_dir', type=str, required=True,
                       help="Thư mục chứa ảnh test (có subfolder fresh/ và spoiled/)")
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help="Thư mục lưu kết quả")
    parser.add_argument('--batch_size', type=int, default=10,
                       help="Số ảnh xử lý mỗi batch (default: 10)")
    parser.add_argument('--save_interval', type=int, default=50,
                       help="Lưu kết quả sau mỗi N ảnh (default: 50)")
    parser.add_argument('--quick_test', action='store_true',
                       help="Chạy test nhanh với ít ảnh (20 ảnh đầu tiên)")

    args = parser.parse_args()

    # Create evaluator
    evaluator = SystemEvaluator(output_dir=args.output)

    # Load models
    evaluator.load_models()

    # Run evaluation
    try:
        if args.quick_test:
            print("\n🧪 QUICK TEST MODE: Chỉ xử lý 20 ảnh đầu tiên")
            evaluator.evaluate_dataset(args.test_dir, batch_size=5, save_interval=10, max_images=20)
        else:
            evaluator.evaluate_dataset(args.test_dir, args.batch_size, args.save_interval)

        # Print results
        evaluator.print_results()

    except KeyboardInterrupt:
        print("\n⏹️  Đã dừng đánh giá theo yêu cầu người dùng")
        print("Kết quả tạm thời đã được lưu tự động")


if __name__ == "__main__":
    main()
