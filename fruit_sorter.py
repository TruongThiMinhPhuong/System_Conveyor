"""
Main System Orchestration - Development of a Conveyor System for Fruit Quality Classification Using AI Camera
Raspberry Pi 4 - YOLOv8 + MobileNetV2
Run this file to start the fruit sorting system
"""

import time
import signal
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from hardware import ConveyorSystem
from ai_models import YOLODetector, MobileNetClassifier, ImagePreprocessor
from utils import Config, SystemLogger, PerformanceMonitor, PerformanceTimer


class FruitSortingSystem:
    """
    Main fruit sorting system controller
    Integrates hardware and AI models
    """
    
    def __init__(self):
        """Initialize system"""
        self.logger = SystemLogger()
        self.logger.system_event("Initializing Fruit Sorting System...")
        
        # Create necessary directories
        Config.create_directories()
        
        # Initialize components
        self.conveyor = None
        self.detector = None
        self.classifier = None
        self.preprocessor = None
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor(window_size=100)
        
        self.is_running = False
        self.consecutive_errors = 0
        
    def initialize(self) -> bool:
        """
        Initialize all system components
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check models exist
            yolo_ok, mobilenet_ok = Config.validate_models()
            
            if not yolo_ok:
                self.logger.error("YOLO model not found. Train first or use demo mode.")
            
            if not mobilenet_ok:
                self.logger.error("MobileNetV2 model not found. Train first.")
                self.logger.system_event("⚠️ Running in DETECTION ONLY mode (no classification)")
            
            if not yolo_ok:
                self.logger.error("At least YOLO model is required. Cannot proceed.")
                return False
            
            # Initialize hardware
            self.logger.system_event("Initializing hardware...")
            self.conveyor = ConveyorSystem()
            if not self.conveyor.initialize():
                self.logger.error("Failed to initialize hardware")
                return False
            
            # Initialize AI models
            self.logger.system_event("Loading AI models...")
            
            # YOLO Detector
            if yolo_ok:
                self.detector = YOLODetector(
                    model_path=Config.YOLO_MODEL_PATH,
                    confidence_threshold=Config.YOLO_CONFIDENCE_THRESHOLD
                )
                if not self.detector.load_model():
                    self.logger.error("Failed to load YOLO model")
                    return False
            
            # MobileNetV2 Classifier
            if mobilenet_ok:
                self.classifier = MobileNetClassifier(
                    model_path=Config.MOBILENET_MODEL_PATH,
                    input_size=Config.MOBILENET_INPUT_SIZE,
                    class_names=Config.FRESHNESS_CLASSES
                )
                if not self.classifier.load_model():
                    self.logger.error("Failed to load MobileNetV2 model")
                    return False
            
            # Image Preprocessor
            self.preprocessor = ImagePreprocessor(
                target_size=(Config.MOBILENET_INPUT_SIZE, Config.MOBILENET_INPUT_SIZE),
                blur_kernel=Config.BLUR_KERNEL_SIZE,
                fast_mode=Config.FAST_PREPROCESSING
            )
            
            self.logger.system_event("✅ System initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error("System initialization failed", e)
            return False
    
    def process_frame(self, frame):
        """
        Process single frame: detect → classify → sort
        
        Args:
            frame: Captured camera frame
            
        Returns:
            Processing result dictionary
        """
        try:
            # Record frame for FPS tracking
            self.perf_monitor.record_frame()
            
            with PerformanceTimer() as total_timer:
                # Step 1: YOLO Detection
                with PerformanceTimer(self.perf_monitor, 'yolo') as yolo_timer:
                    detections = self.detector.detect(frame, verbose=False)
                
                if not detections:
                    return {'detected': False}
                
                # Get highest confidence detection
                detection = max(detections, key=lambda x: x['confidence'])
                
                self.logger.detection(
                    detection['class_name'],
                    detection['confidence']
                )
                
                # Step 2: Extract ROI and preprocess
                bbox = detection['bbox']
                
                with PerformanceTimer(self.perf_monitor, 'preprocessing'):
                    preprocessed_roi = self.preprocessor.preprocess_complete_pipeline(
                        frame, bbox
                    )
                
                if preprocessed_roi is None:
                    self.logger.error("ROI extraction failed")
                    return {'detected': True, 'classified': False}
                
                # Step 3: MobileNetV2 Classification
                with PerformanceTimer(self.perf_monitor, 'mobilenet'):
                    classification = self.classifier.classify_with_details(preprocessed_roi)
                
                self.logger.classification(
                    classification['predicted_class'],
                    classification['confidence'],
                    classification['is_fresh']
                )
                
                # Only sort if confidence is above threshold
                if classification['confidence'] >= Config.CLASSIFICATION_THRESHOLD:
                    is_fresh = classification['is_fresh']
                else:
                    # Default to fresh for low-confidence classifications
                    self.logger.system_event(
                        f"Low confidence ({classification['confidence']:.2%}), "
                        "defaulting to fresh"
                    )
                    is_fresh = True
            
            # Record total processing time
            self.perf_monitor.record_total_time(total_timer.duration)
            
            return {
                'detected': True,
                'classified': True,
                'detection': detection,
                'classification': classification,
                'is_fresh': is_fresh,
                'processing_time': total_timer.duration
            }
            
        except Exception as e:
            self.logger.error("Frame processing failed", e)
            self.perf_monitor.record_error()
            return {'error': str(e)}
    
    def run(self):
        """
        Main system loop
        """
        if not self.is_running:
            self.logger.error("System not running")
            return
        
        self.logger.system_event("🚀 Starting main system loop...")
        
        # Start conveyor
        self.conveyor.start_conveyor(Config.CONVEYOR_SPEED_DETECTION)
        
        frame_count = 0
        last_stats_time = time.time()
        
        try:
            while self.is_running:
                try:
                    # Capture frame
                    frame = self.conveyor.capture_image()
                    
                    if frame is None:
                        self.logger.error("Failed to capture frame")
                        self.consecutive_errors += 1
                        
                        if self.consecutive_errors >= Config.MAX_CONSECUTIVE_ERRORS:
                            self.logger.error("Too many consecutive errors, stopping")
                            break
                        
                        time.sleep(0.1)
                        continue
                    
                    # Reset error count on successful capture
                    self.consecutive_errors = 0
                    
                    # Process frame
                    result = self.process_frame(frame)
                    
                    if result.get('classified'):
                        # Sort fruit based on freshness
                        is_fresh = result['is_fresh']
                        self.logger.sorting(is_fresh)
                        self.conveyor.sort_fruit(is_fresh=is_fresh, pause_conveyor=True)
                    
                    frame_count += 1
                    
                    # Print statistics periodically
                    if time.time() - last_stats_time >= Config.STATS_UPDATE_INTERVAL:
                        self.logger.print_statistics()
                        self.perf_monitor.print_stats()
                        last_stats_time = time.time()
                    
                    # Control FPS
                    if Config.MAX_FPS > 0:
                        time.sleep(1.0 / Config.MAX_FPS)
                    else:
                        time.sleep(Config.DETECTION_INTERVAL)
                    
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    self.logger.error("Error in main loop", e)
                    self.consecutive_errors += 1
                    
                    if self.consecutive_errors >= Config.MAX_CONSECUTIVE_ERRORS:
                        self.logger.error("Too many errors, stopping")
                        break
        
        except KeyboardInterrupt:
            self.logger.system_event("Interrupted by user")
        
        finally:
            self.stop()
    
    def start(self):
        """Start the system"""
        if not self.initialize():
            self.logger.error("Cannot start system - initialization failed")
            return False
        
        self.is_running = True
        self.run()
        return True
    
    def stop(self):
        """Stop the system"""
        self.logger.system_event("Stopping system...")
        self.is_running = False
        
        if self.conveyor:
            self.conveyor.cleanup()
        
        self.logger.print_statistics()
        self.logger.system_event("✅ System stopped")
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop()


def signal_handler(sig, frame):
    """Handle interrupt signals"""
    print("\n🛑 Interrupt received, shutting down...")
    sys.exit(0)


def main():
    """Main entry point"""
    print("=" * 60)
    print("🍎 Development of a Conveyor System for Fruit Quality Classification Using AI Camera")
    print("=" * 60)
    
    # Print configuration
    Config.print_config()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run system
    system = FruitSortingSystem()
    
    try:
        system.start()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        system.cleanup()


if __name__ == '__main__':
    main()
