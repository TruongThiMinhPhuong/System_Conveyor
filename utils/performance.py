"""
Performance Monitoring and Optimization Utilities
Track and analyze system performance in real-time
"""

import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional
from pathlib import Path
import json


class PerformanceMonitor:
    """
    Track and analyze processing performance
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize performance monitor
        
        Args:
            window_size: Number of samples to keep for rolling average
        """
        self.window_size = window_size
        
        # Timing deques for rolling averages
        self.yolo_times = deque(maxlen=window_size)
        self.mobilenet_times = deque(maxlen=window_size)
        self.preprocessing_times = deque(maxlen=window_size)
        self.total_times = deque(maxlen=window_size)
        
        # Frame timing
        self.frame_times = deque(maxlen=window_size)
        self.last_frame_time = None
        
        # Counters
        self.total_frames = 0
        self.processed_frames = 0
        self.skipped_frames = 0
        self.errors = 0
        
        # Start time
        self.start_time = time.time()
    
    def record_yolo_time(self, duration: float):
        """Record YOLO inference time"""
        self.yolo_times.append(duration)
    
    def record_mobilenet_time(self, duration: float):
        """Record MobileNet inference time"""
        self.mobilenet_times.append(duration)
    
    def record_preprocessing_time(self, duration: float):
        """Record preprocessing time"""
        self.preprocessing_times.append(duration)
    
    def record_total_time(self, duration: float):
        """Record total processing time"""
        self.total_times.append(duration)
        self.processed_frames += 1
    
    def record_frame(self):
        """Record frame timing for FPS calculation"""
        current_time = time.time()
        if self.last_frame_time is not None:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
        self.last_frame_time = current_time
        self.total_frames += 1
    
    def record_skip(self):
        """Record skipped frame"""
        self.skipped_frames += 1
    
    def record_error(self):
        """Record processing error"""
        self.errors += 1
    
    def get_stats(self) -> Dict:
        """
        Get current performance statistics
        
        Returns:
            Dictionary of performance metrics
        """
        stats = {
            'total_frames': self.total_frames,
            'processed_frames': self.processed_frames,
            'skipped_frames': self.skipped_frames,
            'errors': self.errors,
            'uptime': time.time() - self.start_time
        }
        
        # Average times
        if self.yolo_times:
            stats['avg_yolo_time'] = np.mean(self.yolo_times)
            stats['max_yolo_time'] = np.max(self.yolo_times)
            stats['min_yolo_time'] = np.min(self.yolo_times)
        
        if self.mobilenet_times:
            stats['avg_mobilenet_time'] = np.mean(self.mobilenet_times)
            stats['max_mobilenet_time'] = np.max(self.mobilenet_times)
            stats['min_mobilenet_time'] = np.min(self.mobilenet_times)
        
        if self.preprocessing_times:
            stats['avg_preprocessing_time'] = np.mean(self.preprocessing_times)
            stats['max_preprocessing_time'] = np.max(self.preprocessing_times)
            stats['min_preprocessing_time'] = np.min(self.preprocessing_times)
        
        if self.total_times:
            stats['avg_total_time'] = np.mean(self.total_times)
            stats['max_total_time'] = np.max(self.total_times)
            stats['min_total_time'] = np.min(self.total_times)
        
        # FPS calculation
        if self.frame_times:
            avg_frame_time = np.mean(self.frame_times)
            stats['fps'] = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            stats['avg_frame_time'] = avg_frame_time
        
        # Processing rate
        if stats['uptime'] > 0:
            stats['processing_rate'] = self.processed_frames / stats['uptime']
        
        return stats
    
    def print_stats(self):
        """Print formatted performance statistics"""
        stats = self.get_stats()
        
        print("\n" + "=" * 60)
        print("Performance Statistics")
        print("=" * 60)
        
        # Frame statistics
        print(f"\n📊 Frame Statistics:")
        print(f"   Total frames: {stats['total_frames']}")
        print(f"   Processed: {stats['processed_frames']}")
        print(f"   Skipped: {stats['skipped_frames']}")
        print(f"   Errors: {stats['errors']}")
        print(f"   Uptime: {stats['uptime']:.1f}s")
        
        # FPS
        if 'fps' in stats:
            print(f"\n⚡ Frame Rate:")
            print(f"   FPS: {stats['fps']:.1f}")
            print(f"   Avg frame time: {stats['avg_frame_time']*1000:.1f}ms")
        
        # Processing times
        if 'avg_total_time' in stats:
            print(f"\n⏱️ Processing Times (avg/max/min):")
            print(f"   YOLO: {stats.get('avg_yolo_time', 0)*1000:.1f}ms / "
                  f"{stats.get('max_yolo_time', 0)*1000:.1f}ms / "
                  f"{stats.get('min_yolo_time', 0)*1000:.1f}ms")
            print(f"   MobileNet: {stats.get('avg_mobilenet_time', 0)*1000:.1f}ms / "
                  f"{stats.get('max_mobilenet_time', 0)*1000:.1f}ms / "
                  f"{stats.get('min_mobilenet_time', 0)*1000:.1f}ms")
            print(f"   Preprocessing: {stats.get('avg_preprocessing_time', 0)*1000:.1f}ms / "
                  f"{stats.get('max_preprocessing_time', 0)*1000:.1f}ms / "
                  f"{stats.get('min_preprocessing_time', 0)*1000:.1f}ms")
            print(f"   Total: {stats['avg_total_time']*1000:.1f}ms / "
                  f"{stats['max_total_time']*1000:.1f}ms / "
                  f"{stats['min_total_time']*1000:.1f}ms")
        
        # Processing rate
        if 'processing_rate' in stats:
            print(f"\n📈 Performance:")
            print(f"   Processing rate: {stats['processing_rate']:.2f} frames/sec")
        
        print("=" * 60)
    
    def save_stats(self, filepath: str):
        """
        Save statistics to JSON file
        
        Args:
            filepath: Path to save statistics
        """
        stats = self.get_stats()
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"💾 Performance statistics saved to {filepath}")
    
    def reset(self):
        """Reset all statistics"""
        self.yolo_times.clear()
        self.mobilenet_times.clear()
        self.preprocessing_times.clear()
        self.total_times.clear()
        self.frame_times.clear()
        
        self.total_frames = 0
        self.processed_frames = 0
        self.skipped_frames = 0
        self.errors = 0
        
        self.start_time = time.time()
        self.last_frame_time = None


class PerformanceTimer:
    """
    Context manager for timing code blocks
    """
    
    def __init__(self, monitor: Optional[PerformanceMonitor] = None, operation: str = ""):
        """
        Initialize timer
        
        Args:
            monitor: PerformanceMonitor to record to
            operation: Name of operation being timed
        """
        self.monitor = monitor
        self.operation = operation
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        """Start timing"""
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        """Stop timing and record"""
        self.duration = time.time() - self.start_time
        
        if self.monitor and self.operation:
            if self.operation == 'yolo':
                self.monitor.record_yolo_time(self.duration)
            elif self.operation == 'mobilenet':
                self.monitor.record_mobilenet_time(self.duration)
            elif self.operation == 'preprocessing':
                self.monitor.record_preprocessing_time(self.duration)
            elif self.operation == 'total':
                self.monitor.record_total_time(self.duration)


# Test code
if __name__ == '__main__':
    print("=== Performance Monitor Test ===\n")
    
    monitor = PerformanceMonitor(window_size=10)
    
    # Simulate some processing
    for i in range(20):
        monitor.record_frame()
        
        with PerformanceTimer(monitor, 'yolo'):
            time.sleep(0.05)  # Simulate 50ms YOLO
        
        with PerformanceTimer(monitor, 'preprocessing'):
            time.sleep(0.01)  # Simulate 10ms preprocessing
        
        with PerformanceTimer(monitor, 'mobilenet'):
            time.sleep(0.03)  # Simulate 30ms MobileNet
        
        with PerformanceTimer(monitor, 'total'):
            time.sleep(0.09)  # Total time
    
    # Print statistics
    monitor.print_stats()
    
    print("\n✅ Performance monitor test complete!")
