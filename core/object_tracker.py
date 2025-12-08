"""
Theo dõi vị trí trái cây trên băng chuyền
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import time
import logging
from collections import deque

from core.fruit_detector import FruitDetection
from config import system_config, hardware_config

logger = logging.getLogger(__name__)

class TrackedFruit:
    """Lưu thông tin theo dõi một trái cây"""
    
    def __init__(self, detection: FruitDetection, track_id: int):
        """
        Khởi tạo tracked fruit
        
        Args:
            detection: Phát hiện ban đầu
            track_id: ID duy nhất
        """
        self.track_id = track_id
        self.detections = deque(maxlen=system_config.TRACKING_BUFFER_SIZE)
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.class_name = detection.class_name
        self.is_good = detection.is_good
        self.is_classified = False
        self.classification_time: Optional[float] = None
        
        # Thêm detection đầu tiên
        detection.track_id = track_id
        self.detections.append(detection)
    
    def update(self, detection: FruitDetection):
        """Cập nhật với detection mới"""
        detection.track_id = self.track_id
        self.detections.append(detection)
        self.last_seen = time.time()
        
        # Cập nhật class name nếu confidence cao hơn
        if detection.confidence > 0.8:
            self.class_name = detection.class_name
            self.is_good = detection.is_good
    
    def get_current_position(self) -> Optional[Tuple[float, float]]:
        """Lấy vị trí hiện tại (tâm)"""
        if not self.detections:
            return None
        return self.detections[-1].centroid
    
    def get_estimated_position(self) -> Optional[Tuple[float, float]]:
        """Ước tính vị trí dựa trên tốc độ băng chuyền"""
        if len(self.detections) < 2:
            return self.get_current_position()
        
        # Tính vận tốc dựa trên 2 điểm cuối
        latest = self.detections[-1]
        prev = self.detections[-2]
        
        dt = latest.timestamp - prev.timestamp
        if dt <= 0:
            return latest.centroid
        
        # Vận tốc pixel/giây
        vx = (latest.centroid[0] - prev.centroid[0]) / dt
        vy = (latest.centroid[1] - prev.centroid[1]) / dt
        
        # Thời gian từ lần seen cuối
        time_since_last = time.time() - latest.timestamp
        
        # Ước tính vị trí hiện tại
        estimated_x = latest.centroid[0] + vx * time_since_last
        estimated_y = latest.centroid[1] + vy * time_since_last
        
        return (estimated_x, estimated_y)
    
    def calculate_speed(self) -> float:
        """Tính tốc độ di chuyển (pixel/giây)"""
        if len(self.detections) < 2:
            return 0
        
        distances = []
        timestamps = []
        
        for i in range(1, len(self.detections)):
            prev = self.detections[i-1]
            curr = self.detections[i]
            
            # Tính khoảng cách Euclidean
            dist = np.sqrt((curr.centroid[0] - prev.centroid[0])**2 + 
                          (curr.centroid[1] - prev.centroid[1])**2)
            
            distances.append(dist)
            timestamps.append(curr.timestamp - prev.timestamp)
        
        if sum(timestamps) == 0:
            return 0
        
        total_distance = sum(distances)
        total_time = sum(timestamps)
        
        return total_distance / total_time if total_time > 0 else 0

class ObjectTracker:
    """Theo dõi nhiều trái cây trên băng chuyền"""
    
    def __init__(self):
        """Khởi tạo tracker"""
        self.tracks: Dict[int, TrackedFruit] = {}
        self.next_track_id = 1
        self.max_disappeared = 5  # Số frame mất tích tối đa
        self.iou_threshold = 0.3  # Ngưỡng để gán detection cho track
        
        # Các track đã hoàn thành phân loại
        self.completed_tracks = []
        
        # Định thời
        self.last_update_time = time.time()
        
    def update(self, detections: List[FruitDetection]) -> List[TrackedFruit]:
        """
        Cập nhật tracks với detections mới
        
        Args:
            detections: Danh sách detections mới
            
        Returns:
            Danh sách active tracks
        """
        current_time = time.time()
        
        # Nếu không có detection, đánh dấu tất cả tracks là disappeared
        if not detections:
            self._mark_disappeared()
            return list(self.tracks.values())
        
        # Tạo ma trận IoU giữa tracks hiện tại và detections mới
        if self.tracks:
            track_ids = list(self.tracks.keys())
            track_positions = [t.get_current_position() for t in self.tracks.values()]
            
            # Chỉ giữ tracks có vị trí hợp lệ
            valid_indices = [i for i, pos in enumerate(track_positions) if pos is not None]
            valid_track_ids = [track_ids[i] for i in valid_indices]
            valid_positions = [track_positions[i] for i in valid_indices]
            
            if valid_positions:
                # Tính IoU hoặc khoảng cách
                assignments = self._assign_detections_to_tracks(detections, valid_positions)
                
                # Cập nhật matched tracks
                for det_idx, track_idx in enumerate(assignments):
                    if track_idx is not None:
                        track_id = valid_track_ids[track_idx]
                        self.tracks[track_id].update(detections[det_idx])
                        detections[det_idx].track_id = track_id
        
        # Tạo tracks mới cho detections chưa được gán
        for detection in detections:
            if detection.track_id is None:
                new_track = TrackedFruit(detection, self.next_track_id)
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1
        
        # Đánh dấu disappeared tracks và xóa tracks cũ
        self._cleanup_old_tracks()
        
        # Kiểm tra nếu trái cây đã đến vị trí phân loại
        active_tracks = self._check_classification_positions()
        
        self.last_update_time = current_time
        return active_tracks
    
    def _assign_detections_to_tracks(self, detections: List[FruitDetection],
                                    track_positions: List[Tuple[float, float]]) -> List[Optional[int]]:
        """Gán detections cho tracks dựa trên khoảng cách"""
        assignments = [None] * len(detections)
        
        if not track_positions:
            return assignments
        
        # Tính ma trận khoảng cách
        distance_matrix = np.zeros((len(detections), len(track_positions)))
        
        for i, det in enumerate(detections):
            for j, track_pos in enumerate(track_positions):
                if track_pos is not None:
                    dist = np.sqrt((det.centroid[0] - track_pos[0])**2 + 
                                  (det.centroid[1] - track_pos[1])**2)
                    distance_matrix[i, j] = dist
        
        # Greedy assignment based on distance
        max_distance = 100  # pixel
        
        for i in range(len(detections)):
            min_dist = np.min(distance_matrix[i])
            if min_dist < max_distance:
                j = np.argmin(distance_matrix[i])
                assignments[i] = j
                # Đặt hàng và cột thành inf để tránh gán trùng
                distance_matrix[:, j] = np.inf
                distance_matrix[i, :] = np.inf
        
        return assignments
    
    def _mark_disappeared(self):
        """Đánh dấu tracks là disappeared"""
        current_time = time.time()
        for track in self.tracks.values():
            if current_time - track.last_seen > 0.5:  # 0.5 giây
                # Có thể thêm logic disappeared counter ở đây
                pass
    
    def _cleanup_old_tracks(self):
        """Xóa tracks cũ"""
        current_time = time.time()
        tracks_to_delete = []
        
        for track_id, track in self.tracks.items():
            # Xóa track nếu không thấy trong 2 giây
            if current_time - track.last_seen > 2.0:
                tracks_to_delete.append(track_id)
        
        for track_id in tracks_to_delete:
            if track_id in self.tracks:
                del self.tracks[track_id]
    
    def _check_classification_positions(self) -> List[TrackedFruit]:
        """
        Kiểm tra nếu trái cây đã đến vị trí cần phân loại
        
        Returns:
            Danh sách tracks cần phân loại
        """
        classification_y = 400  # Vị trí Y để phân loại (pixel)
        
        tracks_to_classify = []
        
        for track_id, track in self.tracks.items():
            # Chỉ phân loại nếu chưa được phân loại
            if not track.is_classified:
                pos = track.get_current_position()
                if pos and pos[1] >= classification_y:  # Đã đến vị trí phân loại
                    track.is_classified = True
                    track.classification_time = time.time()
                    tracks_to_classify.append(track)
                    
                    # Di chuyển sang completed tracks
                    self.completed_tracks.append(track)
        
        return tracks_to_classify
    
    def get_track_count(self) -> int:
        """Lấy số lượng tracks đang active"""
        return len(self.tracks)
    
    def get_completed_count(self) -> int:
        """Lấy số lượng tracks đã hoàn thành"""
        return len(self.completed_tracks)
    
    def get_stats(self) -> Dict:
        """Lấy thống kê tracker"""
        return {
            "active_tracks": len(self.tracks),
            "completed_tracks": len(self.completed_tracks),
            "next_track_id": self.next_track_id,
            "avg_track_lifetime": self._calculate_avg_lifetime()
        }
    
    def _calculate_avg_lifetime(self) -> float:
        """Tính thời gian sống trung bình của tracks"""
        if not self.completed_tracks:
            return 0
        
        lifetimes = []
        for track in self.completed_tracks:
            if track.classification_time and track.first_seen:
                lifetimes.append(track.classification_time - track.first_seen)
        
        return sum(lifetimes) / len(lifetimes) if lifetimes else 0