"""
Điều khiển động cơ băng chuyền và servo phân loại
"""

import RPi.GPIO as GPIO
import time
import threading
import logging
from typing import Optional
from queue import Queue, Empty

from config import hardware_config

logger = logging.getLogger(__name__)

class MotorController:
    """Điều khiển động cơ DC và servo"""
    
    def __init__(self):
        """Khởi tạo GPIO và PWM"""
        self.gpio_initialized = False
        self.motor_pwm = None
        self.servo_pwm = None
        self.is_running = False
        
        # Hàng đợi lệnh servo
        self.servo_queue = Queue()
        self.servo_thread = None
        
        # Trạng thái hiện tại
        self.current_motor_speed = 0
        self.current_servo_angle = hardware_config.SERVO_ANGLE_PASS
        self.conveyor_running = False
        
        # Thống kê
        self.total_fruits_processed = 0
        self.good_fruits = 0
        self.bad_fruits = 0
        
    def initialize(self) -> bool:
        """Khởi tạo GPIO và PWM"""
        try:
            # Cảnh báo GPIO
            GPIO.setwarnings(False)
            
            # Chế độ BCM
            GPIO.setmode(GPIO.BCM)
            
            # Thiết lập chân động cơ
            GPIO.setup(hardware_config.MOTOR_IN1_PIN, GPIO.OUT)
            GPIO.setup(hardware_config.MOTOR_IN2_PIN, GPIO.OUT)
            GPIO.setup(hardware_config.MOTOR_ENA_PIN, GPIO.OUT)
            
            # Thiết lập chân servo
            GPIO.setup(hardware_config.SERVO_PIN, GPIO.OUT)
            
            # Thiết lập chân cảm biến (nếu có)
            GPIO.setup(hardware_config.SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            # Khởi tạo PWM cho động cơ
            self.motor_pwm = GPIO.PWM(hardware_config.MOTOR_ENA_PIN, 
                                     hardware_config.MOTOR_PWM_FREQ)
            self.motor_pwm.start(0)  # Bắt đầu với duty cycle 0%
            
            # Khởi tạo PWM cho servo
            self.servo_pwm = GPIO.PWM(hardware_config.SERVO_PIN, 
                                     hardware_config.SERVO_PWM_FREQ)
            self.servo_pwm.start(0)
            
            # Đặt servo về vị trí ban đầu
            self._set_servo_angle(hardware_config.SERVO_ANGLE_PASS)
            
            # Khởi động luồng xử lý servo
            self.is_running = True
            self.servo_thread = threading.Thread(target=self._servo_worker, daemon=True)
            self.servo_thread.start()
            
            self.gpio_initialized = True
            logger.info("Motor controller initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize motor controller: {e}")
            return False
    
    def _angle_to_duty_cycle(self, angle: float) -> float:
        """Chuyển góc sang duty cycle cho servo"""
        # Servo SG90: 0° = 2.5%, 90° = 7.5%, 180° = 12.5%
        # Điều chỉnh cho servo của bạn nếu cần
        min_duty = 2.5
        max_duty = 12.5
        duty = min_duty + (angle / 180.0) * (max_duty - min_duty)
        return duty
    
    def _set_servo_angle(self, angle: float):
        """Đặt góc cho servo (chỉ dùng nội bộ)"""
        try:
            angle = max(0, min(180, angle))  # Giới hạn góc
            duty_cycle = self._angle_to_duty_cycle(angle)
            self.servo_pwm.ChangeDutyCycle(duty_cycle)
            self.current_servo_angle = angle
            time.sleep(0.05)  # Thời gian ổn định
            self.servo_pwm.ChangeDutyCycle(0)  # Dừng xung để tránh rung
        except Exception as e:
            logger.error(f"Error setting servo angle: {e}")
    
    def _servo_worker(self):
        """Luồng xử lý lệnh servo"""
        while self.is_running:
            try:
                # Lấy lệnh từ queue
                command = self.servo_queue.get(timeout=0.1)
                
                if command == "good":
                    self._set_servo_angle(hardware_config.SERVO_ANGLE_PASS)
                    logger.debug("Servo: Good fruit - let pass")
                    
                elif command == "bad":
                    self._set_servo_angle(hardware_config.SERVO_ANGLE_REJECT)
                    logger.debug("Servo: Bad fruit - push to reject")
                    
                    # Giữ góc đẩy trong 0.2 giây
                    time.sleep(0.2)
                    
                    # Trở về vị trí ban đầu
                    self._set_servo_angle(hardware_config.SERVO_ANGLE_PASS)
                    
                elif isinstance(command, (int, float)):
                    # Đặt góc cụ thể
                    self._set_servo_angle(float(command))
                    
                # Đánh dấu task hoàn thành
                self.servo_queue.task_done()
                
            except Empty:
                # Queue trống, tiếp tục
                pass
            except Exception as e:
                logger.error(f"Error in servo worker: {e}")
    
    def start_conveyor(self, speed: Optional[int] = None):
        """Bắt đầu chạy băng chuyền"""
        if not self.gpio_initialized:
            logger.error("Motor controller not initialized")
            return
        
        try:
            if speed is None:
                speed = hardware_config.MOTOR_SPEED_NORMAL
            
            # Đặt chiều quay (tiến)
            GPIO.output(hardware_config.MOTOR_IN1_PIN, GPIO.HIGH)
            GPIO.output(hardware_config.MOTOR_IN2_PIN, GPIO.LOW)
            
            # Đặt tốc độ
            speed = max(0, min(100, speed))
            self.motor_pwm.ChangeDutyCycle(speed)
            self.current_motor_speed = speed
            self.conveyor_running = True
            
            logger.info(f"Conveyor started at {speed}% speed")
            
        except Exception as e:
            logger.error(f"Error starting conveyor: {e}")
    
    def stop_conveyor(self):
        """Dừng băng chuyền"""
        try:
            self.motor_pwm.ChangeDutyCycle(0)
            self.conveyor_running = False
            logger.info("Conveyor stopped")
        except Exception as e:
            logger.error(f"Error stopping conveyor: {e}")
    
    def set_motor_speed(self, speed: int):
        """Điều chỉnh tốc độ động cơ"""
        try:
            speed = max(0, min(100, speed))
            self.motor_pwm.ChangeDutyCycle(speed)
            self.current_motor_speed = speed
            logger.debug(f"Motor speed set to {speed}%")
        except Exception as e:
            logger.error(f"Error setting motor speed: {e}")
    
    def classify_fruit(self, is_good: bool):
        """
        Phân loại trái cây và điều khiển servo
        
        Args:
            is_good: True nếu trái cây tốt, False nếu xấu
        """
        if not self.gpio_initialized:
            logger.error("Motor controller not initialized")
            return
        
        # Cập nhật thống kê
        self.total_fruits_processed += 1
        if is_good:
            self.good_fruits += 1
            command = "good"
        else:
            self.bad_fruits += 1
            command = "bad"
        
        # Thêm lệnh vào queue
        try:
            self.servo_queue.put(command, timeout=0.5)
            logger.info(f"Fruit classified: {'GOOD' if is_good else 'BAD'} "
                       f"(Total: {self.total_fruits_processed})")
        except Exception as e:
            logger.error(f"Error queuing servo command: {e}")
    
    def check_sensor(self) -> bool:
        """Kiểm tra cảm biến (nếu có)"""
        if not self.gpio_initialized:
            return False
        
        try:
            # Cảm biến IR: LOW khi có vật cản
            return GPIO.input(hardware_config.SENSOR_PIN) == GPIO.LOW
        except:
            return False
    
    def get_status(self) -> dict:
        """Lấy trạng thái hiện tại"""
        return {
            "conveyor_running": self.conveyor_running,
            "motor_speed": self.current_motor_speed,
            "servo_angle": self.current_servo_angle,
            "total_fruits": self.total_fruits_processed,
            "good_fruits": self.good_fruits,
            "bad_fruits": self.bad_fruits,
            "queue_size": self.servo_queue.qsize()
        }
    
    def cleanup(self):
        """Dọn dẹp GPIO"""
        self.is_running = False
        
        if self.servo_thread:
            self.servo_thread.join(timeout=2)
        
        try:
            self.stop_conveyor()
            
            if self.motor_pwm:
                self.motor_pwm.stop()
            
            if self.servo_pwm:
                self.servo_pwm.stop()
            
            GPIO.cleanup()
            logger.info("Motor controller cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up motor controller: {e}")