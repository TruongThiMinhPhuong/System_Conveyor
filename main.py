"""
Chương trình chính - Điểm khởi đầu của hệ thống
"""

import sys
import os
import logging
import argparse
from datetime import datetime

# Thêm đường dẫn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ai_processor import AIProcessor
from web_interface.app import app as web_app
import threading

def setup_logging():
    """Thiết lập hệ thống logging"""
    # Tạo thư mục logs nếu chưa tồn tại
    os.makedirs("data/logs", exist_ok=True)
    
    # Định dạng thời gian
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Cấu hình logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(f"data/logs/system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_arguments():
    """Phân tích tham số dòng lệnh"""
    parser = argparse.ArgumentParser(description='Hệ thống Phân loại Trái cây AI')
    
    parser.add_argument('--mode', type=str, default='web',
                       choices=['web', 'cli', 'test'],
                       help='Chế độ chạy: web (giao diện), cli (dòng lệnh), test (kiểm thử)')
    
    parser.add_argument('--camera', type=str, default='pi',
                       choices=['pi', 'usb'],
                       help='Loại camera: pi (Raspberry Pi), usb (USB webcam)')
    
    parser.add_argument('--no-motor', action='store_true',
                       help='Không khởi tạo motor controller (cho test)')
    
    parser.add_argument('--speed', type=int, default=70,
                       help='Tốc độ băng chuyền ban đầu (0-100)')
    
    parser.add_argument('--port', type=int, default=5000,
                       help='Port cho web server')
    
    return parser.parse_args()

def run_cli_mode(processor):
    """Chạy chế độ dòng lệnh"""
    print("\n" + "="*60)
    print("Development of a Conveyor System for Fruit Quality Classification Using AI Camera - CHẾ ĐỘ DÒNG LỆNH")
    print("="*60)
    
    print("\nCác lệnh có sẵn:")
    print("  start    - Bắt đầu xử lý")
    print("  stop     - Dừng xử lý")
    print("  status   - Xem trạng thái")
    print("  speed N  - Đặt tốc độ động cơ (0-100)")
    print("  exit     - Thoát chương trình")
    
    while True:
        try:
            command = input("\n>>> ").strip().lower()
            
            if command == 'start':
                processor.start_processing()
                print("Đã bắt đầu xử lý")
                
            elif command == 'stop':
                processor.stop()
                print("Đã dừng xử lý")
                
            elif command == 'status':
                status = processor.get_status()
                print("\n--- TRẠNG THÁI HỆ THỐNG ---")
                print(f"Đang chạy: {'Có' if status.get('is_running') else 'Không'}")
                print(f"FPS Camera: {status.get('camera_fps', 0):.1f}")
                print(f"Tổng frame: {status.get('total_frames_processed', 0)}")
                print(f"Tổng trái cây: {status.get('total_fruits_detected', 0)}")
                print(f"Tốc độ động cơ: {status.get('motor_speed', 0)}%")
                print(f"Thời gian inference: {status.get('avg_inference_time_ms', 0):.1f}ms")
                
            elif command.startswith('speed '):
                try:
                    speed = int(command.split()[1])
                    processor.set_motor_speed(speed)
                    print(f"Đã đặt tốc độ: {speed}%")
                except:
                    print("Lỗi: Vui lòng nhập số từ 0-100")
                    
            elif command == 'exit':
                print("Đang dừng hệ thống...")
                processor.stop()
                break
                
            else:
                print("Lệnh không hợp lệ. Vui lòng thử lại.")
                
        except KeyboardInterrupt:
            print("\n\nĐang dừng hệ thống...")
            processor.stop()
            break
        except Exception as e:
            print(f"Lỗi: {e}")

def run_test_mode():
    """Chạy chế độ kiểm thử"""
    print("\n" + "="*60)
    print("CHẾ ĐỘ KIỂM THỬ PHẦN CỨNG")
    print("="*60)
    
    # TODO: Thêm các test phần cứng
    print("Chức năng kiểm thử đang được phát triển...")

def main():
    """Hàm chính"""
    # Thiết lập logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Phân tích tham số
    args = parse_arguments()
    
    logger.info(f"Bắt đầu hệ thống với chế độ: {args.mode}")
    logger.info(f"Sử dụng camera: {args.camera}")
    
    try:
        # Khởi tạo AI Processor
        processor = AIProcessor()
        
        # Khởi tạo hệ thống
        if processor.initialize():
            logger.info("Khởi tạo hệ thống thành công")
            
            # Đặt tốc độ ban đầu
            processor.set_motor_speed(args.speed)
            
            # Chạy theo chế độ đã chọn
            if args.mode == 'web':
                # Chạy web server trong thread riêng
                web_thread = threading.Thread(
                    target=lambda: web_app.run(
                        host='0.0.0.0',
                        port=args.port,
                        debug=False,
                        threaded=True
                    ),
                    daemon=True
                )
                web_thread.start()
                
                print(f"\n{'='*60}")
                print("Development of a Conveyor System for Fruit Quality Classification Using AI Camera")
                print("="*60)
                print(f"\nGiao diện web: http://localhost:{args.port}")
                print(f"Hoặc: http://<địa-chỉ-ip-raspberry>:{args.port}")
                print("\nNhấn Ctrl+C để dừng chương trình\n")
                
                # Giữ chương trình chạy
                try:
                    while True:
                        # Có thể thêm các xử lý khác ở đây
                        import time
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nĐang dừng hệ thống...")
                    
            elif args.mode == 'cli':
                run_cli_mode(processor)
                
            elif args.mode == 'test':
                run_test_mode()
            
            # Dọn dẹp
            processor.stop()
            logger.info("Hệ thống đã dừng")
            
        else:
            logger.error("Khởi tạo hệ thống thất bại")
            return 1
            
    except Exception as e:
        logger.error(f"Lỗi không mong muốn: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())