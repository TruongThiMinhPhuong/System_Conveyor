"""
Test servo control without AI models
Allows manual testing of servo mechanism independently

This script tests the conveyor and servo hardware without requiring
any AI models to be loaded. Useful for:
- Hardware debugging
- Servo calibration
- Testing sorting mechanism
"""

import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from hardware import ConveyorSystem


def test_servo_only():
    """Test servo control independently of AI models"""
    print("=" * 60)
    print("Servo-Only Test (No AI Models Required)")
    print("=" * 60)
    print()
    
    try:
        with ConveyorSystem() as conveyor:
            print("✅ Hardware initialized successfully!\n")
            
            # Start conveyor at medium speed
            print("🔄 Starting conveyor at 50% speed...")
            conveyor.start_conveyor(50)
            time.sleep(2)
            
            # Test fresh sorting (straight through)
            print("\n" + "=" * 40)
            print("Test 1: Sorting FRESH fruit (0° - straight)")
            print("=" * 40)
            conveyor.sort_fruit(is_fresh=True, pause_conveyor=True)
            time.sleep(3)
            
            # Test spoiled sorting (push right)
            print("\n" + "=" * 40)
            print("Test 2: Sorting SPOILED fruit (180° - right)")
            print("=" * 40)
            conveyor.sort_fruit(is_fresh=False, pause_conveyor=True)
            time.sleep(3)
            
            # Test alternating sorts
            print("\n" + "=" * 40)
            print("Test 3: Alternating sorts (5 cycles)")
            print("=" * 40)
            for i in range(5):
                is_fresh = (i % 2 == 0)
                sort_type = "FRESH" if is_fresh else "SPOILED"
                print(f"\nCycle {i+1}/5: Sorting {sort_type}...")
                conveyor.sort_fruit(is_fresh=is_fresh, pause_conveyor=True)
                time.sleep(2)
            
            # Stop conveyor
            print("\n🛑 Stopping conveyor...")
            conveyor.stop_conveyor()
            
            print("\n" + "=" * 60)
            print("✅ All servo tests completed successfully!")
            print("=" * 60)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_continuous_sorting():
    """Test continuous sorting with timed intervals"""
    print("=" * 60)
    print("Continuous Sorting Test (Manual Mode Simulation)")
    print("=" * 60)
    print()
    
    try:
        with ConveyorSystem() as conveyor:
            print("✅ Hardware initialized successfully!\n")
            
            # Start conveyor
            conveyor.start_conveyor(70)
            
            print("🔄 Starting continuous sorting...")
            print("   Sorting every 5 seconds (alternating fresh/spoiled)")
            print("   Press Ctrl+C to stop\n")
            
            count = 0
            while True:
                is_fresh = (count % 2 == 0)
                sort_type = "FRESH" if is_fresh else "SPOILED"
                print(f"[{count+1}] Sorting {sort_type}...")
                
                conveyor.sort_fruit(is_fresh=is_fresh, pause_conveyor=True)
                
                count += 1
                time.sleep(5)  # Wait 5 seconds between sorts
                
    except KeyboardInterrupt:
        print("\n\n⚠️ Continuous test stopped by user")
        print(f"✅ Completed {count} sorting cycles")
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("\n🔧 Servo Control Test\n")
    print("Select test mode:")
    print("1. Basic servo test (3 tests)")
    print("2. Continuous sorting (press Ctrl+C to stop)")
    print()
    
    choice = input("Enter choice (1 or 2, default=1): ").strip()
    
    if choice == '2':
        test_continuous_sorting()
    else:
        test_servo_only()
    
    print("\n✅ Test complete!\n")
