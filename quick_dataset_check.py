#!/usr/bin/env python3
"""
Quick Dataset Check for Fruit Sorting System
"""

from pathlib import Path

def quick_check():
    """Quick dataset statistics"""
    raw_images = Path("raw_images")

    if not raw_images.exists():
        print("❌ raw_images directory not found")
        return

    fresh_dir = raw_images / "fresh"
    spoiled_dir = raw_images / "spoiled"

    fresh_count = len(list(fresh_dir.glob("*.jpg"))) if fresh_dir.exists() else 0
    spoiled_count = len(list(spoiled_dir.glob("*.jpg"))) if spoiled_dir.exists() else 0

    total = fresh_count + spoiled_count
    balance_ratio = min(fresh_count, spoiled_count) / max(fresh_count, spoiled_count) if max(fresh_count, spoiled_count) > 0 else 0

    print("🔍 Dataset Quick Check")
    print("=" * 30)
    print(f"Fresh images: {fresh_count}")
    print(f"Spoiled images: {spoiled_count}")
    print(f"Total images: {total}")
    print(".2f")
    print()

    # Recommendations
    recommendations = []

    if total < 400:  # 200 per class
        recommendations.append(f"❌ Need more data: {400 - total} more images total")

    if balance_ratio < 0.8:
        minority = "spoiled" if fresh_count > spoiled_count else "fresh"
        majority_count = max(fresh_count, spoiled_count)
        minority_count = min(fresh_count, spoiled_count)
        needed = majority_count - minority_count
        recommendations.append(f"⚠️ Class imbalance: Collect {needed} more {minority} images")

    if not recommendations:
        print("✅ Dataset meets minimum requirements")
    else:
        print("📋 Recommendations:")
        for rec in recommendations:
            print(f"   {rec}")

if __name__ == "__main__":
    quick_check()
