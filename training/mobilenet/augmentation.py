"""
Data Augmentation for MobileNetV2 Training
TensorFlow/Keras augmentation pipeline
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_augmentation_pipeline(
    rotation_range: float = 25.0,
    width_shift_range: float = 0.25,
    height_shift_range: float = 0.25,
    zoom_range: float = 0.25,
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
    brightness_range: tuple = (0.6, 1.4),
    include_preprocessing: bool = True
):
    """
    Create data augmentation pipeline optimized for conveyor belt conditions
    
    Args:
        rotation_range: Rotation angle range (degrees)
        width_shift_range: Horizontal shift fraction
        height_shift_range: Vertical shift fraction
        zoom_range: Zoom fraction
        horizontal_flip: Whether to apply horizontal flip
        vertical_flip: Whether to apply vertical flip
        brightness_range: Brightness adjustment range (wider for varying lighting)
        include_preprocessing: Include MobileNetV2 preprocessing
        
    Returns:
        Keras Sequential model for augmentation
    """
    augmentation_layers = []
    
    # Random horizontal flip (fruits can appear from either side)
    if horizontal_flip:
        augmentation_layers.append(
            layers.RandomFlip("horizontal")
        )
    
    # Random vertical flip (fruits can be oriented differently)
    if vertical_flip:
        augmentation_layers.append(
            layers.RandomFlip("vertical")
        )
    
    # Random rotation (fruits roll on conveyor)
    if rotation_range > 0:
        augmentation_layers.append(
            layers.RandomRotation(rotation_range / 360.0)
        )
    
    # Random zoom (different fruit sizes, varying distances from camera)
    if zoom_range > 0:
        augmentation_layers.append(
            layers.RandomZoom(
                height_factor=(-zoom_range, zoom_range),
                width_factor=(-zoom_range, zoom_range)
            )
        )
    
    # Random translation (fruits move across frame)
    if width_shift_range > 0 or height_shift_range > 0:
        augmentation_layers.append(
            layers.RandomTranslation(
                height_factor=height_shift_range,
                width_factor=width_shift_range,
                fill_mode='reflect'
            )
        )
    
    # Random brightness (varying lighting conditions on conveyor)
    if brightness_range:
        augmentation_layers.append(
            layers.RandomBrightness(
                factor=brightness_range
            )
        )
    
    # Random contrast (different lighting conditions)
    augmentation_layers.append(
        layers.RandomContrast(factor=0.3)
    )
    
    # MobileNetV2 preprocessing - CRITICAL: Must match inference preprocessing
    if include_preprocessing:
        # Normalize to [-1, 1] range as expected by MobileNetV2
        augmentation_layers.append(
            layers.Rescaling(1./127.5, offset=-1)
        )
    
    return keras.Sequential(augmentation_layers, name="augmentation")


def create_data_generator(
    directory: str,
    image_size: tuple = (224, 224),
    batch_size: int = 32,
    class_names: list = ['fresh', 'spoiled'],
    shuffle: bool = True,
    augment: bool = False
):
    """
    Create data generator from directory
    
    Args:
        directory: Path to data directory
        image_size: Target image size (height, width)
        batch_size: Batch size
        class_names: List of class names
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentation
        
    Returns:
        tf.data.Dataset
    """
    # Load dataset
    dataset = keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',  # One-hot encoding
        class_names=class_names,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=42
    )
    
    # Apply augmentation if requested
    if augment:
        augmentation = create_augmentation_pipeline()
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        # Just normalize
        normalization = layers.Rescaling(1./127.5, offset=-1)
        dataset = dataset.map(
            lambda x, y: (normalization(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Optimize performance
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset


def visualize_augmentation(dataset, augmentation_pipeline, num_images: int = 9):
    """
    Visualize augmentation effects
    
    Args:
        dataset: TensorFlow dataset
        augmentation_pipeline: Augmentation Sequential model
        num_images: Number of images to show
    """
    import matplotlib.pyplot as plt
    
    # Get a batch of images
    for images, labels in dataset.take(1):
        # Take first image
        image = images[0:1]
        
        # Create figure
        plt.figure(figsize=(12, 12))
        
        for i in range(num_images):
            # Apply augmentation
            augmented = augmentation_pipeline(image, training=True)
            
            # Plot
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow((augmented[0].numpy() + 1) / 2)  # Denormalize
            plt.axis('off')
        
        plt.suptitle('Data Augmentation Examples')
        plt.tight_layout()
        plt.savefig('augmentation_examples.png')
        print("💾 Saved augmentation examples to 'augmentation_examples.png'")
        plt.close()


if __name__ == '__main__':
    print("=== Data Augmentation Test ===\n")
    
    # Create augmentation pipeline
    augmentation = create_augmentation_pipeline()
    
    print("📊 Augmentation Pipeline:")
    augmentation.summary()
    
    # Test with random image
    test_image = tf.random.uniform((1, 224, 224, 3), minval=0, maxval=255)
    
    print(f"\n🧪 Testing augmentation...")
    print(f"   Input shape: {test_image.shape}")
    
    augmented = augmentation(test_image, training=True)
    
    print(f"   Output shape: {augmented.shape}")
    print(f"   Value range: [{tf.reduce_min(augmented).numpy():.3f}, "
          f"{tf.reduce_max(augmented).numpy():.3f}]")
    
    print("\n✅ Augmentation test complete!")
