"""
Improved MobileNet Training Script for 90%+ Accuracy
Enhanced training with fine-tuning and better augmentation
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from datetime import datetime
import json

def create_enhanced_augmentation():
    """Enhanced augmentation pipeline for better generalization"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.3),
        tf.keras.layers.RandomTranslation(0.3, 0.3, fill_mode='reflect'),
        tf.keras.layers.RandomBrightness((-0.5, 0.5)),
        tf.keras.layers.RandomContrast(0.4),
        tf.keras.layers.GaussianNoise(0.01),
        tf.keras.layers.Rescaling(1./127.5, offset=-1)
    ], name="enhanced_augmentation")

def create_data_generator(directory, image_size=(224, 224), batch_size=16, shuffle=True, augment=False):
    """Create optimized data generator"""
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',
        class_names=['fresh', 'spoiled'],
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=42
    )

    AUTOTUNE = tf.data.AUTOTUNE
    dataset = dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    if augment:
        augmentation = create_enhanced_augmentation()
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )
    else:
        normalization = tf.keras.layers.Rescaling(1./127.5, offset=-1)
        dataset = dataset.map(
            lambda x, y: (normalization(x), y),
            num_parallel_calls=AUTOTUNE
        )

    return dataset

def create_improved_mobilenet_model(input_shape=(224, 224, 3), num_classes=2):
    """Create improved MobileNetV2 model with fine-tuning"""

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Fine-tune the last 30 layers
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.6)(x)

    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

def train_improved_model(train_dir, val_dir, output_dir="improved_training"):
    """Train improved model for 90%+ accuracy"""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("🚀 Training Improved MobileNetV2 Model")
    print("=" * 50)

    BATCH_SIZE = 16
    IMAGE_SIZE = 224

    train_dataset = create_data_generator(
        train_dir,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=True
    )

    val_dataset = create_data_generator(
        val_dir,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False
    )

    train_fresh = len(list(Path(train_dir, 'fresh').glob('*.[jp][pn][g]*')))
    train_spoiled = len(list(Path(train_dir, 'spoiled').glob('*.[jp][pn][g]*')))

    total_count = train_fresh + train_spoiled
    weight_fresh = total_count / (2 * train_fresh) if train_fresh > 0 else 1.0
    weight_spoiled = total_count / (2 * train_spoiled) if train_spoiled > 0 else 1.0
    class_weights = {0: weight_fresh, 1: weight_spoiled}

    print(f"📊 Dataset: {train_fresh} fresh, {train_spoiled} spoiled")
    print(f"⚖️ Class weights: Fresh={weight_fresh:.3f}, Spoiled={weight_spoiled:.3f}")

    model = create_improved_mobilenet_model()

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(output_path / 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
    ]

    EPOCHS = 100

    print("\n🚀 Starting training...")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Fine-tuning last 30 layers")
    print("-" * 60)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    model.save(str(output_path / 'final_model.keras'))

    results = model.evaluate(val_dataset, verbose=0)

    print("\n📊 Final Validation Results:")
    print(f"   Loss: {results[0]:.4f}")
    print(f"   Accuracy: {results[1]:.2%}")
    print(f"   Precision: {results[2]:.2%}")
    print(f"   Recall: {results[3]:.2%}")

    precision = results[2]
    recall = results[3]
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"   F1 Score: {f1:.2%}")

    print("\n✅ Training complete!")
    print(f"   Best model saved: {output_path / 'best_model.keras'}")
    print(f"   Final model saved: {output_path / 'final_model.keras'}")
    print(f"   Achieved F1: {f1:.2%}")

    return f1 >= 0.90

if __name__ == "__main__":
    success = train_improved_model(
        train_dir="dataset/train",
        val_dir="dataset/val",
        output_dir="improved_training"
    )

    if success:
        print("\n🎉 SUCCESS: Achieved 90%+ F1 score!")
    else:
        print("\n⚠️  Target not reached. Consider collecting more data or adjusting hyperparameters.")

    print("\nNext steps:")
    print("1. Convert best model to TFLite: python3 convert_to_tflite.py improved_training/best_model.keras")
    print("2. Copy to models directory: cp improved_training/mobilenet_classifier.tflite models/")
    print("3. Test with evaluation: python3 evaluate_system.py --test_dir dataset/test")
