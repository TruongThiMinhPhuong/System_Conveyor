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
        tf.keras.layers.RandomRotation(0.15),  # Increased rotation
        tf.keras.layers.RandomZoom(0.3),  # Increased zoom
        tf.keras.layers.RandomTranslation(0.3, 0.3, fill_mode='reflect'),
        tf.keras.layers.RandomBrightness((-0.5, 0.5)),  # Increased brightness range
        tf.keras.layers.RandomContrast(0.4),  # Increased contrast
        tf.keras.layers.GaussianNoise(0.01),  # Add noise for robustness
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

    # Optimize dataset performance
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

    # Load pretrained base
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Fine-tune the last 30 layers instead of freezing all
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Build improved model
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)

    # Enhanced feature extraction
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Improved dense layers with better regularization
    x = tf.keras.layers.Dense(
        512,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.6)(x)  # Increased dropout

    x = tf.keras.layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

def train_improved_model(train_dir, val_dir, output_dir="improved_training"):
    """Train improved model for 90%+ accuracy"""

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("🚀 Training Improved MobileNetV2 Model")
    print("=" * 50)

    # Data generators
    BATCH_SIZE = 16  # Smaller batch size for better generalization
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

    # Calculate class weights
    train_fresh = len(list(Path(train_dir, 'fresh').glob('*.[jp][pn][g]*')))
    train_spoiled = len(list(Path(train_dir, 'spoiled').glob('*.[jp][pn][g]*')))

    total_count = train_fresh + train_spoiled
    weight_fresh = total_count / (2 * train_fresh) if train_fresh > 0 else 1.0
    weight_spoiled = total_count / (2 * train_spoiled) if train_spoiled > 0 else 1.0
    class_weights = {0: weight_fresh, 1: weight_spoiled}

    print(f"📊 Dataset: {train_fresh} fresh, {train_spoiled} spoiled")
    print(".3f")

    # Create model
    model = create_improved_mobilenet_model()
    model.summary()

    # Compile with improved optimizer
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=1e-4,  # Lower learning rate for fine-tuning
            weight_decay=1e-4
        ),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.F1Score()
        ]
    )

    # Enhanced callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(output_path / 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(output_path / 'best_f1_model.keras'),
            monitor='val_f1_score',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,  # More patience
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(output_path / 'training.log'))
    ]

    # Train with more epochs
    EPOCHS = 100

    print("
🚀 Starting training..."    print(f"   Epochs: {EPOCHS}")
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

    # Save final model
    model.save(str(output_path / 'final_model.keras'))

    # Evaluate
    results = model.evaluate(val_dataset, verbose=0)

    print("
📊 Final Validation Results:"    print(".4f"    print(".2%"    print(".2%"    print(".2%")

    # Calculate F1
    precision = results[1]
    recall = results[2]
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(".2%")

    # Save training history
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history.history, f, indent=2)

    # Plot training curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Val')
    axes[0, 1].set_title('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Precision
    axes[0, 2].plot(history.history['precision'], label='Train')
    axes[0, 2].plot(history.history['val_precision'], label='Val')
    axes[0, 2].set_title('Precision')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Recall
    axes[1, 0].plot(history.history['recall'], label='Train')
    axes[1, 0].plot(history.history['val_recall'], label='Val')
    axes[1, 0].set_title('Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # F1 Score
    if 'f1_score' in history.history:
        axes[1, 1].plot(history.history['f1_score'], label='Train')
        axes[1, 1].plot(history.history['val_f1_score'], label='Val')
        axes[1, 1].set_title('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    # Learning Rate
    if 'lr' in history.history:
        axes[1, 2].plot(history.history['lr'])
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(str(output_path / 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("
✅ Training complete!"    print(f"   Best model saved: {output_path / 'best_model.keras'}")
    print(f"   Final model saved: {output_path / 'final_model.keras'}")
    print(".2%")

    return f1 >= 0.90  # Return True if achieved 90% F1

if __name__ == "__main__":
    # Train improved model
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
