"""
MobileNetV2 Training Script for Fresh/Spoiled Classification
Train on PC/Laptop with GPU, export to TensorFlow Lite for Raspberry Pi
"""

import argparse
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from augmentation import create_data_generator


def create_mobilenet_model(
    input_shape: tuple = (224, 224, 3),
    num_classes: int = 2,
    freeze_base: bool = True,
    dropout_rate: float = 0.5
):
    """
    Create MobileNetV2 transfer learning model with improved regularization
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        freeze_base: Whether to freeze base model weights
        dropout_rate: Dropout rate (increased for better generalization)
        
    Returns:
        Keras model
    """
    # Load MobileNetV2 base model (pretrained on ImageNet)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model if requested
    base_model.trainable = not freeze_base
    
    # Create model
    inputs = keras.Input(shape=input_shape)
    
    # Preprocessing is done in data generator
    x = base_model(inputs, training=False)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with L2 regularization
    x = layers.Dense(
        256, 
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(
        128, 
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model


def train_mobilenet(
    dataset_dir: str = './datasets/fruit_classification',
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    image_size: int = 224,
    output_dir: str = './mobilenet_training',
    model_name: str = 'mobilenet_fruit_classifier'
):
    """
    Train MobileNetV2 classifier
    
    Args:
        dataset_dir: Path to dataset directory
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        image_size: Input image size
        output_dir: Output directory for models and logs
        model_name: Model name
    """
    print("=" * 60)
    print("MobileNetV2 Fresh/Spoiled Fruit Classifier Training")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data generators
    print(f"\n📊 Loading dataset from {dataset_dir}...")
    
    train_dataset= create_data_generator(
        directory=f"{dataset_dir}/train",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        augment=True  # Apply augmentation to training data
    )
    
    val_dataset = create_data_generator(
        directory=f"{dataset_dir}/val",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
        augment=False  # No augmentation for validation
    )
    
    print("✅ Datasets loaded")
    
    # Calculate class weights to handle potential class imbalance
    print(f"\n⚖️ Calculating class weights...")
    
    # Count samples per class
    fresh_count = len(list(Path(f"{dataset_dir}/train/fresh").glob("*.[jp][pn][g]*")))
    spoiled_count = len(list(Path(f"{dataset_dir}/train/spoiled").glob("*.[jp][pn][g]*")))
    total_count = fresh_count + spoiled_count
    
    print(f"   Fresh: {fresh_count} samples")
    print(f"   Spoiled: {spoiled_count} samples")
    
    # Calculate weights (inverse frequency)
    weight_fresh = total_count / (2 * fresh_count) if fresh_count > 0 else 1.0
    weight_spoiled = total_count / (2 * spoiled_count) if spoiled_count > 0 else 1.0
    
    class_weights = {0: weight_fresh, 1: weight_spoiled}
    
    print(f"   Class weights: Fresh={weight_fresh:.3f}, Spoiled={weight_spoiled:.3f}")
    
    # Create model
    print(f"\n🏗️ Building MobileNetV2 model...")
    model = create_mobilenet_model(
        input_shape=(image_size, image_size, 3),
        num_classes=2,
        freeze_base=True,
        dropout_rate=0.5  # Increased for better generalization
    )
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    print("✅ Model built")
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=str(output_path / f'{model_name}_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(output_path / 'logs'),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    # Train model
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Using class weights: {class_weights}")
    print("-" * 60)
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,  # Handle class imbalance
        verbose=1
    )
    
    print("-" * 60)
    print("✅ Training complete!")
    
    # Save final model
    model.save(output_path / f'{model_name}_final.keras')
    print(f"💾 Model saved to {output_path}")
    
    # Plot training history
    plot_training_history(history, output_path)
    
    # Evaluate on validation set
    print(f"\n📊 Evaluating on validation set...")
    results = model.evaluate(val_dataset, verbose=0)
    
    print(f"\n📈 Validation Metrics:")
    print(f"   Loss: {results[0]:.4f}")
    print(f"   Accuracy: {results[1]:.2%}")
    print(f"   Precision: {results[2]:.2%}")
    print(f"   Recall: {results[3]:.2%}")
    
    # Calculate F1 score
    precision = results[2]
    recall = results[3]
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"   F1 Score: {f1_score:.2%}")
    
    print(f"\n🎉 Training complete!")
    print(f"📦 Next step: Convert to TensorFlow Lite")
    print(f"   Run: python export_tflite.py --model {output_path / f'{model_name}_best.keras'}")
    
    return model, history


def plot_training_history(history, output_dir):
    """
    Plot and save training history
    
    Args:
        history: Training history object
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Val')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train')
    axes[1, 0].plot(history.history['val_precision'], label='Val')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train')
    axes[1, 1].plot(history.history['val_recall'], label='Val')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150)
    print(f"💾 Training history saved to {output_dir / 'training_history.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Train MobileNetV2 for fresh/spoiled classification'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='./datasets/fruit_classification',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Image size'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./mobilenet_training',
        help='Output directory'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='mobilenet_fruit_classifier',
        help='Model name'
    )
    
    args = parser.parse_args()
    
    train_mobilenet(
        dataset_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        image_size=args.image_size,
        output_dir=args.output,
        model_name=args.name
    )


if __name__ == '__main__':
    main()
