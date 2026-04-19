# -*- coding: utf-8 -*-
"""
EXPERIMENT 2: AUTOENCODER & TRANSFER LEARNING
Chest X-Ray (Pneumonia) Classification
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Prevent GPU OOM crashes
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

def _resolve_dataset_root():
    """Find chest_xray dataset root from common project locations."""
    candidates = [
        os.path.join(BASE_PATH, 'archive', 'chest_xray'),
        os.path.join(BASE_PATH, '..', 'dataset', 'Chest_X-Ray_Images_archive', 'chest_xray', 'chest_xray'),
        os.path.join(BASE_PATH, '..', 'dataset', 'Chest_X-Ray_Images_archive', 'chest_xray'),
    ]
    checked = []
    for path in candidates:
        normalized = os.path.abspath(path)
        checked.append(normalized)
        if os.path.isdir(os.path.join(normalized, 'train')) and os.path.isdir(os.path.join(normalized, 'test')):
            return normalized
    raise FileNotFoundError(
        "Could not find chest_xray dataset. Expected folders 'train' and 'test' under one of:\n"
        + "\n".join(checked)
    )

# Using same Chest X-Ray dataset as Exp1 and Exp3
DATASET_ROOT = _resolve_dataset_root()
FULL_TRAIN_DIR = os.path.join(DATASET_ROOT, 'train')
TEST_DIR = os.path.join(DATASET_ROOT, 'test')

RESULTS_DIR = os.path.join(BASE_PATH, 'results_xray', 'exp2_autoencoder')
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)  # Consistency with Exp1 and Exp3
BATCH_SIZE_AUTOENCODER = 16  # Smaller batch for autoencoder (memory intensive)
BATCH_SIZE = 32  # Normal batch size for transfer learning
EPOCHS = 15
NUM_CLASSES = 2  # Normal vs. Pneumonia
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

print("=" * 80)
print("EXPERIMENT 2: AUTOENCODER & TRANSFER LEARNING (CHEST X-RAY)")
print("=" * 80)

# 1. Data Pipeline with 80/20 train/val split
preprocess_input = tf.keras.applications.resnet50.preprocess_input
AUTOTUNE = tf.data.AUTOTUNE

def create_autoencoder_datasets():
    """Create fresh datasets for autoencoder training (smaller batch size)."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        FULL_TRAIN_DIR, seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE_AUTOENCODER, label_mode='categorical',
        color_mode='rgb', validation_split=0.2, subset='training'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        FULL_TRAIN_DIR, seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE_AUTOENCODER, label_mode='categorical',
        color_mode='rgb', validation_split=0.2, subset='validation', shuffle=False
    )
    # Unsupervised: input = target
    train_unsup = train_ds.map(lambda x, y: (preprocess_input(x), preprocess_input(x)), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_unsup = val_ds.map(lambda x, y: (preprocess_input(x), preprocess_input(x)), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return train_unsup, val_unsup

def create_supervised_datasets():
    """Create fresh supervised datasets for transfer learning."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        FULL_TRAIN_DIR, seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical',
        color_mode='rgb', validation_split=0.2, subset='training'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        FULL_TRAIN_DIR, seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical',
        color_mode='rgb', validation_split=0.2, subset='validation', shuffle=False
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical',
        color_mode='rgb', shuffle=False
    )
    train_supervised = train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_supervised = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_supervised = test_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return train_supervised, val_supervised, test_supervised

# 2. Phase 1: Train Autoencoder
autoencoder_path = os.path.join(RESULTS_DIR, 'exp2_xray_autoencoder.keras')

# Encoder Definition - VGG-style blocks with BatchNorm
encoder_input = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))  # 3 Channels (RGB X-Ray)

x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_input)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Dropout(0.2)(x)

# Bottleneck
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
encoded_features = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
encoded_features = layers.BatchNormalization()(encoded_features)

if os.path.exists(autoencoder_path):
    print("🚀 Loaded pre-trained Autoencoder from disk.")
    autoencoder = tf.keras.models.load_model(autoencoder_path)
else:
    print("Training Unsupervised Autoencoder on X-Rays...")
    # Create fresh datasets for autoencoder
    train_unsup, val_unsup = create_autoencoder_datasets()
    
    # Decoder Definition 
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(encoded_features)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Output layer (3 Channels to reconstruct the RGB X-Ray)
    # Use linear to match ResNet50 preprocessing range of unbounded positive/negative values
    decoded_output = layers.Conv2D(3, (3, 3), activation='linear', padding='same')(x)
    
    autoencoder = models.Model(encoder_input, decoded_output)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    
    autoencoder.fit(train_unsup, validation_data=val_unsup, epochs=20, 
                    callbacks=[callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
    autoencoder.save(autoencoder_path)
    print(f"✓ Autoencoder saved to {autoencoder_path}")

# 3. Phase 2: Transfer Learning
transfer_path = os.path.join(RESULTS_DIR, 'exp2_xray_transfer.keras')
history_path = os.path.join(RESULTS_DIR, 'exp2_xray_history.json')

# Create fresh datasets for supervised phase
train_supervised, val_supervised, test_supervised = create_supervised_datasets()

if os.path.exists(transfer_path):
    print("🚀 Loaded Transfer Learning Classifier from disk.")
    transfer_model = tf.keras.models.load_model(transfer_path)
    
    # Recreate test dataset for fresh evaluation
    _, _, test_supervised = create_supervised_datasets()
    test_loss, test_acc = transfer_model.evaluate(test_supervised, verbose=0)
    
    # Get predictions for detailed metrics
    test_predictions = transfer_model.predict(test_supervised, verbose=0)
    test_pred_classes = np.argmax(test_predictions, axis=1)
    
    # Collect true labels
    test_true_labels = []
    for _, y in test_supervised:
        test_true_labels.extend(np.argmax(y.numpy(), axis=1))
    test_true_labels = np.array(test_true_labels)
    
    # Calculate detailed metrics
    test_precision = precision_score(test_true_labels, test_pred_classes, zero_division=0)
    test_recall = recall_score(test_true_labels, test_pred_classes, zero_division=0)
    test_f1 = f1_score(test_true_labels, test_pred_classes, zero_division=0)
    test_cm = confusion_matrix(test_true_labels, test_pred_classes, labels=[0, 1])
    
    print(f"\n✓ Phase 2 (Loaded) - Test Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    print("\nConfusion Matrix (Test Set):")
    print(f"{test_cm.tolist()}")
else:
    print("Training Transfer Learning Classifier on X-Rays...")
    # Extract Encoder and UNFREEZE it so it can learn pneumonia-specific patterns
    encoder_model = models.Model(encoder_input, encoded_features)
    encoder_model.trainable = True  

    transfer_model = models.Sequential([
        encoder_model,
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.6),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')
    ])
    
    transfer_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy'] 
    )
    
    hist = transfer_model.fit(
        train_supervised, validation_data=val_supervised, epochs=EPOCHS, 
        callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )
    
    transfer_model.save(transfer_path)
    with open(history_path, 'w') as f: 
        json.dump({k: [float(v) for v in vals] for k, vals in hist.history.items()}, f)
    print(f"✓ Transfer model saved to {transfer_path}")
    
    # Recreate datasets for fresh iterators
    train_supervised, val_supervised, test_supervised = create_supervised_datasets()
    
    # Evaluate on test set
    test_loss, test_acc = transfer_model.evaluate(test_supervised, verbose=0)
    
    # Recreate test dataset (fresh iterator for predictions)
    _, _, test_supervised = create_supervised_datasets()
    
    # Get predictions for detailed metrics
    test_predictions = transfer_model.predict(test_supervised, verbose=0)
    test_pred_classes = np.argmax(test_predictions, axis=1)
    
    # Recreate test dataset again for label collection (iterator was consumed)
    _, _, test_supervised = create_supervised_datasets()
    
    # Collect true labels
    test_true_labels = []
    for _, y in test_supervised:
        test_true_labels.extend(np.argmax(y.numpy(), axis=1))
    test_true_labels = np.array(test_true_labels)
    
    # Calculate detailed metrics
    test_precision = precision_score(test_true_labels, test_pred_classes, zero_division=0)
    test_recall = recall_score(test_true_labels, test_pred_classes, zero_division=0)
    test_f1 = f1_score(test_true_labels, test_pred_classes, zero_division=0)
    test_cm = confusion_matrix(test_true_labels, test_pred_classes, labels=[0, 1])
    
    print(f"\n✓ Phase 2 - Test Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    print("\nConfusion Matrix (Test Set) - [True Neg, False Pos; False Neg, True Pos]:")
    print(f"{test_cm.tolist()}")
    
    # Save results
    results = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'test_confusion_matrix': test_cm.tolist()
    }
    results_path = os.path.join(RESULTS_DIR, 'exp2_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

print("\n" + "="*80)
print("EXPERIMENT 2 SUMMARY - AUTOENCODER + TRANSFER LEARNING")
print("="*80)
print(f"Test Accuracy: {test_acc:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")
print("="*80)
print("\n✓ Experiment 2 Complete!")