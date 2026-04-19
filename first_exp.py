# -*- coding: utf-8 -*-
"""
EXPERIMENT 1: CUSTOM CNN ARCHITECTURES
Chest X-Ray (Pneumonia) Classification
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import numpy as np
import os
import json
import time
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

# Using Chest X-Ray dataset (same as Exp3)
# Note: Pre-made val folder has only 16 images, so we'll create proper split from training data
DATASET_ROOT = _resolve_dataset_root()
FULL_TRAIN_DIR = os.path.join(DATASET_ROOT, 'train')
TEST_DIR = os.path.join(DATASET_ROOT, 'test')

RESULTS_DIR = os.path.join(BASE_PATH, 'results_xray', 'exp1_custom_cnn')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Use 224x224 for consistency with Exp3 (ResNet50 standard)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 2  # Normal vs. Pneumonia
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

print("=" * 80)
print("EXPERIMENT 1: CUSTOM CNN ARCHITECTURES (CHEST X-RAY PNEUMONIA)")
print("=" * 80)

# 1. Data Pipeline - datasets created dynamically per model (avoid iterator exhaustion)

preprocess_input = tf.keras.applications.resnet50.preprocess_input
AUTOTUNE = tf.data.AUTOTUNE

def create_datasets():
    """Create fresh dataset objects with proper train/val split.
    Uses 80% training, 20% validation from the full training data.
    Avoids iterator exhaustion issues."""
    
    # Create training dataset with 80/20 split
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
    
    train_ds_norm = train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds_norm = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds_norm = test_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    
    return train_ds_norm, val_ds_norm, test_ds_norm

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# 2. Configurations
configs = [
    {'name': 'Baseline', 'filters': [32, 64, 128],      'dense': 256, 'dropout': 0.5},
    {'name': 'Deep',     'filters': [32, 64, 128, 256], 'dense': 256, 'dropout': 0.5},
    {'name': 'Wide',     'filters': [64, 128, 256],     'dense': 512, 'dropout': 0.5},
]

experiment1_results = {}

# 3. Training Loop
for config in configs:
    print(f"\nProcessing: {config['name']} CNN...")
    model_path = os.path.join(RESULTS_DIR, f'exp1_mri_{config["name"].lower()}_cnn.keras')
    history_path = os.path.join(RESULTS_DIR, f'exp1_mri_{config["name"].lower()}_history.json')
    
    # Create fresh datasets for this model (prevents iterator exhaustion issues)
    train_ds_norm, val_ds_norm, test_ds_norm = create_datasets()
    
    if os.path.exists(model_path) and os.path.exists(history_path):
        print(f"🚀 Loaded {config['name']} from disk (Skipping training).")
        model = tf.keras.models.load_model(model_path)
        with open(history_path, 'r') as f: history_dict = json.load(f)
        val_loss, val_acc = model.evaluate(val_ds_norm, verbose=0)
        test_loss, test_acc = model.evaluate(test_ds_norm, verbose=0)
        training_time = 0
    else:
        inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))  # ✅ RGB X-Ray images
        x = data_augmentation(inputs)
        
        for f in config['filters']:
            x = layers.Conv2D(f, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(0.001))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(config['dense'], activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.Dropout(config['dropout'])(x)
        outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
        
        model = models.Model(inputs, outputs)
        
        # Removed top_5 accuracy since there are only 2 classes
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        start_time = time.time()
        history = model.fit(train_ds_norm, validation_data=val_ds_norm, epochs=EPOCHS, 
                            callbacks=[callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True),
                                       callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)])
        training_time = time.time() - start_time
        
        # Recreate datasets again for fresh iterators before evaluation
        train_ds_norm, val_ds_norm, test_ds_norm = create_datasets()
        
        val_loss, val_acc = model.evaluate(val_ds_norm, verbose=0)
        test_loss, test_acc = model.evaluate(test_ds_norm, verbose=0)
        history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
        
        model.save(model_path)
        with open(history_path, 'w') as f: json.dump(history_dict, f)
    
    # Recreate datasets one more time for predictions (fresh iterators)
    train_ds_norm, val_ds_norm, test_ds_norm = create_datasets()
            
    # Get predictions for detailed metrics
    test_predictions = model.predict(test_ds_norm, verbose=0)
    test_pred_classes = np.argmax(test_predictions, axis=1)
    
    # Recreate test dataset to collect true labels (iterator was consumed by predict)
    _, _, test_ds_norm = create_datasets()
    test_true_labels = []
    for _, y in test_ds_norm:
        test_true_labels.extend(np.argmax(y.numpy(), axis=1))
    test_true_labels = np.array(test_true_labels)
    
    # Calculate detailed metrics
    test_precision = precision_score(test_true_labels, test_pred_classes, zero_division=0)
    test_recall = recall_score(test_true_labels, test_pred_classes, zero_division=0)
    test_f1 = f1_score(test_true_labels, test_pred_classes, zero_division=0)
    test_cm = confusion_matrix(test_true_labels, test_pred_classes, labels=[0, 1])  # Explicitly specify both classes
    
    experiment1_results[config['name']] = {
        'history': history_dict,
        'val_accuracy': float(val_acc),
        'val_loss': float(val_loss),
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'test_confusion_matrix': test_cm.tolist(),
        'training_time': float(training_time)
    }

# Save comprehensive results summary
results_summary_path = os.path.join(RESULTS_DIR, 'exp1_mri_summary.json')
with open(results_summary_path, 'w') as f:
    json.dump(experiment1_results, f, indent=2)

print("\n" + "="*80)
print("EXPERIMENT 1 SUMMARY - CUSTOM CNN ARCHITECTURES (with Test Set)")
print("="*80)
print(f"{'Model':<12} | {'Val Acc':<8} | {'Test Acc':<8} | {'Precision':<10} | {'Recall':<8} | {'F1':<8}")
print("-"*80)
for model_name, metrics in experiment1_results.items():
    print(f"{model_name:<12} | {metrics['val_accuracy']:.4f}    | {metrics['test_accuracy']:.4f}    | {metrics['test_precision']:.4f}      | {metrics['test_recall']:.4f}   | {metrics['test_f1']:.4f}")
print("="*80)
print("\nDetailed Confusion Matrices (Test Set) - [True Neg, False Pos; False Neg, True Pos]:")
for model_name, metrics in experiment1_results.items():
    cm = metrics['test_confusion_matrix']
    print(f"{model_name}: {cm}")
print("="*80)
print("\n✓ Experiment Complete! Models and results saved to 'results_xray' folder.")