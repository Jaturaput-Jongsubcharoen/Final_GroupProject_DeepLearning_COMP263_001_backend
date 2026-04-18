# -*- coding: utf-8 -*-
"""
EXPERIMENT 3: SOTA MODELS (ResNet50)
Chest X-Ray (Pneumonia) Classification
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Prevent GPU OOM crashes
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

BASE_PATH = os.getcwd()

# ✅ Using same dataset structure as Exp1 & Exp2 - train/test only, val split from train
FULL_TRAIN_DIR = os.path.join(BASE_PATH, 'archive', 'chest_xray', 'train')
TEST_DIR = os.path.join(BASE_PATH, 'archive', 'chest_xray', 'test')

RESULTS_DIR = os.path.join(BASE_PATH, 'results_xray')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 224x224 is the native size for ResNet50
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
EPOCHS_SCRATCH = 25  # Extended epochs for fixed from-scratch training
NUM_CLASSES = 2  # Normal vs. Pneumonia
SEED = 42
tf.random.set_seed(SEED)

print("=" * 80)
print("EXPERIMENT 3: RESNET-50 (CHEST X-RAY PNEUMONIA)")
print("=" * 80)

# 1. Data Pipeline with 80/20 train/val split
preprocess_input = tf.keras.applications.resnet50.preprocess_input
AUTOTUNE = tf.data.AUTOTUNE

def create_datasets():
    """Create fresh dataset objects with proper 80/20 train/val split. Avoids iterator exhaustion."""
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
    
    train_sota = train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_sota = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_sota = test_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    
    return train_sota, val_sota, test_sota, train_ds

# Standard augmentation for Transfer Learning
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"), 
    layers.RandomRotation(0.1), 
    layers.RandomZoom(0.1)
])

# Heavy augmentation for From-Scratch
data_augmentation_heavy = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.2),
])

# 2. ResNet50 TRANSFER LEARNING vs FROM-SCRATCH
transfer_path = os.path.join(RESULTS_DIR, 'exp3_xray_resnet_transfer.keras')
transfer_hist_path = os.path.join(RESULTS_DIR, 'exp3_xray_transfer_history.json')

# Renamed to force retraining only for this model
scratch_path = os.path.join(RESULTS_DIR, 'exp3_xray_resnet_scratch_FIXED.keras')
scratch_hist_path = os.path.join(RESULTS_DIR, 'exp3_xray_scratch_history_FIXED.json')

results_summary = {'transfer_learning': {}, 'from_scratch': {}}

# TRANSFER LEARNING MODEL
print("\n" + "="*80)
print("TRANSFER LEARNING: ResNet50 (ImageNet Weights)")
print("="*80)

# Create fresh datasets for this model
train_sota, val_sota, test_sota, _ = create_datasets()

if os.path.exists(transfer_path):
    print("✓ Loaded ResNet50 Transfer Learning model from disk.")
    model_transfer = tf.keras.models.load_model(transfer_path)
    with open(transfer_hist_path, 'r') as f:
        hist_transfer = json.load(f)
else:
    print("Training ResNet50 with ImageNet weights...")
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = data_augmentation(inputs)
    
    # Load ResNet50 with ImageNet weights
    resnet_base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', pooling='avg')
    x = resnet_base(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
    
    model_transfer = models.Model(inputs, outputs)
    model_transfer.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    hist_transfer = model_transfer.fit(
        train_sota, 
        validation_data=val_sota, 
        epochs=EPOCHS, 
        callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )
    
    model_transfer.save(transfer_path)
    with open(transfer_hist_path, 'w') as f: 
        json.dump({k: [float(v) for v in vals] for k, vals in hist_transfer.history.items()}, f)

# Recreate datasets for fresh evaluation
train_sota, val_sota, test_sota, _ = create_datasets()

# Evaluate on test set
test_loss_transfer, test_acc_transfer = model_transfer.evaluate(test_sota, verbose=0)

# Recreate test dataset (iterator was consumed by evaluate)
_, _, test_sota, _ = create_datasets()

# Get predictions for detailed metrics
test_predictions_transfer = model_transfer.predict(test_sota, verbose=0)
test_pred_classes_transfer = np.argmax(test_predictions_transfer, axis=1)

# Recreate test dataset again (iterator consumed by predict)
_, _, test_sota, _ = create_datasets()

# Collect true labels from test set
test_true_labels = []
for _, y in test_sota:
    test_true_labels.extend(np.argmax(y.numpy(), axis=1))
test_true_labels = np.array(test_true_labels)

# Calculate detailed metrics for transfer learning
test_precision_transfer = precision_score(test_true_labels, test_pred_classes_transfer, zero_division=0)
test_recall_transfer = recall_score(test_true_labels, test_pred_classes_transfer, zero_division=0)
test_f1_transfer = f1_score(test_true_labels, test_pred_classes_transfer, zero_division=0)
test_cm_transfer = confusion_matrix(test_true_labels, test_pred_classes_transfer, labels=[0, 1])

results_summary['transfer_learning']['test_accuracy'] = float(test_acc_transfer)
results_summary['transfer_learning']['test_loss'] = float(test_loss_transfer)
results_summary['transfer_learning']['test_precision'] = float(test_precision_transfer)
results_summary['transfer_learning']['test_recall'] = float(test_recall_transfer)
results_summary['transfer_learning']['test_f1'] = float(test_f1_transfer)
results_summary['transfer_learning']['test_confusion_matrix'] = test_cm_transfer.tolist()

print(f"✓ Transfer Learning - Test Accuracy: {test_acc_transfer:.4f}, Precision: {test_precision_transfer:.4f}, Recall: {test_recall_transfer:.4f}, F1: {test_f1_transfer:.4f}")

# FROM-SCRATCH MODEL (No pre-training) - FIXED
print("\n" + "="*80)
print("FROM-SCRATCH: ResNet50 (Random Initialization) - FIXED")
print("="*80)

# Create fresh datasets
train_sota, val_sota, test_sota, train_ds_raw = create_datasets()

if os.path.exists(scratch_path):
    print("✓ Loaded FIXED ResNet50 From-Scratch model from disk.")
    model_scratch = tf.keras.models.load_model(scratch_path)
    with open(scratch_hist_path, 'r') as f:
        hist_scratch = json.load(f)
else:
    print("Training ResNet50 from scratch (FIXED VERSION)...")
    
    # Calculate class weights to handle imbalance
    print("Calculating class weights...")
    all_train_labels = []
    for _, y in train_ds_raw:
        all_train_labels.extend(np.argmax(y.numpy(), axis=1))
    all_train_labels = np.array(all_train_labels)
    
    unique_classes = np.unique(all_train_labels)
    class_weights_array = compute_class_weight('balanced', classes=unique_classes, y=all_train_labels)
    class_weights_dict = {i: class_weights_array[i] for i in range(NUM_CLASSES)}
    print(f"Applied Class Weights: {class_weights_dict}")

    inputs_scratch = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x_scratch = data_augmentation_heavy(inputs_scratch)
    
    # Load ResNet50 WITHOUT pre-training
    resnet_scratch = tf.keras.applications.ResNet50(include_top=False, weights=None, pooling='avg')
    x_scratch = resnet_scratch(x_scratch)
    
    # Improved classifier head with Batch Normalization
    x_scratch = layers.BatchNormalization()(x_scratch)
    x_scratch = layers.Dense(1024, activation='relu')(x_scratch)
    x_scratch = layers.BatchNormalization()(x_scratch)
    x_scratch = layers.Dropout(0.5)(x_scratch)
    x_scratch = layers.Dense(512, activation='relu')(x_scratch)
    x_scratch = layers.BatchNormalization()(x_scratch)
    x_scratch = layers.Dropout(0.5)(x_scratch)
    outputs_scratch = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x_scratch)
    
    model_scratch = models.Model(inputs_scratch, outputs_scratch)
    
    # Lower Learning Rate
    model_scratch.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    # Longer Patience
    cb_scratch = [
        callbacks.EarlyStopping(patience=7, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]
    
    hist_scratch = model_scratch.fit(
        train_sota, 
        validation_data=val_sota, 
        epochs=EPOCHS_SCRATCH, 
        class_weight=class_weights_dict, 
        callbacks=cb_scratch,
        verbose=1
    )
    
    model_scratch.save(scratch_path)
    with open(scratch_hist_path, 'w') as f: 
        json.dump({k: [float(v) for v in vals] for k, vals in hist_scratch.history.items()}, f)

# Recreate datasets for fresh evaluation
train_sota, val_sota, test_sota, _ = create_datasets()

# Evaluate on test set
test_loss_scratch, test_acc_scratch = model_scratch.evaluate(test_sota, verbose=0)

# Recreate test dataset (iterator was consumed by evaluate)
_, _, test_sota, _ = create_datasets()

# Get predictions for detailed metrics
test_predictions_scratch = model_scratch.predict(test_sota, verbose=0)
test_pred_classes_scratch = np.argmax(test_predictions_scratch, axis=1)

# Recreate test dataset again (iterator consumed by predict)
_, _, test_sota, _ = create_datasets()

# Collect true labels for scratch model evaluation
test_true_labels_scratch = []
for _, y in test_sota:
    test_true_labels_scratch.extend(np.argmax(y.numpy(), axis=1))
test_true_labels_scratch = np.array(test_true_labels_scratch)

# Calculate detailed metrics for from-scratch
test_precision_scratch = precision_score(test_true_labels_scratch, test_pred_classes_scratch, zero_division=0)
test_recall_scratch = recall_score(test_true_labels_scratch, test_pred_classes_scratch, zero_division=0)
test_f1_scratch = f1_score(test_true_labels_scratch, test_pred_classes_scratch, zero_division=0)
test_cm_scratch = confusion_matrix(test_true_labels_scratch, test_pred_classes_scratch, labels=[0, 1])

results_summary['from_scratch']['test_accuracy'] = float(test_acc_scratch)
results_summary['from_scratch']['test_loss'] = float(test_loss_scratch)
results_summary['from_scratch']['test_precision'] = float(test_precision_scratch)
results_summary['from_scratch']['test_recall'] = float(test_recall_scratch)
results_summary['from_scratch']['test_f1'] = float(test_f1_scratch)
results_summary['from_scratch']['test_confusion_matrix'] = test_cm_scratch.tolist()

print(f"✓ From-Scratch (FIXED) - Test Accuracy: {test_acc_scratch:.4f}, Precision: {test_precision_scratch:.4f}, Recall: {test_recall_scratch:.4f}, F1: {test_f1_scratch:.4f}")

# Save summary
summary_path = os.path.join(RESULTS_DIR, 'exp3_xray_summary_FIXED.json')
with open(summary_path, 'w') as f:
    json.dump(results_summary, f, indent=2)

def load_history(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        print(f"Could not find {filepath}")
        return None

transfer_hist = load_history(transfer_hist_path)
scratch_hist = load_history(scratch_hist_path)

if transfer_hist and scratch_hist:
    print("✓ Successfully loaded history files. Generating plots...")
    
    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experiment 3: ResNet50 Transfer Learning vs. From-Scratch (Fixed)\nChest X-Ray (Pneumonia) Classification', fontsize=18, y=0.95)

    # --- ROW 1: TRANSFER LEARNING ---
    axes[0, 0].plot(transfer_hist['accuracy'], label='Train Accuracy', color='blue', linewidth=2)
    axes[0, 0].plot(transfer_hist['val_accuracy'], label='Validation Accuracy', color='cyan', linewidth=2, linestyle='--')
    axes[0, 0].set_title('Transfer Learning: Accuracy', fontsize=14)
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].grid(True, linestyle=':', alpha=0.7)

    axes[0, 1].plot(transfer_hist['loss'], label='Train Loss', color='red', linewidth=2)
    axes[0, 1].plot(transfer_hist['val_loss'], label='Validation Loss', color='salmon', linewidth=2, linestyle='--')
    axes[0, 1].set_title('Transfer Learning: Loss', fontsize=14)
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, linestyle=':', alpha=0.7)

    # --- ROW 2: FROM-SCRATCH FIXED ---
    axes[1, 0].plot(scratch_hist['accuracy'], label='Train Accuracy', color='green', linewidth=2)
    axes[1, 0].plot(scratch_hist['val_accuracy'], label='Validation Accuracy', color='lightgreen', linewidth=2, linestyle='--')
    axes[1, 0].set_title('From-Scratch (Fixed): Accuracy', fontsize=14)
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend(loc='lower right')
    axes[1, 0].grid(True, linestyle=':', alpha=0.7)

    axes[1, 1].plot(scratch_hist['loss'], label='Train Loss', color='purple', linewidth=2)
    axes[1, 1].plot(scratch_hist['val_loss'], label='Validation Loss', color='violet', linewidth=2, linestyle='--')
    axes[1, 1].set_title('From-Scratch (Fixed): Loss', fontsize=14)
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92]) 
    
    # Save the plot
    plot_path = os.path.join(RESULTS_DIR, 'exp3_comparison_plot_FIXED.png')
    plt.savefig(plot_path, dpi=300)
    print(f"✓ Beautiful graph saved successfully to: {plot_path}")
    
else:
    print("❌ Error: Missing history files. Make sure you run this script in the same folder as your results_xray folder.")

print("\n" + "="*80)
print("EXPERIMENT 3 SUMMARY - ResNet50 (TRANSFER vs FROM-SCRATCH FIXED)")
print("="*80)
print(f"{'Model':<20} | {'Test Acc':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
print("-"*80)
print(f"{'Transfer Learning':<20} | {test_acc_transfer:.4f}     | {test_precision_transfer:.4f}     | {test_recall_transfer:.4f}     | {test_f1_transfer:.4f}")
print(f"{'From-Scratch':<20} | {test_acc_scratch:.4f}     | {test_precision_scratch:.4f}     | {test_recall_scratch:.4f}     | {test_f1_scratch:.4f}")
print("="*80)
print(f"\nTransfer Learning Advantage: {(test_acc_transfer - test_acc_scratch)*100:.2f}%")
print("\nConfusion Matrices (Test Set) - [True Neg, False Pos; False Neg, True Pos]:")
print(f"Transfer Learning: {test_cm_transfer.tolist()}")
print(f"From-Scratch (Fixed): {test_cm_scratch.tolist()}")
print("="*80)