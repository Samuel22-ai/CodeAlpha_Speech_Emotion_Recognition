import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

def evaluate_model(
    model_path: str, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    label_encoder_path: str = "../models/label_encoder.joblib"
) -> None:
    """
    Evaluates a trained Convolutional Recurrent Neural Network (CRNN) by generating 
    predictions on unseen test data, printing a classification report, and displaying 
    a confusion matrix heatmap.

    This function automatically loads the saved model and the corresponding LabelEncoder 
    to map the model's integer predictions back to their human-readable string labels 
    (e.g., 'happy', 'sad'). It is specifically designed to handle models trained with 
    custom loss functions by loading the model uncompiled and recompiling it with 
    standard metrics for evaluation.

    Args:
        model_path (str): The file path to the saved trained TensorFlow/Keras model (.keras).
        X_test (np.ndarray): The testing feature tensor of shape (samples, features, time_steps, channels).
        y_test (np.ndarray): The one-hot encoded ground truth labels for the test set.
        label_encoder_path (str, optional): The relative or absolute path to the saved 
                                            LabelEncoder. Defaults to "../models/label_encoder.joblib".

    Raises:
        FileNotFoundError: If the specified model or label encoder file paths do not exist.

    Returns:
        None: The function prints the evaluation metrics and classification report to the 
              console and renders a matplotlib/seaborn heatmap.
        
    Example:
        >>> from src.model_evaluation import evaluate_model
        >>> evaluate_model("../models/best_cnn_model.keras", X_test_final, y_test)
    """
    
    # ==========================================
    # 1. LOAD AND PREPARE THE MODEL
    # ==========================================
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Check your path.")
        
    print(f"Loading model from {model_path}...")
    try:
        # Step 1: Force load the model without its saved compilation state.
        # Deep learning models often use custom loss functions (like DynamicFocalLoss) 
        # during training. By setting compile=False, we bypass the need to define and 
        # inject those custom functions just to run inference/evaluation.
        model = load_model(model_path, compile=False) 
        
        # Step 2: Manually recompile it with standard metrics.
        # model.evaluate() requires a compiled model. We use standard categorical crossentropy
        # and accuracy here just to make the test run. This DOES NOT change the model's 
        # pre-trained weights or the final confusion matrix results.
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        print("Model loaded and recompiled successfully.")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Print the model architecture to verify correct loading
    model.summary()
    
    # Run the standard Keras evaluation to get bulk loss and accuracy
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # ==========================================
    # 2. LOAD LABEL ENCODER FOR DECODING
    # ==========================================
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder not found at {label_encoder_path}. Check your path.")
        
    # We need the exact class mapping (e.g., 0='angry', 1='calm') established during training
    label_encoder = joblib.load(label_encoder_path)
    emotion_classes = label_encoder.classes_
    
    # ==========================================
    # 3. GENERATE PREDICTIONS
    # ==========================================
    print("Generating predictions on unseen test data...")
    # The model outputs a probability distribution across all classes for each sample
    y_pred_probs = model.predict(X_test)
    
    # Convert probabilities (e.g., [0.1, 0.8, 0.1...]) to single integer class IDs
    # np.argmax finds the index of the highest probability.
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    # Convert one-hot encoded true labels back to single integer class IDs for comparison
    y_true_classes = np.argmax(y_test, axis=1)
    
    # ==========================================
    # 4. CLASSIFICATION REPORT & CONFUSION MATRIX
    # ==========================================
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    # The classification report provides Precision, Recall, and F1-Score per emotion class
    print(classification_report(y_true_classes, y_pred_classes, target_names=emotion_classes))
    
    # Compute the confusion matrix to see exactly which emotions are being confused for others
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Render the heatmap using Seaborn
    plt.figure(figsize=(10, 8))
    # plt.suptitle(f"Confusion Matrix Heatmap for Model {str(os.path.basename(model_path)).split('_')[-1].split('.')[0].upper()}", fontsize=16, y=1, fontweight='bold')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_classes, 
                yticklabels=emotion_classes,
                cbar=False,
                linewidths=0.5,
                linecolor='gray')
    
    plt.title('Subject-Independent Confusion Matrix (Unseen Actors)', fontsize=14, pad=15)
    plt.ylabel('Actual Emotion (Ground Truth)', fontsize=12)
    plt.xlabel('Predicted Emotion (Model Output)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_training_history(csv_path: str = "../models/training_log.csv") -> None:
    """
    Reads the CSV logger output generated during Keras training and plots the 
    training and validation accuracy/loss curves over epochs.

    This visual check is crucial for diagnosing model fitting issues such as 
    overfitting (where training accuracy continues to rise while validation 
    accuracy drops) or underfitting.

    Args:
        csv_path (str, optional): The path to the training log CSV file. 
                                  Defaults to "../models/training_log.csv".

    Returns:
        None: Renders a matplotlib figure with two side-by-side subplots (Accuracy and Loss).
        
    Example:
        >>> from src.model_evaluation import plot_training_history
        >>> plot_training_history()
    """
    if not os.path.exists(csv_path):
        print(f"Warning: Training log not found at {csv_path}.")
        return
        
    # Load the training history from the CSV file
    history = pd.read_csv(csv_path)
    
    plt.figure(figsize=(14, 5))
    
    # ==========================================
    # 1. PLOT ACCURACY CURVE
    # ==========================================
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['accuracy'], label='Train Accuracy', linewidth=2)
    # Validation metrics are only plotted if they exist in the callback log
    if 'val_accuracy' in history.columns:
        plt.plot(history['epoch'], history['val_accuracy'], label='Validation Accuracy', linewidth=2, linestyle='--')
    plt.title('Model Accuracy over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # ==========================================
    # 2. PLOT LOSS CURVE
    # ==========================================
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history.columns:
        plt.plot(history['epoch'], history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--')
    plt.title('Model Loss over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.show()