from fastai.vision.all import *
from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import multiprocessing
import csv
import torch
import torch.nn.functional as F
import time

def main():
    # Set environment variable to disable multiprocessing for CUDA
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Set path to your dataset
    x = 'dataset/seg_train/seg_train'
    path = Path(x)
    if not path.exists():
        print(f"Error: Dataset path {path} does not exist.")
        return
    
    print("Contents of directory:")
    print(list(path.glob("*/*.jpg"))[:5])  # Show a few sample images
    print(f"Found {sum(1 for _ in path.glob('*/*.jpg'))} images in dataset")
    
    # Set random seed for reproducibility
    np.random.seed(40)
    torch.manual_seed(40)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(40)
    
    print("Creating DataLoaders...")
    # Create DataLoaders with settings that avoid CUDA pickling errors
    dls = ImageDataLoaders.from_folder(
        path,
        valid_pct=0.2,
        seed=40,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(size=224, min_scale=0.75),
        bs=16 if torch.cuda.is_available() else 8,  # Adjust batch size based on GPU availability
        num_workers=0  # Set to 0 to avoid multiprocessing issues with CUDA
    )
    
    print(f"Classes: {dls.vocab}")
    
    # Create the CNN learner with mixed precision to optimize GPU usage
    print("Creating CNN learner with ResNet18...")
    learn = vision_learner(dls, models.resnet18, metrics=[error_rate, accuracy], path=Path("."))
    
    # Use mixed precision for faster training if on GPU
    if torch.cuda.is_available():
        learn = learn.to_fp16()  # Use mixed precision for better GPU performance
    
    # Find the optimal learning rate
    print("Finding optimal learning rate...")
    start_time = time.time()
    
    # Skip lr_find which causes issues with CUDA and multiprocessing
    lr = 1e-3  # Use a safe default learning rate for ResNet18
    print(f"Using default learning rate: {lr}")
    
    # Train the model
    print("Starting model training...")
    learn.fit_one_cycle(10, lr)
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Save the model
    print("Saving model...")
    learn.save('cnn_model')
    
    # Export the model for inference
    print("Exporting model for inference...")
    learn.export('cnn_model_export.pkl')
    
    # Detailed Prediction Analysis
    print("Generating detailed prediction analysis...")
    
    # Get the validation dataset filenames
    val_files = dls.valid.dataset.items
    
    # Create a CSV to store detailed predictions
    with open('prediction_results.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(['Filename', 'True Label', 'Predicted Label', 'Confidence', 'Correct'])
        
        # Start inference
        start_inference_time = time.time()
        
        # Handle predictions in batches to avoid memory issues
        all_preds = []
        all_targets = []
        
        # Process test set in smaller batches to avoid CUDA memory issues
        for batch_idx, (x, y) in enumerate(dls.valid):
            with torch.no_grad():
                batch_preds = learn.model(x)
                all_preds.append(batch_preds.cpu())
                all_targets.append(y.cpu())
                
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx} batches...")
        
        # Concatenate all predictions and targets
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        
        # Process softmax to get probabilities
        preds = F.softmax(preds, dim=1)
        
        # Process each prediction with actual filename
        for i, (pred, true_label) in enumerate(zip(preds, targets)):
            # Get filename (actual name from dataset)
            filename = os.path.basename(str(val_files[i]))
            
            # Get top prediction
            top_pred_idx = pred.argmax().item()
            top_pred_label = dls.vocab[top_pred_idx]
            true_label_name = dls.vocab[true_label.item()]
            
            # Calculate confidence
            confidence = pred[top_pred_idx].item()
            
            # Check if prediction is correct
            is_correct = top_pred_idx == true_label.item()
            
            # Write to CSV
            csvwriter.writerow([
                filename,
                true_label_name,
                top_pred_label,
                f"{confidence:.4f}",
                is_correct
            ])
    
    inference_time = time.time() - start_inference_time
    print(f"Inference completed in {inference_time:.2f} seconds")
    
    # Load predictions for analysis
    print("Analyzing prediction results...")
    df = pd.read_csv('prediction_results.csv')
    
    # Convert boolean strings to actual booleans
    df['Correct'] = df['Correct'].astype(bool)
    
    # Generate summary statistics
    print("\nPrediction Summary:")
    print(f"Total Predictions: {len(df)}")
    print(f"Correct Predictions: {df['Correct'].sum()}")
    print(f"Accuracy: {df['Correct'].mean():.4f}")
    
    # Per-class performance
    class_performance = df.groupby('True Label')['Correct'].agg(['count', 'mean'])
    class_performance.columns = ['Count', 'Accuracy']
    class_performance.to_csv('class_performance.csv')
    print("\nClass-wise Performance:")
    print(class_performance)
    
    # Confusion Matrix
    print("\nGenerating confusion matrix...")
    confusion_matrix = pd.crosstab(df['True Label'], df['Predicted Label'], normalize='index')
    confusion_matrix.to_csv('confusion_matrix.csv')
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Confidence Distribution
    print("\nAnalyzing confidence distributions...")
    confidence_stats = df.groupby('True Label')['Confidence'].agg(['mean', 'median', 'min', 'max'])
    confidence_stats.to_csv('confidence_stats.csv')
    
    # Visualize confidence distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='True Label', y='Confidence', hue='Correct', data=df)
    plt.title('Confidence Distribution by Class')
    plt.tight_layout()
    plt.savefig('confidence_distribution.png')
    
    # Sample misclassifications for visual inspection
    misclassified = df[~df['Correct']].sort_values('Confidence', ascending=False).head(10)
    misclassified.to_csv('top_misclassifications.csv')
    print("\nTop misclassifications saved to 'top_misclassifications.csv'")
    
    print("\nAnalysis complete. Results saved to CSV files and visualizations.")
    
    # Run inference on a new image
    print("\nTesting inference on new images...")
    inference_test(learn)

def inference_test(learn):
    """Test model inference on a few images from the validation set"""
    try:
        # First ensure model is in evaluation mode
        learn.model.eval()
        
        # Get some sample images from validation set
        val_files = learn.dls.valid.dataset.items[:5]
        
        print("\nSample inference results:")
        with torch.no_grad():  # Disable gradient calculation for inference
            for img_path in val_files:
                # Load image
                img = PILImage.create(img_path)
                
                # Use fastai's built-in prediction method which handles all the transforms properly
                start_time = time.time()
                pred_class, pred_idx, probs = learn.predict(img)
                inference_time = time.time() - start_time
                
                # Display results
                print(f"File: {os.path.basename(str(img_path))}")
                print(f"Prediction: {pred_class}")
                print(f"Probability: {probs[pred_idx].item():.4f}")
                print(f"Inference time: {inference_time*1000:.2f}ms\n")
    
    except Exception as e:
        print(f"Error during inference test: {e}")
        import traceback
        traceback.print_exc()
# Multiprocessing support
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()