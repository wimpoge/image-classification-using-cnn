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

def main():
    # Set path to your dataset
    x = 'dataset/seg_train/seg_train'
    path = Path(x)
    print("Contents of directory:")
    print(path.ls())
    
    # Set random seed
    np.random.seed(40)
    
    # Create DataLoaders (using fastai v2 syntax)
    dls = ImageDataLoaders.from_folder(
        path,
        valid_pct=0.2,
        seed=40,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(),
        num_workers=0
    )
    
    # Create the CNN learner
    learn = vision_learner(dls, models.resnet18, metrics=error_rate, path=Path("."))
    
    # Find the optimal learning rate
    lr_finder = learn.lr_find()
    
    # Get suggested learning rate
    try:
        suggested_lr = lr_finder.valley if hasattr(lr_finder, 'valley') else 1e-2
        print(f"Suggested learning rate: {suggested_lr}")
        lr = suggested_lr
    except:
        lr = 1e-2
        print(f"Using default learning rate: {lr}")
    
    # Train the model
    print("Starting model training...")
    learn.fit_one_cycle(10, lr)
    
    # Save the model
    learn.save('cnn_model')
    
    # Detailed Prediction Analysis
    print("Generating detailed prediction analysis...")
    
    # Create a CSV to store detailed predictions
    with open('prediction_results.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(['Filename', 'True Label', 'Predicted Label', 'Confidence', 'Correct'])
        
        # Iterate through validation data
        for x, y in dls.valid:
            preds = learn.get_preds(dl=[(x, y)])
            
            # Process each prediction
            for i, (pred, true_label) in enumerate(zip(preds[0], y)):
                # Get top prediction
                top_pred_idx = pred.argmax().item()
                top_pred_label = dls.vocab[top_pred_idx]
                true_label_name = dls.vocab[true_label.item()]
                
                # Calculate confidence
                confidence = pred[top_pred_idx].item()
                
                # Check if prediction is correct
                is_correct = top_pred_label == true_label_name
                
                # Get filename (this might need adjustment based on your exact data loading)
                filename = f"sample_{i}"
                
                # Write to CSV
                csvwriter.writerow([
                    filename, 
                    true_label_name, 
                    top_pred_label, 
                    f"{confidence:.4f}", 
                    is_correct
                ])
    
    # Analyze and summarize results
    df = pd.read_csv('prediction_results.csv')
    
    # Generate summary statistics
    print("\nPrediction Summary:")
    print(f"Total Predictions: {len(df)}")
    print(f"Correct Predictions: {df['Correct'].sum()}")
    print(f"Accuracy: {df['Correct'].mean():.4f}")
    
    # Confusion Matrix
    confusion_matrix = pd.crosstab(df['True Label'], df['Predicted Label'], normalize='index')
    confusion_matrix.to_csv('confusion_matrix.csv')
    print("\nConfusion Matrix saved to 'confusion_matrix.csv'")
    
    # Confidence Distribution
    confidence_stats = df.groupby('True Label')['Confidence'].agg(['mean', 'median', 'min', 'max'])
    confidence_stats.to_csv('confidence_stats.csv')
    print("\nConfidence Statistics saved to 'confidence_stats.csv'")

# Multiprocessing support for Windows
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()