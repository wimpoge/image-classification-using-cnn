# Image Classification CNN Model

This repository contains code for a CNN-based image classification model that categorizes images into six classes: buildings, forest, glacier, mountain, sea, and street.

## Project Overview

This project uses the fastai library with a ResNet18 architecture to classify natural and urban scene images. The model achieves high accuracy and includes comprehensive analysis tools to evaluate performance across all classes.

## Dataset

The project uses the Intel Image Classification dataset, available on Kaggle:
- [Intel Image Classification Dataset](https://www.kaggle.com/code/arbazkhan971/image-classification-using-cnn-94-accuracy/input)

**You must download the dataset before running the code.** The dataset contains approximately 7,000 training images divided into six categories.

### Dataset Structure

After downloading, organize your files as follows:
```
your_project_directory/
├── dataset/
│   ├── seg_train/
│   │   └── seg_train/
│   │       ├── buildings/
│   │       ├── forest/
│   │       ├── glacier/
│   │       ├── mountain/
│   │       ├── sea/
│   │       └── street/
│   ├── seg_test/
│   └── seg_pred/
└── image_classification_analysis.py
```

## Installation

1. Clone this repository
2. Create a virtual environment (recommended)
3. Install the required packages:
```bash
pip install fastai pandas numpy matplotlib seaborn opencv-python torch
```

## Usage

1. Download the dataset from the Kaggle URL above
2. Place the dataset in the correct directory structure
3. Run the script:
```bash
python image_classification_analysis.py
```

## Features

- **CUDA Optimization**: Configures environment for optimal GPU usage
- **Device Detection**: Automatically uses GPU if available, with fallback to CPU
- **Training**: Trains a ResNet18 model on the dataset with reproducible results (fixed seed)
- **Performance Analysis**: Generates comprehensive reports of model performance
- **Visualization**: Creates visual representations of model performance metrics
- **Inference Testing**: Demonstrates how to use the model for predictions on new images

## Output Files

The script generates several files for analysis:

### CSV Reports
- `prediction_results.csv`: Complete prediction results for all validation images
- `class_performance.csv`: Breakdown of accuracy by class
- `confidence_stats.csv`: Confidence distribution statistics for each class
- `confusion_matrix.csv`: Full confusion matrix for all classes
- `top_misclassifications.csv`: Top 10 most confident incorrect predictions

### Visualizations
- `confusion_matrix.png`: Visual heatmap of the confusion matrix
- `confidence_distribution.png`: Boxplot showing confidence distribution by class

### Model Files
- `cnn_model`: Saved model state for resuming training
- `cnn_model_export.pkl`: Exported model for inference

## Model Parameters

- **Architecture**: ResNet18
- **Image Size**: 224×224 pixels
- **Validation Split**: 20%
- **Batch Size**: 16 on GPU, 8 on CPU
- **Reproducibility**: Fixed seed (40) for reproducible results
- **Training**: 10 epochs using the one-cycle policy
- **Learning Rate**: 1e-3 (default safe value for ResNet18)
- **Optimization**: Mixed precision training on compatible GPUs

## Performance Metrics

The script calculates and reports:
- Overall accuracy
- Class-wise accuracy
- Confusion matrix
- Confidence distribution analysis
- Detailed misclassification analysis

## Extending the Project

You can modify the script to:
- Use different architectures (change `models.resnet18` to another model)
- Add more epochs (change the number in `learn.fit_one_cycle(10, lr)`)
- Adjust image size (change values in `Resize(224)`)
- Implement additional analysis by extending the visualization code

## GPU Optimization

The script includes several optimizations for GPU usage:
- Sets `CUDA_LAUNCH_BLOCKING=1` to prevent asynchronous errors
- Uses mixed precision training (`learn.to_fp16()`) for compatible GPUs
- Adjusts batch size based on available hardware
- Sets `num_workers=0` to avoid CUDA multiprocessing issues

## Troubleshooting

If you encounter issues:
- Ensure you downloaded the correct dataset
- Verify the folder structure matches the expected path
- Check that all required libraries are installed
- On Windows, ensure the multiprocessing freeze_support() is included
- For memory issues, reduce batch size by modifying the `bs` parameter
- If experiencing CUDA out-of-memory errors, try reducing image size or batch size