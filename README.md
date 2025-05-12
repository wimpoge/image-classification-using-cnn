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
pip install fastai pandas numpy matplotlib seaborn opencv-python
```

## Usage

1. Download the dataset from the Kaggle URL above
2. Place the dataset in the correct directory structure
3. Run the script:

```bash
python image_classification_analysis.py
```

## Features

This script provides several analysis tools:

- **Training**: Trains a ResNet18 model on the dataset
- **Performance Analysis**: Generates comprehensive CSV reports of model performance
- **Detailed Predictions**: Creates CSV files with individual prediction results including:
  - Actual filenames matched with predictions
  - Confidence scores for each prediction
  - Class-wise accuracy metrics
  - Misclassification patterns

## Output Files

The script generates several CSV files for analysis:

- `prediction_results.csv`: Complete prediction results for all images
- `misclassified_images.csv`: Only the incorrectly classified images
- `class_accuracy.csv`: Breakdown of accuracy by class
- `common_misclassifications.csv`: Common patterns of misclassification
- `confusion_matrix.csv`: Full confusion matrix for all classes

## Model Parameters

- Architecture: ResNet18
- Image Size: 224×224 pixels
- Validation Split: 20%
- Data Augmentation: fastai's default augmentation transforms
- Training: 10 epochs using the one-cycle policy
- Learning Rate: Automatically determined using the lr_finder

## Extending the Project

You can modify the script to:
- Use different architectures (change `models.resnet18` to another model)
- Add more epochs (change the number in `learn.fit_one_cycle(10, lr)`)
- Adjust image size (change values in `Resize(224)`)
- Create visualizations by adding matplotlib code

## Troubleshooting

If you encounter issues:

- Ensure you downloaded the correct dataset
- Verify the folder structure matches the expected path
- Check that all required libraries are installed
- On Windows, ensure the multiprocessing freeze_support() is included