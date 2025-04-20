# CV-CNN-Comparison: Traditional Computer Vision Methods vs CNN models for Object Recognition

This project presents a comparative study between traditional computer vision (CV) techniques and deep learning models (Customised CNN and ResNet18) for image classification. The traditional approach leverages the Bag-of-Words (BoW) model using SIFT features and SVM classifiers. In contrast, the deep learning pipeline includes a custom-built convolutional neural network (CNN) and a transfer learning approach using ResNet18.

Evaluations are conducted on two datasets:

- **iCubWorld1.0**: A robotics-centric dataset featuring RGB images of objects demonstrated by both humans and robots across 7 categories.
- **CIFAR-10**: A widely used benchmark dataset containing 60,000 32x32 colour images from 10 classes.

---

## Project Structure

```
CV-CNN-Comparison/
├── download_icubworld.py     # Script to download iCubWorld1.0
├── download_cifar10.py       # Script to download CIFAR-10
├── dataset_utils.py          # Dataset preprocessing
├── cnn_model.py               # Custom PyTorch CNN model
├── resnet_model.py            # ResNet18 architecture with fine-tuning
├── traditional_cv.py         # BoW + SIFT + SVM pipeline
├── experiment_runner.py      # Master runner for cross-validation
├── experiment_cnn.py         # CNN-specific experiments
├── experiment_resnet.py      # ResNet-specific experiments
├── experiment_cv.py          # Traditional CV experiments
├── experiment_cv_cnn_resnet.py # Combined CV + CNN + ResNet runner
├── test.py                   # Final test evaluation
├── results/                  # All metrics (CSV format)
├── figures/                  # All visual outputs (confusion matrices, curves, charts)
└── models/                   # Saved model files
```

---

## Datasets

### iCubWorld1.0

- Structure: `human/train`, `human/test`, `robot/train`, `robot/test`
- Categories: bottle, box, octopus, phone, pouch, spray, turtle

### CIFAR-10

- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Automatically downloaded under: `datasets/cifar-10-batches-py/`

---

## Approaches

### Traditional Computer Vision (BoW)

- Feature Extraction: SIFT, ORB, BRISK
- Encoding: KMeans clustering
- Classification: SVM with hyperparameter tuning (GridSearchCV)

### Customised CNN

- Two convolutional layers + fully connected layers
- Lightweight and interpretable architecture
- Designed for computational efficiency with solid performance

### ResNet18

- Transfer learning with pretrained ResNet18
- Fine-tuned on both datasets
- Deeper, more expressive architecture

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- AUC (ROC)
- Training & Inference Time
- Visualisation:
  - Confusion Matrices
  - ROC Curves
  - Training Loss History
  - Cross-Validation Metrics

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
python download_icubworld.py
python download_cifar10.py
```

### 3. Run Cross-Validation for All Models

```bash
python experiment_runner.py
```

This will:
- Perform 5-fold CV on BoW, CNN, and ResNet models
- Save results under `results/` and visualisations under `figures/`

### 4. Run Test Evaluation

```bash
python test.py
```

This will:
- Evaluate final trained models on:
  - iCubWorld1.0 Human Test Set
  - iCubWorld1.0 Robot Test Set
  - CIFAR-10 Test Set
- Output final metrics and confusion matrices to `results/test/` and `figures/test/`

---

## Output Examples

### Cross-Validation
```
results/CIFAR10/bow_vs_cnn_vs_resnet.csv
results/iCubWorld1.0/bow_vs_cnn_vs_resnet.csv
figures/CIFAR10/bow_vs_cnn_vs_resnet.png
figures/iCubWorld1.0/bow_vs_cnn_vs_resnet.png
```

### Test Evaluation
```
results/test/icubworld_test_comparison.csv
results/test/cifar10_test_comparison.csv
figures/test/icubworld_test_human_test_set_comparison.png
figures/test/cifar10_test_comparison.png
```

---

## Summary of Findings

- On both datasets, **CNN and ResNet18 consistently outperform traditional BoW** methods in accuracy, precision, recall, and F1.
- **ResNet18 achieves the best overall performance**, especially on iCubWorld1.0, due to its deep architecture and strong generalisation.
- **Customised CNN** performs better than ResNet18 on CIFAR-10, making it a strong and efficient alternative.
- **BoW**, while interpretable and fast on small datasets, shows clear limitations when dealing with noisy or complex image distributions, such as those in CIFAR-10.

---

## License

MIT License
