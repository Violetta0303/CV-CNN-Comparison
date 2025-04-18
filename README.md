# CV-CNN-Comparison: Bag-of-Words vs Convolutional Neural Networks for Object Recognition

This project presents a comparative study of traditional Computer Vision (CV) techniques using the Bag-of-Words (BoW) model with SIFT features and Support Vector Machines (SVM), versus modern Convolutional Neural Networks (CNNs) for image classification.

The evaluation is carried out on two datasets:

- **iCubWorld1.0**: Contains RGB images of objects captured by a robot (robot) and a human demonstrator (human), including various categories such as bottle, box, octopus, etc.
- **CIFAR-10**: A standard benchmark dataset with 60,000 colour images across 10 classes (e.g. airplane, automobile, bird, etc.).

---

## Project Structure

```
CV-CNN-Comparison/
├── cnn_model.py               # PyTorch CNN model implementation
├── traditional_cv.py         # BoW + SIFT + SVM classifier
├── model_base.py             # Base class shared by both models
├── dataset_utils.py          # Dataset loading and preprocessing utilities
├── download_icubworld.py     # iCubWorld1.0 downloader
├── experiment_runner.py      # Performs 5-fold CV training & saves comparison plots
├── test.py                   # Evaluates trained models on test sets & outputs results
├── results/                  # CSV outputs of metrics (per dataset/model)
├── figures/                  # Visualisations (confusion matrices, comparison charts)
└── models/                   # Saved BoW/CNN model files
```

---

## Dataset Details

### iCubWorld1.0

- Contains RGB images in:
  - `human/train`, `human/test`, `robot/train`, `robot/test`
- Object Categories:
  - bottle, box, octopus, phone, pouch, spray, turtle

### CIFAR-10

- 10 RGB categories:
  - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Automatically downloaded in Python version format under:
  - `datasets/cifar-10-batches-py/`

---

## Features

- **BoW Model**:
  - Feature Extraction: SIFT / ORB / BRISK
  - Clustering: KMeans
  - Classifier: SVM (GridSearchCV)

- **CNN Model**:
  - Implemented in PyTorch
  - Medium-sized custom CNN

- **Metrics**:
  - Accuracy, Precision, Recall, F1 Score
  - Inference & Training Time

- **Visualisation**:
  - Confusion matrices
  - Cross-validation (CV) metric curves
  - Training time comparisons

- **Organised Outputs**:
  - `results/<dataset>/...csv`
  - `figures/<dataset>/...png`
  - `results/test/`, `figures/test/` for final evaluations

---

## How to Run

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
python download_icubworld.py
```

```bash
python download_cifar10.py
```

### 3. Train Models via Cross-Validation

```bash
python experiment_runner.py
```

This will:
- Train and cross-validate BoW and CNN models on both datasets
- Save trained models under `models/`
- Save evaluation results in `results/` and `figures/`

### 4. Run Final Evaluation on Test Sets

```bash
python test.py
```

This will:
- Evaluate trained models on:
  - iCub Human Test Set
  - iCub Robot Test Set
  - CIFAR-10 Test Set
- Output confusion matrices and metric comparisons
- Save results to `results/test/` and `figures/test/`

---

## Output Examples

### Cross-Validation
```
figures/CIFAR10/bow_vs_cnn.png
figures/CIFAR10/bow_vs_cnn_times.png
results/CIFAR10/bow_vs_cnn.csv
results/CIFAR10/bow_vs_cnn_times.csv
```

### Test Evaluation
```
figures/test/cnn_cifar10_test_confusion_matrix.png
figures/test/bow_cifar10_test_confusion_matrix.png
figures/test/cifar10_test_comparison.png
results/test/cifar10_test_comparison.csv

figures/test/cnn_icub_human_test_confusion_matrix.png
figures/test/bow_icub_robot_test_confusion_matrix.png
figures/test/icubworld_test_comparison.png
results/test/icubworld_test_comparison.csv
```

---

## Notes

- All CNN inputs are resized to 64x64 RGB.
- All results are clearly separated by dataset (CIFAR-10 vs iCub).
- BoW and CNN results are evaluated with identical cross-validation folds.
- All code is modular and extensible.

---

## License

MIT License
