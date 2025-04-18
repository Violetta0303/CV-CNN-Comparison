import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dataset_utils import load_icub_world, load_cifar10
from traditional_cv import BoWImageClassifier
from cnn_model import CNNModel
from resnet_model import ResNetModel

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def clean_dataset_name(name):
    name = str(name).strip().lower().replace(" ", "_")
    if name.endswith("_test"):
        name = name[:-5]
    return name

def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    os.makedirs("figures/test", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                fmt='.2f')
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    safe_filename = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in str(filename))
    plt.savefig(f"figures/test/{safe_filename}_confusion_matrix.png")
    plt.close()

def evaluate_model_on_icub(path, class_names, bow_model, cnn_model, label="Unknown", dataset_name='icub', resnet_model=None):
    label = str(label)
    print(f"\n--- Evaluating on {label} ---")
    raw_images, raw_labels, dataset_class_names = load_icub_world(dataset_path=path, version='transformations')
    processed_images, processed_labels = [], []
    img_size = (64, 64)
    for img, lab in zip(raw_images, raw_labels):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] > 3:
            img = img[:, :, :3]
        img_resized = cv2.resize(img, img_size)
        processed_images.append(img_resized)
        processed_labels.append(lab)
    test_images = np.array(processed_images)
    test_labels = np.array(processed_labels)
    test_resized = test_images.transpose(0, 3, 1, 2)

    bow_results = bow_model.evaluate(test_images, test_labels, class_names=dataset_class_names, plot_confusion=False)
    bow_pred = bow_model.predict(test_images)
    plot_confusion_matrix(test_labels, bow_pred, dataset_class_names, f'BoW Confusion Matrix - {dataset_name}', f'bow_{clean_dataset_name(dataset_name)}_test')

    cnn_results = cnn_model.evaluate(test_resized, test_labels, class_names=dataset_class_names, plot_confusion=False)
    cnn_pred = cnn_model.predict(test_resized)
    plot_confusion_matrix(test_labels, cnn_pred, dataset_class_names, f'CNN Confusion Matrix - {dataset_name}', f'cnn_{clean_dataset_name(dataset_name)}_test')

    if resnet_model:
        resnet_results = resnet_model.evaluate(test_resized, test_labels, class_names=dataset_class_names, plot_confusion=False)
        resnet_pred = resnet_model.predict(test_resized)
        plot_confusion_matrix(test_labels, resnet_pred, dataset_class_names, f'ResNet Confusion Matrix - {dataset_name}', f'resnet_{clean_dataset_name(dataset_name)}_test')
    else:
        resnet_results = {m: 0 for m in ['accuracy', 'precision', 'recall', 'f1']}

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    results = {
        'BoW': {m: bow_results[m] for m in metrics},
        'CNN': {m: cnn_results[m] for m in metrics},
        'ResNet': {m: resnet_results[m] for m in metrics}
    }
    return results

def evaluate_model_on_cifar10(bow_model, cnn_model, class_names, resnet_model=None):
    print("\n--- Evaluating on CIFAR-10 ---")
    test_images, test_labels, _ = load_cifar10(split='test', dataset_path='datasets/cifar-10-batches-py')
    resized_images = np.array([cv2.resize(img, (64, 64)) for img in test_images])
    resized_tensor = resized_images.transpose(0, 3, 1, 2)

    bow_results = bow_model.evaluate(resized_images, test_labels, class_names=class_names, plot_confusion=False)
    bow_pred = bow_model.predict(resized_images)
    plot_confusion_matrix(test_labels, bow_pred, class_names, 'BoW Confusion Matrix - CIFAR-10', 'bow_cifar10_test')

    cnn_results = cnn_model.evaluate(resized_tensor, test_labels, class_names=class_names, plot_confusion=False)
    cnn_pred = cnn_model.predict(resized_tensor)
    plot_confusion_matrix(test_labels, cnn_pred, class_names, 'CNN Confusion Matrix - CIFAR-10', 'cnn_cifar10_test')

    if resnet_model:
        resnet_results = resnet_model.evaluate(resized_tensor, test_labels, class_names=class_names, plot_confusion=False)
        resnet_pred = resnet_model.predict(resized_tensor)
        plot_confusion_matrix(test_labels, resnet_pred, class_names, 'ResNet Confusion Matrix - CIFAR-10', 'resnet_cifar10_test')
    else:
        resnet_results = {m: 0 for m in ['accuracy', 'precision', 'recall', 'f1']}

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    results = {
        'BoW': {m: bow_results[m] for m in metrics},
        'CNN': {m: cnn_results[m] for m in metrics},
        'ResNet': {m: resnet_results[m] for m in metrics}
    }
    return results

def export_and_plot_comparison(results_dict, filename_prefix="test"):
    os.makedirs("results/test", exist_ok=True)
    os.makedirs("figures/test", exist_ok=True)
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    csv_path = os.path.join("results", "test", f"{filename_prefix}_comparison.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['Model', 'Test Set'] + metrics
        writer.writerow(header)
        for test_set in results_dict:
            for model in ['BoW', 'CNN', 'ResNet']:
                row = [model, test_set] + [results_dict[test_set][model][m] for m in metrics]
                writer.writerow(row)
    print(f"Results exported to {csv_path}")
    x = np.arange(len(metrics))
    width = 0.25
    for test_set in results_dict:
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, model in enumerate(['BoW', 'CNN', 'ResNet']):
            scores = [results_dict[test_set][model][m] for m in metrics]
            ax.bar(x + (i - 1) * width, scores, width, label=model)
        ax.set_ylabel('Score', fontsize=13)
        ax.set_title(f'Model Comparison on {test_set}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        fig.tight_layout()
        safe_name = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in test_set.lower().replace(' ', '_'))
        fig.savefig(f"figures/test/{filename_prefix}_{safe_name}_comparison.png", dpi=300)
        plt.close()
        print(f"Figure saved for {test_set}")

def run_icubworld_test_evaluation():
    print("\n=== Running Test Evaluation on iCubWorld1.0 ===")
    dataset_root = "./datasets/iCubWorld1.0"
    if not os.path.exists(dataset_root):
        print(f"iCubWorld1.0 dataset not found at {dataset_root}")
        return
    human_test_path = os.path.join(dataset_root, 'human', 'test')
    robot_test_path = os.path.join(dataset_root, 'robot', 'test')

    bow_model = BoWImageClassifier(name='BoW_iCubWorld1.0')
    bow_model.load("models/BoW_iCubWorld1.0.joblib")
    cnn_model = CNNModel(input_shape=(3, 64, 64), num_classes=7, model_type='medium', name="CNN", dataset_name="iCubWorld1.0")
    cnn_model.load("models/CNN_iCubWorld1.0.pth")
    resnet_model = ResNetModel(input_shape=(3, 64, 64), num_classes=7, name="ResNet18", dataset_name="iCubWorld1.0")
    resnet_model.load("models/ResNet18_iCubWorld1.0.pth")

    _, _, class_names = load_icub_world(dataset_path=human_test_path, version='transformations')
    results = {}
    results["Human Test Set"] = evaluate_model_on_icub(human_test_path, class_names, bow_model, cnn_model, label="iCub Human Test", dataset_name="iCub_Human", resnet_model=resnet_model)
    results["Robot Test Set"] = evaluate_model_on_icub(robot_test_path, class_names, bow_model, cnn_model, label="iCub Robot Test", dataset_name="iCub_Robot", resnet_model=resnet_model)
    export_and_plot_comparison(results, filename_prefix="icubworld_test")

def run_cifar10_test_evaluation():
    print("\n=== Running Test Evaluation on CIFAR-10 ===")
    bow_model = BoWImageClassifier(name='BoW_CIFAR10')
    bow_model.load("models/BoW_CIFAR10.joblib")
    cnn_model = CNNModel(input_shape=(3, 64, 64), num_classes=10, model_type='medium', name="CNN", dataset_name="CIFAR10")
    cnn_model.load("models/CNN_CIFAR10.pth")
    resnet_model = ResNetModel(input_shape=(3, 64, 64), num_classes=10, name="ResNet18", dataset_name="CIFAR10")
    resnet_model.load("models/ResNet18_CIFAR10.pth")

    _, _, class_names = load_cifar10(dataset_path='datasets/cifar-10-batches-py')
    results = {}
    results["CIFAR-10 Test Set"] = evaluate_model_on_cifar10(bow_model, cnn_model, class_names, resnet_model=resnet_model)
    export_and_plot_comparison(results, filename_prefix="cifar10_test")

def main():
    print("=== Computer Vision Evaluation (BoW vs CNN vs ResNet) ===")
    run_icubworld_test_evaluation()
    run_cifar10_test_evaluation()
    print("\n=== All Tests Completed ===")

if __name__ == "__main__":
    main()
