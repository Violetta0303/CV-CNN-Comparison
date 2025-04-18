import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from dataset_utils import (
    load_icub_world,
    create_balanced_subset
)
from cnn_model import CNNModel
from resnet_model import ResNetModel
from traditional_cv import BoWImageClassifier
from download_icubworld import download_icubworld


def plot_cv_comparison(bow_results, cnn_results, resnet_results, fig_path, csv_path):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    bow_means, cnn_means, resnet_means = [], [], []
    bow_stds, cnn_stds, resnet_stds = [], [], []

    for m in metrics:
        bow_means.append(bow_results.get(f'{m}_mean', 0))
        bow_stds.append(bow_results.get(f'{m}_std', 0))
        cnn_means.append(cnn_results.get(f'{m}_mean', 0))
        cnn_stds.append(cnn_results.get(f'{m}_std', 0))
        resnet_means.append(resnet_results.get(f'{m}_mean', 0))
        resnet_stds.append(resnet_results.get(f'{m}_std', 0))

    x = np.arange(len(metrics))
    width = 0.25

    plt.figure(figsize=(12, 8))
    plt.bar(x - width, bow_means, width, yerr=bow_stds, capsize=5, label="BoW", color='royalblue')
    plt.bar(x, cnn_means, width, yerr=cnn_stds, capsize=5, label="CNN", color='darkorange')
    plt.bar(x + width, resnet_means, width, yerr=resnet_stds, capsize=5, label="ResNet", color='forestgreen')

    plt.xticks(x, [m.capitalize() for m in metrics], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1)
    plt.ylabel("Score", fontsize=14)
    plt.title("BoW vs CNN vs ResNet - Cross-Validation Performance", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved performance plot: {fig_path}")
    plt.close()

    with open(csv_path, 'w', newline='') as f:
        f.write('Metric,BoW_Mean,BoW_Std,CNN_Mean,CNN_Std,ResNet_Mean,ResNet_Std\n')
        for i, m in enumerate(metrics):
            f.write(f"{m},{bow_means[i]:.4f},{bow_stds[i]:.4f},"
                    f"{cnn_means[i]:.4f},{cnn_stds[i]:.4f},"
                    f"{resnet_means[i]:.4f},{resnet_stds[i]:.4f}\n")
    print(f"Saved CSV: {csv_path}")


def compare_training_times(bow_results, cnn_results, resnet_results, fig_path, csv_path):
    categories = ['Training Time', 'Inference Time']
    bow_times = [bow_results.get('train_time_mean', 1), bow_results.get('inference_time_mean', 0.1)]
    cnn_times = [cnn_results.get('train_time_mean', 1), cnn_results.get('inference_time_mean', 0.1)]
    resnet_times = [resnet_results.get('train_time_mean', 1), resnet_results.get('inference_time_mean', 0.1)]

    x = np.arange(len(categories))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, bow_times, width, label='BoW', color='royalblue')
    plt.bar(x, cnn_times, width, label='CNN', color='darkorange')
    plt.bar(x + width, resnet_times, width, label='ResNet', color='forestgreen')

    plt.ylabel("Time (seconds)", fontsize=14)
    plt.title("Training & Inference Time Comparison", fontsize=16)
    plt.xticks(x, categories, fontsize=12)
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved time plot: {fig_path}")
    plt.close()

    with open(csv_path, 'w', newline='') as f:
        f.write("Metric,BoW,CNN,ResNet\n")
        for i, cat in enumerate(categories):
            f.write(f"{cat},{bow_times[i]:.4f},{cnn_times[i]:.4f},{resnet_times[i]:.4f}\n")
    print(f"Saved time CSV: {csv_path}")


def run_icubworld_cv_experiment():
    print("\n=== Running iCubWorld1.0 CV Experiment (BoW vs CNN vs ResNet) ===")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_root = download_icubworld(dest_folder='./datasets', version='1.0')
    human_path = os.path.join(dataset_root, 'human', 'train')
    robot_path = os.path.join(dataset_root, 'robot', 'train')

    human_images, human_labels, class_names = load_icub_world(human_path, version='1.0')
    robot_images, robot_labels, _ = load_icub_world(robot_path, version='1.0')

    images = np.concatenate((human_images, robot_images), axis=0)
    labels = np.concatenate((human_labels, robot_labels), axis=0)
    images, labels = create_balanced_subset(images, labels, n_per_class=400)
    resized_images = np.array([cv2.resize(img, (64, 64)) for img in images])

    dataset_info = {'dataset': 'iCubWorld1.0', 'num_classes': len(class_names)}
    figure_dir = os.path.join("figures", dataset_info["dataset"])
    results_dir = os.path.join("results", dataset_info["dataset"])
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # BoW
    bow_model = BoWImageClassifier(num_clusters=300, feature_detector='sift', classifier_type='svm',
                                   name="BoW_iCubWorld1.0", dataset_name="iCubWorld1.0")
    bow_model.set_dataset_info(dataset_info)
    bow_results, _ = bow_model.cross_validate(
        train_images=images,
        train_labels=labels,
        class_names=class_names,
        n_folds=5
    )
    bow_model.save("models/BoW_iCubWorld1.0.joblib")

    # CNN
    cnn_model = CNNModel((3, 64, 64), len(class_names), 'medium', 'CNN', 'iCubWorld1.0')
    cnn_model.set_dataset_info(dataset_info)
    cnn_results = cnn_model.cross_validate(
        images=resized_images,
        labels=labels,
        class_names=class_names,
        epochs=20,
        batch_size=32
    )
    cnn_model.save("models/CNN_iCubWorld1.0.pth")

    # ResNet
    resnet_model = ResNetModel((3, 64, 64), len(class_names), name='ResNet18', dataset_name='iCubWorld1.0')
    resnet_model.set_dataset_info(dataset_info)
    resnet_results = resnet_model.cross_validate(
        images=resized_images,
        labels=labels,
        class_names=class_names,
        epochs=20,
        batch_size=128
    )
    resnet_model.save("models/ResNet18_iCubWorld1.0.pth")

    # Save comparison plots
    plot_cv_comparison(
        bow_results, cnn_results, resnet_results,
        fig_path=os.path.join(figure_dir, "bow_vs_cnn_vs_resnet.png"),
        csv_path=os.path.join(results_dir, "bow_vs_cnn_vs_resnet.csv")
    )

    compare_training_times(
        bow_results, cnn_results, resnet_results,
        fig_path=os.path.join(figure_dir, "bow_vs_cnn_vs_resnet_times.png"),
        csv_path=os.path.join(results_dir, "bow_vs_cnn_vs_resnet_times.csv")
    )


def run_cifar10_cv_experiment():
    print("\n=== Running CIFAR-10 CV Experiment (BoW vs CNN vs ResNet) ===")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    from dataset_utils import load_cifar10

    images, labels, class_names = load_cifar10(dataset_path='datasets/cifar-10-batches-py')
    images, labels = create_balanced_subset(images, labels, n_per_class=1000)
    resized_images = np.array([cv2.resize(img, (64, 64)) for img in images])

    dataset_info = {'dataset': 'CIFAR10', 'num_classes': len(class_names)}
    figure_dir = os.path.join("figures", dataset_info["dataset"])
    results_dir = os.path.join("results", dataset_info["dataset"])
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # BoW
    bow_model = BoWImageClassifier(num_clusters=300, feature_detector='sift', classifier_type='svm',
                                   name="BoW_CIFAR10", dataset_name="CIFAR10")
    bow_model.set_dataset_info(dataset_info)
    bow_results, _ = bow_model.cross_validate(
        train_images=images,
        train_labels=labels,
        class_names=class_names,
        n_folds=5
    )
    bow_model.save("models/BoW_CIFAR10.joblib")

    # CNN
    cnn_model = CNNModel((3, 64, 64), len(class_names), 'medium', 'CNN', 'CIFAR10')
    cnn_model.set_dataset_info(dataset_info)
    cnn_results = cnn_model.cross_validate(
        images=resized_images,
        labels=labels,
        class_names=class_names,
        epochs=20,
        batch_size=32
    )
    cnn_model.save("models/CNN_CIFAR10.pth")

    # ResNet
    resnet_model = ResNetModel((3, 64, 64), len(class_names), name='ResNet18', dataset_name='CIFAR10')
    resnet_model.set_dataset_info(dataset_info)
    resnet_results = resnet_model.cross_validate(
        images=resized_images,
        labels=labels,
        class_names=class_names,
        epochs=20,
        batch_size=128
    )
    resnet_model.save("models/ResNet18_CIFAR10.pth")

    # 输出图表和数据
    plot_cv_comparison(
        bow_results, cnn_results, resnet_results,
        fig_path=os.path.join(figure_dir, "bow_vs_cnn_vs_resnet.png"),
        csv_path=os.path.join(results_dir, "bow_vs_cnn_vs_resnet.csv")
    )

    compare_training_times(
        bow_results, cnn_results, resnet_results,
        fig_path=os.path.join(figure_dir, "bow_vs_cnn_vs_resnet_times.png"),
        csv_path=os.path.join(results_dir, "bow_vs_cnn_vs_resnet_times.csv")
    )


if __name__ == "__main__":
    run_icubworld_cv_experiment()
    run_cifar10_cv_experiment()


