import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from dataset_utils import (
    load_icub_world,
    load_cifar10,
    create_balanced_subset,
    preprocess_for_cnn
)
from cnn_model import CNNModel
from resnet_model import ResNetModel
from download_icubworld import download_icubworld


def plot_cv_comparison(cnn_results, bow_results, fig_path, csv_path):
    import numpy as np
    import matplotlib.pyplot as plt

    metrics = ['accuracy', 'precision', 'recall', 'f1']

    cnn_means = []
    cnn_stds = []
    resnet_means = []
    resnet_stds = []

    for m in metrics:
        cnn_means.append(float(cnn_results.get(f'{m}_mean', 0)))
        cnn_stds.append(float(cnn_results.get(f'{m}_std', 0)))
        resnet_means.append(float(bow_results.get(f'{m}_mean', 0)))
        resnet_stds.append(float(bow_results.get(f'{m}_std', 0)))

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(12, 8))
    plt.bar(x - width / 2, cnn_means, width, label='CNN', yerr=cnn_stds, capsize=5, color='darkorange')
    plt.bar(x + width / 2, resnet_means, width, label='ResNet', yerr=resnet_stds, capsize=5, color='royalblue')

    plt.ylabel("Score", fontsize=14)
    plt.title("CNN vs ResNet - Cross-Validation Performance", fontsize=16)
    plt.xticks(x, [m.capitalize() for m in metrics], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for i, v in enumerate(cnn_means):
        plt.text(i - width / 2, v + cnn_stds[i] + 0.02, f"{v:.3f}", ha='center', fontsize=10)
    for i, v in enumerate(resnet_means):
        plt.text(i + width / 2, v + resnet_stds[i] + 0.02, f"{v:.3f}", ha='center', fontsize=10)

    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {fig_path}")
    plt.close()

    with open(csv_path, 'w', newline='') as f:
        f.write('Metric,CNN_Mean,CNN_StdDev,ResNet_Mean,ResNet_StdDev\n')
        for i, metric in enumerate(metrics):
            f.write(f"{metric},{cnn_means[i]:.4f},{cnn_stds[i]:.4f},{resnet_means[i]:.4f},{resnet_stds[i]:.4f}\n")
    print(f"Saved comparison CSV: {csv_path}")


def compare_training_times(cnn_results, bow_results, fig_path, csv_path):
    import numpy as np
    import matplotlib.pyplot as plt

    categories = ['Training Time', 'Inference Time']
    cnn_times = []
    resnet_times = []

    cnn_times.append(float(cnn_results.get('train_time_mean', 1.0)))
    cnn_times.append(float(cnn_results.get('inference_time_mean', 0.1)))

    resnet_times.append(float(bow_results.get('train_time_mean', 1.0)))
    resnet_times.append(float(bow_results.get('inference_time_mean', 0.1)))

    x = np.arange(len(categories))
    width = 0.35

    resnet_times = [max(t, 1e-3) for t in resnet_times]
    cnn_times = [max(t, 1e-3) for t in cnn_times]

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, cnn_times, width, label='CNN', color='darkorange')
    plt.bar(x + width / 2, resnet_times, width, label='ResNet', color='royalblue')

    plt.ylabel("Time (seconds)", fontsize=14)
    plt.title("CNN vs ResNet - Training and Inference Times", fontsize=16)
    plt.xticks(x, categories, fontsize=12)
    plt.yticks(fontsize=12)
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for i, v in enumerate(cnn_times):
        plt.text(i - width / 2, v * 1.1, f"{v:.2f}s", ha='center', fontsize=10)
    for i, v in enumerate(resnet_times):
        plt.text(i + width / 2, v * 1.1, f"{v:.2f}s", ha='center', fontsize=10)

    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved timing comparison plot: {fig_path}")
    plt.close()

    with open(csv_path, 'w', newline='') as f:
        f.write('Metric,CNN,ResNet\n')
        for i, category in enumerate(categories):
            f.write(f"{category},{cnn_times[i]:.4f},{resnet_times[i]:.4f}\n")
    print(f"Saved timing CSV: {csv_path}")


def run_icubworld_1_experiment():
    print("\n=== iCubWorld1.0 Experiment ===")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    dataset_root = download_icubworld(dest_folder='./datasets', version='1.0')
    train_human_path = os.path.join(dataset_root, 'human', 'train')
    train_robot_path = os.path.join(dataset_root, 'robot', 'train')

    human_images, human_labels, class_names = load_icub_world(train_human_path, version='transformations')
    robot_images, robot_labels, _ = load_icub_world(train_robot_path, version='transformations')

    combined_images = np.concatenate((human_images, robot_images), axis=0)
    combined_labels = np.concatenate((human_labels, robot_labels), axis=0)

    images, labels = create_balanced_subset(combined_images, combined_labels, n_per_class=400)

    img_size = (64, 64)
    resized_images = np.array([cv2.resize(img, img_size) for img in images])

    dataset_info = {'dataset': 'iCubWorld1.0', 'num_classes': len(class_names)}

    cnn_model = CNNModel((3, img_size[0], img_size[1]), len(class_names), 'medium', 'CNN', 'iCubWorld1.0')
    cnn_model.set_dataset_info(dataset_info)
    cnn_cv_results = cnn_model.cross_validate(
        images=resized_images,
        labels=labels,
        class_names=class_names,
        epochs=20,
        batch_size=32
    )
    cnn_model.save("models/CNN_iCubWorld1.0.pth")

    resnet_model = ResNetModel((3, img_size[0], img_size[1]), len(class_names), name='ResNet18', dataset_name='iCubWorld1.0')
    resnet_model.set_dataset_info(dataset_info)
    resnet_cv_results = resnet_model.cross_validate(
        images=resized_images,
        labels=labels,
        class_names=class_names,
        epochs=20,
        batch_size=32
    )
    resnet_model.save("models/ResNet18_iCubWorld1.0.pth")

    figure_dir = os.path.join("figures", dataset_info["dataset"])
    results_dir = os.path.join("results", dataset_info["dataset"])
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    plot_cv_comparison(
        cnn_results=cnn_cv_results,
        bow_results=resnet_cv_results,
        fig_path=os.path.join(figure_dir, "cnn_vs_resnet.png"),
        csv_path=os.path.join(results_dir, "cnn_vs_resnet.csv")
    )

    compare_training_times(
        cnn_results=cnn_cv_results,
        bow_results=resnet_cv_results,
        fig_path=os.path.join(figure_dir, "cnn_vs_resnet_times.png"),
        csv_path=os.path.join(results_dir, "cnn_vs_resnet_times.csv")
    )


def run_cifar10_experiment():
    print("\n=== CIFAR-10 Experiment ===")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    images, labels, class_names = load_cifar10(dataset_path='datasets/cifar-10-batches-py')
    images, labels = create_balanced_subset(images, labels, n_per_class=1000)

    img_size = (64, 64)
    resized_images = np.array([cv2.resize(img, img_size) for img in images])

    dataset_info = {'dataset': 'CIFAR10', 'num_classes': len(class_names)}

    cnn_model = CNNModel((3, img_size[0], img_size[1]), len(class_names), 'medium', 'CNN', 'CIFAR10')
    cnn_model.set_dataset_info(dataset_info)
    cnn_cv_results = cnn_model.cross_validate(
        images=resized_images,
        labels=labels,
        class_names=class_names,
        epochs=20,
        batch_size=32
    )
    cnn_model.save("models/CNN_CIFAR10.pth")

    resnet_model = ResNetModel((3, img_size[0], img_size[1]), len(class_names), name='ResNet18', dataset_name='CIFAR10')
    resnet_model.set_dataset_info(dataset_info)
    resnet_cv_results = resnet_model.cross_validate(
        images=resized_images,
        labels=labels,
        class_names=class_names,
        epochs=20,
        batch_size=32
    )
    resnet_model.save("models/ResNet18_CIFAR10.pth")

    figures_dir = os.path.join("figures", dataset_info["dataset"])
    results_dir = os.path.join("results", dataset_info["dataset"])
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    plot_cv_comparison(
        cnn_results=cnn_cv_results,
        bow_results=resnet_cv_results,
        fig_path=os.path.join(figures_dir, "cnn_vs_resnet.png"),
        csv_path=os.path.join(results_dir, "cnn_vs_resnet.csv")
    )

    compare_training_times(
        cnn_results=cnn_cv_results,
        bow_results=resnet_cv_results,
        fig_path=os.path.join(figures_dir, "cnn_vs_resnet_times.png"),
        csv_path=os.path.join(results_dir, "cnn_vs_resnet_times.csv")
    )


if __name__ == "__main__":
    run_icubworld_1_experiment()
    run_cifar10_experiment()
