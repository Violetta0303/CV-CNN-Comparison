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
from traditional_cv import BoWImageClassifier
from download_icubworld import download_icubworld


def plot_cv_comparison(bow_results, cnn_results, fig_path, csv_path):
    """
    Plot comparison of BoW and CNN cross-validation results

    Parameters:
        bow_results (dict): BoW cross-validation results with metric_mean and metric_std
        cnn_results (dict): CNN cross-validation results with metric_mean and metric_std
        fig_path (str): Path to save the comparison plot (PNG)
        csv_path (str): Path to save the results (CSV)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    metrics = ['accuracy', 'precision', 'recall', 'f1']

    bow_means = []
    bow_stds = []
    cnn_means = []
    cnn_stds = []

    for m in metrics:
        bow_means.append(float(bow_results.get(f'{m}_mean', 0)))
        bow_stds.append(float(bow_results.get(f'{m}_std', 0)))
        cnn_means.append(float(cnn_results.get(f'{m}_mean', 0)))
        cnn_stds.append(float(cnn_results.get(f'{m}_std', 0)))

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(12, 8))
    plt.bar(x - width / 2, bow_means, width, label='BoW', yerr=bow_stds, capsize=5, color='royalblue')
    plt.bar(x + width / 2, cnn_means, width, label='CNN', yerr=cnn_stds, capsize=5, color='darkorange')

    plt.ylabel("Score", fontsize=14)
    plt.title("BoW vs CNN - Cross-Validation Performance", fontsize=16)
    plt.xticks(x, [m.capitalize() for m in metrics], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for i, v in enumerate(bow_means):
        plt.text(i - width / 2, v + bow_stds[i] + 0.02, f"{v:.3f}", ha='center', fontsize=10)
    for i, v in enumerate(cnn_means):
        plt.text(i + width / 2, v + cnn_stds[i] + 0.02, f"{v:.3f}", ha='center', fontsize=10)

    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {fig_path}")
    plt.close()

    with open(csv_path, 'w', newline='') as f:
        f.write('Metric,BoW_Mean,BoW_StdDev,CNN_Mean,CNN_StdDev\n')
        for i, metric in enumerate(metrics):
            f.write(f"{metric},{bow_means[i]:.4f},{bow_stds[i]:.4f},{cnn_means[i]:.4f},{cnn_stds[i]:.4f}\n")
    print(f"Saved comparison CSV: {csv_path}")


def compare_training_times(bow_results, cnn_results, fig_path, csv_path):
    """
    Plot comparison of training and inference times for BoW and CNN.

    Parameters:
        bow_results (dict): Dictionary with BoW training and inference times.
        cnn_results (dict): Dictionary with CNN training and inference times.
        fig_path (str): File path to save the PNG figure.
        csv_path (str): File path to save the CSV data.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    categories = ['Training Time', 'Inference Time']
    bow_times = []
    cnn_times = []

    # Extract BoW times
    bow_times.append(float(bow_results.get('train_time_mean', 1.0)))
    bow_times.append(float(bow_results.get('inference_time_mean', 0.1)))

    # Extract CNN times
    cnn_times.append(float(cnn_results.get('train_time_mean', 1.0)))
    cnn_times.append(float(cnn_results.get('inference_time_mean', 0.1)))

    x = np.arange(len(categories))
    width = 0.35

    # Ensure times are positive for log scale
    bow_times = [max(t, 1e-3) for t in bow_times]
    cnn_times = [max(t, 1e-3) for t in cnn_times]

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, bow_times, width, label='BoW', color='royalblue')
    plt.bar(x + width / 2, cnn_times, width, label='CNN', color='darkorange')

    plt.ylabel("Time (seconds)", fontsize=14)
    plt.title("BoW vs CNN - Training and Inference Times", fontsize=16)
    plt.xticks(x, categories, fontsize=12)
    plt.yticks(fontsize=12)
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add value labels
    for i, v in enumerate(bow_times):
        plt.text(i - width / 2, v * 1.1, f"{v:.2f}s", ha='center', fontsize=10)
    for i, v in enumerate(cnn_times):
        plt.text(i + width / 2, v * 1.1, f"{v:.2f}s", ha='center', fontsize=10)

    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved timing comparison plot: {fig_path}")
    plt.show()
    plt.close()

    # Write CSV file to correct CSV path
    with open(csv_path, 'w', newline='') as f:
        f.write('Metric,BoW,CNN\n')
        for i, category in enumerate(categories):
            f.write(f"{category},{bow_times[i]:.4f},{cnn_times[i]:.4f}\n")
    print(f"Saved timing CSV: {csv_path}")


def run_grid_search():
    """
    Grid‑search CNN performance for different kernel sizes (3, 5, 7, 9)
    on iCubWorld1.0 and CIFAR‑10. BoW is omitted.
    """
    print("\n=== Running CNN Grid Search for All Datasets ===")

    # ------------------- CNN hyper‑parameter grid -------------------
    cnn_param_grid = [
        {"kernel_size": k, "batch_size": 32, "lr": 0.0005}
        for k in (3, 5, 7, 9)
    ]

    # ------------------- Prepare datasets --------------------------
    dataset_root = download_icubworld(dest_folder='./datasets', version='1.0')
    human_imgs, human_lbls, icub_classes = load_icub_world(
        os.path.join(dataset_root, 'human', 'train'), version='1.0')
    robot_imgs, robot_lbls, _ = load_icub_world(
        os.path.join(dataset_root, 'robot', 'train'), version='1.0')

    icub_imgs = np.concatenate((human_imgs, robot_imgs), axis=0)
    icub_lbls = np.concatenate((human_lbls, robot_lbls), axis=0)
    icub_imgs, icub_lbls = create_balanced_subset(icub_imgs, icub_lbls, n_per_class=400)
    icub_resized = np.array([cv2.resize(img, (64, 64)) for img in icub_imgs], dtype=np.uint8)

    cifar_imgs, cifar_lbls, cifar_classes = load_cifar10(dataset_path='datasets/cifar-10-batches-py')
    cifar_imgs, cifar_lbls = create_balanced_subset(cifar_imgs, cifar_lbls, n_per_class=1000)
    cifar_resized = np.array([cv2.resize(img, (64, 64)) for img in cifar_imgs], dtype=np.uint8)

    datasets = {
        "iCubGS":  (icub_resized,  icub_lbls,  icub_classes),
        "CIFARGS": (cifar_resized, cifar_lbls, cifar_classes),
    }

    best_results = []

    # ------------------- Main loop over datasets -------------------
    for ds_name, (imgs, lbls, class_names) in datasets.items():
        print(f"\n--- CNN Grid Search on {ds_name} ---")
        ds_info = {'dataset': ds_name, 'num_classes': len(class_names)}

        best_cnn = {"score": 0, "name": "N/A", "params": {}}

        # Loop over kernel sizes
        for cnn_p in tqdm(cnn_param_grid, desc=f"CNN [{ds_name}]"):
            cnn_name = f"CNN_{ds_name}_k{cnn_p['kernel_size']}"
            cnn = CNNModel(
                (3, 64, 64), len(class_names),
                'medium', cnn_name, ds_name,
                kernel_size=cnn_p['kernel_size']
            )
            cnn.set_dataset_info(ds_info)
            cnn.optimizer = torch.optim.Adam(cnn.model.parameters(), lr=cnn_p['lr'])

            # k‑fold cross‑validation
            cnn_res = cnn.cross_validate(
                images=imgs,
                labels=lbls,
                class_names=class_names,
                epochs=20,
                batch_size=cnn_p['batch_size']
            )
            cnn.save(f"models/{cnn_name}.pth")

            # Track best kernel size
            if cnn_res['f1_mean'] > best_cnn['score']:
                best_cnn = {"score": cnn_res['f1_mean'], "name": cnn_name, "params": cnn_p}

        best_results.append((ds_name, best_cnn))

    # ------------------- Summary -----------------------------------
    print("\n=== Best Kernel Size per Dataset ===")
    for ds_name, best_cnn in best_results:
        print(f"\nDataset: {ds_name}")
        print(f"  Best CNN: {best_cnn['name']} | F1 = {best_cnn['score']:.4f} | {best_cnn['params']}")


if __name__ == "__main__":
    run_grid_search()