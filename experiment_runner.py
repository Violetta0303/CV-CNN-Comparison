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



def run_icubworld_1_experiment():
    """Run experiment using iCubWorld1.0 dataset."""

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

    # Load and combine human + robot training data
    human_images, human_labels, class_names = load_icub_world(train_human_path, version='1.0')
    robot_images, robot_labels, _ = load_icub_world(train_robot_path, version='1.0')

    combined_images = np.concatenate((human_images, robot_images), axis=0)
    combined_labels = np.concatenate((human_labels, robot_labels), axis=0)

    images, labels = create_balanced_subset(combined_images, combined_labels, n_per_class=400)

    img_size = (64, 64)
    resized_images = np.array([cv2.resize(img, img_size) for img in images])

    dataset_info = {'dataset': 'iCubWorld1.0', 'num_classes': len(class_names)}

    bow_model = BoWImageClassifier(300, 'sift', 'svm', 'BoW_iCubWorld1.0')
    bow_model.set_dataset_info(dataset_info)
    bow_cv_results, _ = bow_model.cross_validate(
        train_images=images,
        train_labels=labels,
        class_names=class_names,
        n_folds=5
    )
    bow_model.save("models/BoW_iCubWorld1.0.joblib")

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

    # Create dataset-specific subdirectories
    figure_dir = os.path.join("figures", dataset_info["dataset"])
    results_dir = os.path.join("results", dataset_info["dataset"])
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Save comparison visualisations and CSVs
    plot_cv_comparison(
        bow_results=bow_cv_results,
        cnn_results=cnn_cv_results,
        fig_path=os.path.join(figure_dir, "bow_vs_cnn.png"),
        csv_path=os.path.join(results_dir, "bow_vs_cnn.csv")
    )

    compare_training_times(
        bow_results=bow_cv_results,
        cnn_results=cnn_cv_results,
        fig_path=os.path.join(figure_dir, "bow_vs_cnn_times.png"),
        csv_path=os.path.join(results_dir, "bow_vs_cnn_times.csv")
    )


def run_cifar10_experiment():
    """Run experiment using CIFAR-10 dataset."""

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

    bow_model = BoWImageClassifier(300, 'sift', 'svm', 'BoW_CIFAR10')
    bow_model.set_dataset_info(dataset_info)
    bow_cv_results, _ = bow_model.cross_validate(
        train_images=images,
        train_labels=labels,
        class_names=class_names,
        n_folds=5
    )
    bow_model.save("models/BoW_CIFAR10.joblib")

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

    # Create subdirectory for dataset-specific figures and results
    figures_dir = os.path.join("figures", dataset_info["dataset"])
    results_dir = os.path.join("results", dataset_info["dataset"])
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Save comparison plots to dataset-specific subdirectories
    plot_cv_comparison(
        bow_results=bow_cv_results,
        cnn_results=cnn_cv_results,
        fig_path=os.path.join(figures_dir, "bow_vs_cnn.png"),
        csv_path=os.path.join(results_dir, "bow_vs_cnn.csv")
    )

    compare_training_times(
        bow_results=bow_cv_results,
        cnn_results=cnn_cv_results,
        fig_path=os.path.join(figures_dir, "bow_vs_cnn_times.png"),
        csv_path=os.path.join(results_dir, "bow_vs_cnn_times.csv")
    )


def run_grid_search():
    """
    Run grid search over CNN and BoW parameter combinations for both iCubWorld1.0 and CIFAR-10.
    """
    print("\n=== Running Grid Search for All Datasets ===")

    # Define BoW and CNN parameter grids
    # bow_param_grid = [
    #     {"num_clusters": 100, "feature_detector": "sift"},
    #     {"num_clusters": 200, "feature_detector": "sift"},
    #     {"num_clusters": 300, "feature_detector": "sift"},
    #     {"num_clusters": 100, "feature_detector": "orb"},
    #     {"num_clusters": 200, "feature_detector": "orb"},
    #     {"num_clusters": 300, "feature_detector": "orb"},
    #     {"num_clusters": 100, "feature_detector": "harris+brief"},
    #     {"num_clusters": 200, "feature_detector": "harris+brief"},
    #     {"num_clusters": 300, "feature_detector": "harris+brief"},
    # ]

    bow_param_grid = [
        {"num_clusters": n, "feature_detector": d}
        for n in [100, 200, 300]
        for d in ["sift", "orb", "harris+brief"]
    ]

    # cnn_param_grid = [
    #     {"batch_size": 16, "lr": 0.001},
    #     {"batch_size": 32, "lr": 0.001},
    #     {"batch_size": 64, "lr": 0.001},
    #     {"batch_size": 128, "lr": 0.001},
    #     {"batch_size": 16, "lr": 0.0005},
    #     {"batch_size": 32, "lr": 0.0005},
    #     {"batch_size": 64, "lr": 0.0005},
    #     {"batch_size": 128, "lr": 0.0005},
    # ]

    cnn_param_grid = [
        {"batch_size": b, "lr": lr}
        for b in [16, 32, 64, 128]
        for lr in [0.001, 0.0005]
    ]

    # Prepare datasets
    dataset_root = download_icubworld(dest_folder='./datasets', version='1.0')
    human_images, human_labels, class_names = load_icub_world(os.path.join(dataset_root, 'human', 'train'),
                                                              version='1.0')
    robot_images, robot_labels, _ = load_icub_world(os.path.join(dataset_root, 'robot', 'train'),
                                                    version='1.0')
    icub_images = np.concatenate((human_images, robot_images), axis=0)
    icub_labels = np.concatenate((human_labels, robot_labels), axis=0)
    icub_images, icub_labels = create_balanced_subset(icub_images, icub_labels, n_per_class=400)
    icub_images = icub_images.astype(np.uint8)
    icub_resized = np.array([cv2.resize(img, (64, 64)) for img in icub_images])

    cifar_images, cifar_labels, cifar_classes = load_cifar10(dataset_path='datasets/cifar-10-batches-py')
    cifar_images, cifar_labels = create_balanced_subset(cifar_images, cifar_labels, n_per_class=1000)
    cifar_images = cifar_images.astype(np.uint8)
    cifar_resized = np.array([cv2.resize(img, (64, 64)) for img in cifar_images])

    best_results = []

    for dataset_name, (images, labels, resized_images, class_names) in {
        "iCubGS": (icub_images, icub_labels, icub_resized, class_names),
        "CIFARGS": (cifar_images, cifar_labels, cifar_resized, cifar_classes),
    }.items():
        print(f"\n--- Grid Search on {dataset_name} ---")
        ds_info = {'dataset': dataset_name, 'num_classes': len(class_names)}

        best_bow = {"score": 0}
        for idx, bow_params in enumerate(tqdm(bow_param_grid, desc=f"BoW [{dataset_name}]")):
            model_name = f"BoW_{dataset_name}_{idx + 1}"
            model = BoWImageClassifier(
                num_clusters=bow_params['num_clusters'],
                feature_detector=bow_params['feature_detector'],
                name=model_name,
                dataset_name=dataset_name
            )
            model.set_dataset_info(ds_info)
            try:
                bow_results, _ = model.cross_validate(
                    train_images=images,
                    train_labels=labels,
                    class_names=class_names,
                    n_folds=5
                )
                model.save(f"models/{model_name}.joblib")
                if bow_results['f1_mean'] > best_bow["score"]:
                    best_bow = {"score": bow_results['f1_mean'], "name": model_name, "params": bow_params}
            except ValueError as e:
                print(f"[Warning] Skipping {model_name} due to error: {e}")

        best_cnn = {"score": 0}
        for idx, cnn_params in enumerate(tqdm(cnn_param_grid, desc=f"CNN [{dataset_name}]")):
            model_name = f"CNN_{dataset_name}_{idx + 1}"
            model = CNNModel((3, 64, 64), len(class_names), 'medium', model_name, dataset_name)
            model.set_dataset_info(ds_info)
            model.optimizer = torch.optim.Adam(model.model.parameters(), lr=cnn_params['lr'])
            cnn_results = model.cross_validate(
                images=resized_images,
                labels=labels,
                class_names=class_names,
                epochs=20,
                batch_size=cnn_params['batch_size']
            )
            model.save(f"models/{model_name}.pth")

            if cnn_results['f1_mean'] > best_cnn["score"]:
                best_cnn = {"score": cnn_results['f1_mean'], "name": model_name, "params": cnn_params}

        # best_results.append((dataset_name, best_cnn, best_bow))
        best_results.append((dataset_name, best_bow))

    print("\n=== Best Configurations Summary ===")

    for dataset_name, best_cnn, best_bow in best_results:
        print(f"\nDataset: {dataset_name}")
        print(f"  Best BoW: {best_bow['name']} with F1 = {best_bow['score']:.4f} | Params = {best_bow['params']}")
        print(f"  Best CNN: {best_cnn['name']} with F1 = {best_cnn['score']:.4f} | Params = {best_cnn['params']}")

    # for dataset_name, best_bow in best_results:
    #     print(f"\nDataset: {dataset_name}")
    #     print(f"  Best BoW: {best_bow['name']} with F1 = {best_bow['score']:.4f} | Params = {best_bow['params']}")


if __name__ == "__main__":
    run_icubworld_1_experiment()
    run_cifar10_experiment()
    # run_grid_search()
