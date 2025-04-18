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

# -----------------------------------------------------------------------------
# Utility functions for visual comparison of BoW and CNN results
# -----------------------------------------------------------------------------

def plot_cv_comparison(bow_results, cnn_results, fig_path, csv_path):
    """Plot accuracy / precision / recall / F1 for BoW vs. CNN."""
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    bow_means = [float(bow_results.get(f'{m}_mean', 0)) for m in metrics]
    bow_stds = [float(bow_results.get(f'{m}_std', 0)) for m in metrics]
    cnn_means = [float(cnn_results.get(f'{m}_mean', 0)) for m in metrics]
    cnn_stds = [float(cnn_results.get(f'{m}_std', 0)) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(12, 8))
    plt.bar(x - width / 2, bow_means, width, label='BoW', yerr=bow_stds,
            capsize=5, colour='royalblue')
    plt.bar(x + width / 2, cnn_means, width, label='CNN', yerr=cnn_stds,
            capsize=5, colour='darkorange')

    plt.ylabel("Score", fontsize=14)
    plt.title("BoW vs CNN - Cross‑Validation Performance", fontsize=16)
    plt.xticks(x, [m.capitalize() for m in metrics], fontsize=12)
    plt.ylim(0, 1)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Annotate bar tops for quick reading
    for i in range(len(metrics)):
        plt.text(x[i] - width / 2, bow_means[i] + bow_stds[i] + 0.02,
                 f"{bow_means[i]:.3f}", ha='center', fontsize=10)
        plt.text(x[i] + width / 2, cnn_means[i] + cnn_stds[i] + 0.02,
                 f"{cnn_means[i]:.3f}", ha='center', fontsize=10)

    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {fig_path}")

    # Export CSV for later analysis
    with open(csv_path, 'w', newline='') as f:
        f.write('Metric,BoW_Mean,BoW_StdDev,CNN_Mean,CNN_StdDev\n')
        for i, m in enumerate(metrics):
            f.write(f"{m},{bow_means[i]:.4f},{bow_stds[i]:.4f},{cnn_means[i]:.4f},{cnn_stds[i]:.4f}\n")
    print(f"Saved comparison CSV: {csv_path}")


def compare_training_times(bow_results, cnn_results, fig_path, csv_path):
    """Plot training / inference time comparison on a log‑scale."""
    categories = ['Training Time', 'Inference Time']
    bow_times = [float(bow_results.get('train_time_mean', 1.0)),
                 float(bow_results.get('inference_time_mean', 0.1))]
    cnn_times = [float(cnn_results.get('train_time_mean', 1.0)),
                 float(cnn_results.get('inference_time_mean', 0.1))]

    # Values must be > 0 for log‑scale
    bow_times = [max(t, 1e-3) for t in bow_times]
    cnn_times = [max(t, 1e-3) for t in cnn_times]

    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, bow_times, width, label='BoW', colour='royalblue')
    plt.bar(x + width / 2, cnn_times, width, label='CNN', colour='darkorange')

    plt.ylabel("Time (seconds)", fontsize=14)
    plt.title("BoW vs CNN - Training and Inference Times", fontsize=16)
    plt.xticks(x, categories, fontsize=12)
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Annotate bars
    for i in range(len(categories)):
        plt.text(x[i] - width / 2, bow_times[i] * 1.1, f"{bow_times[i]:.2f}s",
                 ha='center', fontsize=10)
        plt.text(x[i] + width / 2, cnn_times[i] * 1.1, f"{cnn_times[i]:.2f}s",
                 ha='center', fontsize=10)

    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved timing comparison plot: {fig_path}")

    # Also emit CSV
    with open(csv_path, 'w', newline='') as f:
        f.write('Metric,BoW,CNN\n')
        for i, cat in enumerate(categories):
            f.write(f"{cat},{bow_times[i]:.4f},{cnn_times[i]:.4f}\n")
    print(f"Saved timing CSV: {csv_path}")

# -----------------------------------------------------------------------------
# Single‑dataset experiment helpers (quick runs rather than big grid search)
# -----------------------------------------------------------------------------

def run_icubworld_1_experiment(bow_classifier='svm'):
    """Run a quick BoW vs CNN experiment on iCubWorld 1.0."""

    print("\n=== iCubWorld1.0 Experiment ===")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create standard directory tree if absent
    for d in ["figures", "results", "models"]:
        os.makedirs(d, exist_ok=True)

    dataset_root = download_icubworld(dest_folder='./datasets', version='1.0')
    train_human_path = os.path.join(dataset_root, 'human', 'train')
    train_robot_path = os.path.join(dataset_root, 'robot', 'train')

    # Load and merge human + robot streams
    human_images, human_labels, class_names = load_icub_world(train_human_path,
                                                              version='1.0')
    robot_images, robot_labels, _ = load_icub_world(train_robot_path,
                                                    version='1.0')

    combined_images = np.concatenate((human_images, robot_images), axis=0)
    combined_labels = np.concatenate((human_labels, robot_labels), axis=0)

    images, labels = create_balanced_subset(combined_images, combined_labels,
                                           n_per_class=400)

    # Resize copy for CNN backbone
    img_size = (64, 64)
    resized_images = np.array([cv2.resize(img, img_size) for img in images])

    dataset_info = {'dataset': 'iCubWorld1.0', 'num_classes': len(class_names)}

    # ----------------- Bag‑of‑Words branch -----------------
    bow_model = BoWImageClassifier(num_clusters=300,
                                   feature_detector='sift',
                                   classifier_type=bow_classifier,
                                   name=f'BoW_iCubWorld1.0_{bow_classifier}',
                                   dataset_name='iCubWorld1.0')
    bow_model.set_dataset_info(dataset_info)
    bow_cv_results, _ = bow_model.cross_validate(train_images=images,
                                                 train_labels=labels,
                                                 class_names=class_names,
                                                 n_folds=5)
    bow_model.save(f"models/BoW_iCubWorld1.0_{bow_classifier}.joblib")

    # ----------------- CNN branch -----------------
    cnn_model = CNNModel((3, *img_size), len(class_names), 'medium', 'CNN',
                         'iCubWorld1.0')
    cnn_model.set_dataset_info(dataset_info)
    cnn_cv_results = cnn_model.cross_validate(images=resized_images,
                                              labels=labels,
                                              class_names=class_names,
                                              epochs=20,
                                              batch_size=32)
    cnn_model.save("models/CNN_iCubWorld1.0.pth")

    # Create dataset‑specific output folders
    fig_dir = os.path.join('figures', dataset_info['dataset'])
    res_dir = os.path.join('results', dataset_info['dataset'])
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    plot_cv_comparison(bow_cv_results, cnn_cv_results,
                       fig_path=os.path.join(fig_dir, 'bow_vs_cnn.png'),
                       csv_path=os.path.join(res_dir, 'bow_vs_cnn.csv'))

    compare_training_times(bow_cv_results, cnn_cv_results,
                           fig_path=os.path.join(fig_dir, 'bow_vs_cnn_times.png'),
                           csv_path=os.path.join(res_dir, 'bow_vs_cnn_times.csv'))


def run_cifar10_experiment(bow_classifier='svm'):
    """Run a quick BoW vs CNN experiment on CIFAR‑10."""

    print("\n=== CIFAR‑10 Experiment ===")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for d in ["figures", "results", "models"]:
        os.makedirs(d, exist_ok=True)

    images, labels, class_names = load_cifar10(dataset_path='datasets/cifar-10-batches-py')
    images, labels = create_balanced_subset(images, labels, n_per_class=1000)

    img_size = (64, 64)
    resized_images = np.array([cv2.resize(img, img_size) for img in images])

    dataset_info = {'dataset': 'CIFAR10', 'num_classes': len(class_names)}

    # BoW branch
    bow_model = BoWImageClassifier(num_clusters=300,
                                   feature_detector='sift',
                                   classifier_type=bow_classifier,
                                   name=f'BoW_CIFAR10_{bow_classifier}',
                                   dataset_name='CIFAR10')
    bow_model.set_dataset_info(dataset_info)
    bow_cv_results, _ = bow_model.cross_validate(train_images=images,
                                                 train_labels=labels,
                                                 class_names=class_names,
                                                 n_folds=5)
    bow_model.save(f"models/BoW_CIFAR10_{bow_classifier}.joblib")

    # CNN branch
    cnn_model = CNNModel((3, *img_size), len(class_names), 'medium', 'CNN',
                         'CIFAR10')
    cnn_model.set_dataset_info(dataset_info)
    cnn_cv_results = cnn_model.cross_validate(images=resized_images,
                                              labels=labels,
                                              class_names=class_names,
                                              epochs=20,
                                              batch_size=32)
    cnn_model.save("models/CNN_CIFAR10.pth")

    fig_dir = os.path.join('figures', dataset_info['dataset'])
    res_dir = os.path.join('results', dataset_info['dataset'])
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    plot_cv_comparison(bow_cv_results, cnn_cv_results,
                       fig_path=os.path.join(fig_dir, 'bow_vs_cnn.png'),
                       csv_path=os.path.join(res_dir, 'bow_vs_cnn.csv'))

    compare_training_times(bow_cv_results, cnn_cv_results,
                           fig_path=os.path.join(fig_dir, 'bow_vs_cnn_times.png'),
                           csv_path=os.path.join(res_dir, 'bow_vs_cnn_times.csv'))

# -----------------------------------------------------------------------------
# Comprehensive grid search (multiple detectors / classifiers)
# -----------------------------------------------------------------------------

def run_grid_search():
    """Run a grid search across detectors, cluster counts and classifier types."""
    print("\n=== Running Grid Search for All Datasets ===")

    # Baseline BoW parameter combinations (detector + cluster count)
    bow_param_grid = [
        {"num_clusters": 300, "feature_detector": "sift"},
        # {"num_clusters": 300, "feature_detector": "orb"},
        # {"num_clusters": 300, "feature_detector": "harris+brief"},
    ]

    classifier_types = ['svm', 'knn', 'decision_tree', 'random_forest']

    # ------------------------------------------------------------------
    # Prepare both datasets (iCubWorld dataset + CIFAR dataset)
    # ------------------------------------------------------------------
    dataset_root = download_icubworld(dest_folder='./datasets', version='1.0')
    human_images, human_labels, class_names_icub = load_icub_world(
        os.path.join(dataset_root, 'human', 'train'), version='1.0')
    robot_images, robot_labels, _ = load_icub_world(
        os.path.join(dataset_root, 'robot', 'train'), version='1.0')

    icub_images = np.concatenate((human_images, robot_images), axis=0)
    icub_labels = np.concatenate((human_labels, robot_labels), axis=0)
    icub_images, icub_labels = create_balanced_subset(icub_images, icub_labels,
                                                     n_per_class=400)
    icub_images = icub_images.astype(np.uint8)
    icub_resized = np.array([cv2.resize(img, (64, 64)) for img in icub_images])

    cifar_images, cifar_labels, class_names_cifar = load_cifar10(
        dataset_path='datasets/cifar-10-batches-py')
    cifar_images, cifar_labels = create_balanced_subset(cifar_images, cifar_labels,
                                                        n_per_class=1000)
    cifar_images = cifar_images.astype(np.uint8)
    cifar_resized = np.array([cv2.resize(img, (64, 64)) for img in cifar_images])

    datasets = {
        'iCubGS': (icub_images, icub_labels, icub_resized, class_names_icub),
        'CIFARGS': (cifar_images, cifar_labels, cifar_resized, class_names_cifar)
    }

    best_results = []

    for dataset_name, (images, labels, resized_images, class_names) in datasets.items():
        print(f"\n--- Grid Search on {dataset_name} ---")
        ds_info = {'dataset': dataset_name, 'num_classes': len(class_names)}

        best_bow = {"score": 0}
        param_iter = [(p, clf) for p in bow_param_grid for clf in classifier_types]

        for idx, (bow_params, clf_name) in enumerate(
                tqdm(param_iter, desc=f"BoW [{dataset_name}]")):
            model_name = f"BoW_{dataset_name}_{idx + 1}_{clf_name}"
            model = BoWImageClassifier(num_clusters=bow_params['num_clusters'],
                                       feature_detector=bow_params['feature_detector'],
                                       classifier_type=clf_name,
                                       name=model_name,
                                       dataset_name=dataset_name)
            model.set_dataset_info(ds_info)
            try:
                bow_results, _ = model.cross_validate(train_images=images,
                                                       train_labels=labels,
                                                       class_names=class_names,
                                                       n_folds=5)
                model.save(f"models/{model_name}.joblib")
                if bow_results['f1_mean'] > best_bow['score']:
                    best_bow = {
                        'score': bow_results['f1_mean'],
                        'name': model_name,
                        'params': {**bow_params, 'classifier_type': clf_name}
                    }
            except ValueError as err:
                # Gracefully skip configurations that fail (e.g., too few descriptors)
                print(f"[Warning] Skipping {model_name}: {err}")

        best_results.append((dataset_name, best_bow))

    # ------------------------------------------------------------------
    # Summarise optimum configs per dataset
    # ------------------------------------------------------------------
    print("\n=== Best Configurations Summary ===")
    for dataset_name, best_bow in best_results:
        print(f"\nDataset: {dataset_name}")
        print(f"  Best BoW: {best_bow['name']} with F1 = {best_bow['score']:.4f}")
        print(f"  Params : {best_bow['params']}")

# -----------------------------------------------------------------------------
# Entry‑point guard
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Uncomment one of the quick experiments, or run the full grid search.

    # run_icubworld_1_experiment(bow_classifier='random_forest')
    # run_cifar10_experiment(bow_classifier='random_forest')
    run_grid_search()