import csv
import os
import random

import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from dataset_utils import load_icub_world, load_cifar10, create_balanced_subset
from resnet_model import ResNetModel
from download_icubworld import download_icubworld

def run_resnet_icub_cv_experiment():
    print("\n=== ResNet CV Experiment ===")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_root = download_icubworld(dest_folder='./datasets', version='1.0')
    train_human_path = os.path.join(dataset_root, 'human', 'train')
    train_robot_path = os.path.join(dataset_root, 'robot', 'train')

    human_images, human_labels, class_names = load_icub_world(train_human_path, version='1.0')
    robot_images, robot_labels, _ = load_icub_world(train_robot_path, version='1.0')

    combined_images = np.concatenate((human_images, robot_images), axis=0)
    combined_labels = np.concatenate((human_labels, robot_labels), axis=0)

    images, labels = create_balanced_subset(combined_images, combined_labels, n_per_class=400)
    img_size = (64, 64)
    resized_images = np.array([cv2.resize(img, img_size) for img in images])

    dataset_info = {'dataset': 'iCubWorld1.0', 'num_classes': len(class_names)}

    resnet_model = ResNetModel((3, img_size[0], img_size[1]), len(class_names), name="ResNet18", dataset_name="iCubWorld1.0")
    resnet_model.set_dataset_info(dataset_info)
    resnet_cv_results = resnet_model.cross_validate(
        images=resized_images,
        labels=labels,
        class_names=class_names,
        epochs=20,
        batch_size=128
    )
    resnet_model.save("models/ResNet18_iCubWorld1.0.pth")

    print(f"Finished ResNet CV on iCubWorld1.0, mean F1 = {resnet_cv_results['f1_mean']:.4f}")


def run_resnet_cv_experiment_cifar():
    print("\n=== ResNet CV Experiment on CIFAR-10 ===")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    images, labels, class_names = load_cifar10(dataset_path='datasets/cifar-10-batches-py')
    images, labels = create_balanced_subset(images, labels, n_per_class=1000)
    resized_images = np.array([cv2.resize(img, (64, 64)) for img in images])

    dataset_info = {'dataset': 'CIFAR10', 'num_classes': len(class_names)}

    resnet_model = ResNetModel((3, 64, 64), len(class_names), name="ResNet18", dataset_name="CIFAR10")
    resnet_model.set_dataset_info(dataset_info)
    resnet_cv_results = resnet_model.cross_validate(
        images=resized_images,
        labels=labels,
        class_names=class_names,
        epochs=20,
        batch_size=128
    )
    resnet_model.save("models/ResNet18_CIFAR10.pth")

    print(f"Finished ResNet CV on CIFAR-10, mean F1 = {resnet_cv_results['f1_mean']:.4f}")


def run_resnet_grid_search():
    print("\n=== Stage-wise ResNet18 Grid Search ===")

    # Step 1: Fix batch_size=32 and lr=0.001 and search only kernel_size
    kernel_sizes = [3, 5, 7, 9]

    # Step 2: Fix kernel_size to the best value and search for a combination of these two parameters
    search_params = {
        "batch_sizes": [16, 32, 64, 128],
        "learning_rates": [0.001, 0.0005]
    }

    # Load the iCub and CIFAR data
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

    for ds_name, (images, labels, class_names) in datasets.items():
        print(f"\n--- ResNet Grid Search on {ds_name} ---")
        ds_info = {'dataset': ds_name, 'num_classes': len(class_names)}

        # === Phase 1: Search for kernel_size
        best_kernel = 3
        best_f1_kernel = 0

        for k in kernel_sizes:
            model_name = f"ResNet18_{ds_name}_k{k}"
            model = ResNetModel((3, 64, 64), len(class_names), name=model_name, dataset_name=ds_name)
            model.set_dataset_info(ds_info)
            model.model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=k, stride=2, padding=k // 2, bias=False)
            model.optimizer = torch.optim.Adam(model.model.parameters(), lr=0.001)

            results = model.cross_validate(
                images=images,
                labels=labels,
                class_names=class_names,
                epochs=20,
                batch_size=32
            )
            model.save(f"models/{model_name}.pth")

            if results['f1_mean'] > best_f1_kernel:
                best_f1_kernel = results['f1_mean']
                best_kernel = k

        print(f"[Stage 1] Best kernel size for {ds_name}: {best_kernel} (F1 = {best_f1_kernel:.4f})")

        # === Phase 2: Search for batch_size and lr at the best kernel_size
        best_combo = {"f1": 0, "params": {}}
        for b in search_params['batch_sizes']:
            for lr in search_params['learning_rates']:
                model_name = f"ResNet18_{ds_name}_k{best_kernel}_b{b}_lr{lr}"
                model = ResNetModel((3, 64, 64), len(class_names), name=model_name, dataset_name=ds_name)
                model.set_dataset_info(ds_info)
                model.model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=best_kernel, stride=2, padding=best_kernel // 2, bias=False)
                model.optimizer = torch.optim.Adam(model.model.parameters(), lr=lr)

                results = model.cross_validate(
                    images=images,
                    labels=labels,
                    class_names=class_names,
                    epochs=20,
                    batch_size=b
                )
                model.save(f"models/{model_name}.pth")

                if results['f1_mean'] > best_combo['f1']:
                    best_combo = {"f1": results['f1_mean'], "params": {"batch_size": b, "lr": lr}}

        print(f"[Stage 2] Best final config for {ds_name}: kernel={best_kernel}, "
              f"batch_size={best_combo['params']['batch_size']}, lr={best_combo['params']['lr']} "
              f"(F1 = {best_combo['f1']:.4f})")

def run_resnet_per_class_size_search():
    print("\n=== ResNet Cross-Validation with Varying Samples per Class ===")
    for dataset_name, loader_func, default_path, class_count, npc_list in [
        ("iCubWorld1.0", load_icub_world, "datasets/iCubWorld1.0", 7, [50, 100, 200, 300, 400]),
        ("CIFAR10", load_cifar10, "datasets/cifar-10-batches-py", 10, [200, 400, 600, 800, 1000])
    ]:
        print(f"\n--- Dataset: {dataset_name} ---")

        results_dir = os.path.join("results", dataset_name)
        figure_dir = os.path.join("figures", dataset_name)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(figure_dir, exist_ok=True)

        csv_path = os.path.join(results_dir, "resnet_npc.csv")
        fig_path = os.path.join(figure_dir, "resnet_npc.png")

        all_metrics = {"npc": [], "accuracy": [], "precision": [], "recall": [], "f1": []}

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["num_per_class", "accuracy", "precision", "recall", "f1"])

            for n_per_class in npc_list:
                print(f"\n>>> Running CV with {n_per_class} samples per class")

                if dataset_name == "iCubWorld1.0":
                    human_path = os.path.join(default_path, 'human', 'train')
                    robot_path = os.path.join(default_path, 'robot', 'train')
                    human_images, human_labels, class_names = loader_func(human_path, version='1.0')
                    robot_images, robot_labels, _ = loader_func(robot_path, version='1.0')
                    images = np.concatenate((human_images, robot_images), axis=0)
                    labels = np.concatenate((human_labels, robot_labels), axis=0)
                else:
                    images, labels, class_names = loader_func(dataset_path=default_path)

                images, labels = create_balanced_subset(images, labels, n_per_class=n_per_class)
                resized_images = np.array([cv2.resize(img, (64, 64)) for img in images])
                dataset_info = {'dataset': dataset_name, 'num_classes': len(class_names)}

                model = ResNetModel((3, 64, 64), len(class_names),
                                    name=f"ResNet18_{dataset_name}_n{n_per_class}",
                                    dataset_name=dataset_name)
                model.set_dataset_info(dataset_info)
                model.optimizer = torch.optim.Adam(model.model.parameters(), lr=0.001)  # Explicit LR setting
                results = model.cross_validate(
                    images=resized_images,
                    labels=labels,
                    class_names=class_names,
                    epochs=20,
                    batch_size=128
                )
                model.save(f"models/ResNet18_{dataset_name}_n{n_per_class}.pth")

                writer.writerow([
                    n_per_class,
                    results["accuracy_mean"],
                    results["precision_mean"],
                    results["recall_mean"],
                    results["f1_mean"]
                ])

                all_metrics["npc"].append(n_per_class)
                all_metrics["accuracy"].append(results["accuracy_mean"])
                all_metrics["precision"].append(results["precision_mean"])
                all_metrics["recall"].append(results["recall_mean"])
                all_metrics["f1"].append(results["f1_mean"])

                print(f">>> F1 score: {results['f1_mean']:.4f}")

        # Plot and save figure
        plt.figure(figsize=(10, 6))
        for metric in ["accuracy", "precision", "recall", "f1"]:
            plt.plot(all_metrics["npc"], all_metrics[metric], marker='o', label=metric.capitalize())
        plt.title(f"ResNet CV Performance vs Num per Class - {dataset_name}", fontsize=14)
        plt.xlabel("Number of Samples per Class", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

        print(f">>> Figure saved to {fig_path}")

if __name__ == "__main__":
    # run_resnet_icub_cv_experiment()
    # run_resnet_cv_experiment_cifar()
    # run_resnet_grid_search()
    run_resnet_per_class_size_search()