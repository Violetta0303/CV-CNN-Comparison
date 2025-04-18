import csv
import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from model_base import ModelBase
from dataset_utils import preprocess_for_cnn

# Select device (CPU / GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images).float()
        if self.images.ndim == 4 and self.images.shape[1] != 3:
            self.images = self.images.permute(0, 3, 1, 2)  # NHWC → NCHW
        self.images = self.images / 255.0
        self.labels = torch.tensor(labels).long()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class CNNModel(ModelBase):
    # def __init__(self, input_shape, num_classes, model_type='medium', name='CNN', dataset_name=''):
    def __init__(self, input_shape, num_classes, model_type='medium', name='CNN', dataset_name='', kernel_size=3):
        super().__init__(f"{name}_{dataset_name}")
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type
        # Set Kernel Size
        self.kernel_size = kernel_size
        self.model = self._build_model().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)

    def _build_model(self):
        c, h, w = self.input_shape  # channels, height, width
        k = self.kernel_size  # kernel size to test (3, 5, 7, 9)
        p = k // 2  # same‑padding keeps spatial dims

        # model = nn.Sequential(
        #     nn.Conv2d(c, 32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Flatten(),
        #     nn.Linear(64 * (h // 4) * (w // 4), 64),
        #     nn.ReLU(),
        #     nn.Linear(64, self.num_classes)

        model = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=k, padding=p),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=k, padding=p),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (h // 4) * (w // 4), 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes)
        )
        return model

    def preprocess_data(self, images, labels):
        images = preprocess_for_cnn(np.array(images))
        return images, np.array(labels)

    def train(self, train_images, train_labels, validation_data=None, epochs=20, batch_size=32,
              data_augmentation=False, early_stopping=True):
        train_dataset = ImageDataset(train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        start_time = time.time()
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {correct / total:.4f}")

        self.training_time = time.time() - start_time
        self.is_trained = True
        print(f"Training completed in {self.training_time:.2f} seconds")

    def predict(self, images):
        self.model.eval()
        dataset = ImageDataset(images, [0] * len(images))
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        all_preds = []
        start_time = time.time()
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(device)
                outputs = self.model(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
        self.inference_time = time.time() - start_time
        return np.array(all_preds)

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=device))
        self.model.to(device)
        self.model.eval()

    def evaluate(self, test_images, test_labels, class_names=None, plot_confusion=False):
        dataset = ImageDataset(test_images, test_labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                outputs = self.model(imgs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        acc, prec, rec, f1 = self._compute_metrics(all_labels, all_preds)

        if plot_confusion and class_names:
            out_dir = self._get_dataset_subdir("figures")
            self._plot_confusion_matrix(all_labels, all_preds, class_names, out_dir=out_dir)
            self._plot_multiclass_roc(all_labels, np.array(all_probs), class_names, out_dir=out_dir)

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "predictions": all_preds,
            "ground_truth": all_labels,
            "inference_time": self.inference_time
        }


    def _plot_training_history(self, training_history, n_folds):
        """
        Plot training and validation loss/accuracy curves for each fold

        Parameters:
            training_history: List of dictionaries containing training metrics per fold
            n_folds: Number of folds
        """
        # Plot training/validation loss
        plt.figure(figsize=(12, 10))

        # First subplot for losses
        plt.subplot(2, 1, 1)
        epochs = range(1, len(training_history[0]['train_loss']) + 1)

        for fold in range(n_folds):
            plt.plot(epochs, training_history[fold]['train_loss'],
                     f'C{fold}-', label=f'Fold {fold + 1} Training Loss')
            plt.plot(epochs, training_history[fold]['val_loss'],
                     f'C{fold}--', label=f'Fold {fold + 1} Validation Loss')

        plt.title(f'{self.name} Training and Validation Loss per Epoch', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=10)

        # Second subplot for accuracies
        plt.subplot(2, 1, 2)

        for fold in range(n_folds):
            plt.plot(epochs, training_history[fold]['train_acc'],
                     f'C{fold}-', label=f'Fold {fold + 1} Training Accuracy')
            plt.plot(epochs, training_history[fold]['val_acc'],
                     f'C{fold}--', label=f'Fold {fold + 1} Validation Accuracy')

        plt.title(f'{self.name} Training and Validation Accuracy per Epoch', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=10)

        # plt.tight_layout()
        figure_dir = self._get_dataset_subdir("figures")
        filename = os.path.join(figure_dir, f"{self.name}_training_history.png")
        plt.savefig(filename, dpi=300)
        print(f"Saved training history: {filename}")
        plt.close()

        # Also plot average loss and accuracy across folds
        plt.figure(figsize=(12, 10))

        # Calculate averages
        avg_train_loss = np.mean([fold['train_loss'] for fold in training_history], axis=0)
        avg_val_loss = np.mean([fold['val_loss'] for fold in training_history], axis=0)
        avg_train_acc = np.mean([fold['train_acc'] for fold in training_history], axis=0)
        avg_val_acc = np.mean([fold['val_acc'] for fold in training_history], axis=0)

        # Loss subplot
        plt.subplot(2, 1, 1)
        plt.plot(epochs, avg_train_loss, 'b-', linewidth=2, label='Avg. Training Loss')
        plt.plot(epochs, avg_val_loss, 'r--', linewidth=2, label='Avg. Validation Loss')
        plt.title(f'{self.name} Average Loss across {n_folds} Folds', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=12)

        # Accuracy subplot
        plt.subplot(2, 1, 2)
        plt.plot(epochs, avg_train_acc, 'b-', linewidth=2, label='Avg. Training Accuracy')
        plt.plot(epochs, avg_val_acc, 'r--', linewidth=2, label='Avg. Validation Accuracy')
        plt.title(f'{self.name} Average Accuracy across {n_folds} Folds', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=12)

        # plt.tight_layout()
        figure_dir = self._get_dataset_subdir("figures")
        filename = os.path.join(figure_dir, f"{self.name}_avg_training_history.png")

        plt.savefig(filename, dpi=300)
        print(f"Saved average training history: {filename}")
        plt.close()

        # Export training history to CSV
        results_dir = self._get_dataset_subdir("results")
        csv_path = os.path.join(results_dir, f"{self.name}_training_history.csv")

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Fold', 'Epoch', 'Train_Loss', 'Val_Loss', 'Train_Acc', 'Val_Acc'])

            for fold_idx, fold_data in enumerate(training_history):
                for epoch_idx in range(len(epochs)):
                    writer.writerow([
                        fold_idx + 1,
                        epoch_idx + 1,
                        fold_data['train_loss'][epoch_idx],
                        fold_data['val_loss'][epoch_idx],
                        fold_data['train_acc'][epoch_idx],
                        fold_data['val_acc'][epoch_idx]
                    ])

        print(f"Saved training history to CSV: {csv_path}")


    def cross_validate(self, images, labels, class_names, n_folds=5, epochs=20, batch_size=32):
        """
        Cross-validate CNN model while recording and plotting training and validation loss curves
        """
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        accs, precs, recalls, f1s = [], [], [], []
        training_history = []
        train_times = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels), start=1):
            print(f"\n=== Fold {fold}/{n_folds} ===")
            x_train, x_val = images[train_idx], images[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            self.model = self._build_model().to(device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)

            train_dataset = ImageDataset(x_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataset = ImageDataset(x_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            fold_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

            start_time = time.time()
            for epoch in range(epochs):
                self.model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                epoch_loss = running_loss
                epoch_acc = correct / total
                fold_history['train_loss'].append(epoch_loss)
                fold_history['train_acc'].append(epoch_acc)

                self.model.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()

                epoch_val_loss = val_loss
                epoch_val_acc = correct / total
                fold_history['val_loss'].append(epoch_val_loss)
                fold_history['val_acc'].append(epoch_val_acc)

                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, "
                      f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

            training_history.append(fold_history)
            fold_time = time.time() - start_time
            train_times.append(fold_time)
            print(f"Training completed in {fold_time:.2f} seconds")

            results = self.evaluate(x_val, y_val, class_names=class_names, plot_confusion=True)
            accs.append(results["accuracy"])
            precs.append(results["precision"])
            recalls.append(results["recall"])
            f1s.append(results["f1"])

        self._plot_training_history(training_history, n_folds)

        metrics_dict = {
            "accuracy": accs,
            "precision": precs,
            "recall": recalls,
            "f1": f1s
        }
        figure_dir = self._get_dataset_subdir("figures")
        self._plot_cv_metrics(metrics_dict, n_folds, out_dir=figure_dir)

        summary = {
            "accuracy": accs,
            "precision": precs,
            "recall": recalls,
            "f1": f1s
        }

        print("\n=== CNN Cross-Validation Summary ===")
        for m in summary:
            print(f"{m.capitalize()}: {np.mean(summary[m]):.4f} ± {np.std(summary[m]):.4f}")

        os.makedirs("results", exist_ok=True)
        results_dir = self._get_dataset_subdir("results")
        fold_csv_path = os.path.join(results_dir, f"{self.name}_cv_fold_metrics.csv")
        summary_csv_path = os.path.join(results_dir, f"{self.name}_cv_summary.csv")

        with open(fold_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Fold', 'Accuracy', 'Precision', 'Recall', 'F1'])
            for i in range(n_folds):
                writer.writerow([i + 1, accs[i], precs[i], recalls[i], f1s[i]])

        with open(summary_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Mean', 'Std'])
            for m in summary:
                writer.writerow([m, np.mean(summary[m]), np.std(summary[m])])

        print(f"Saved CSV to: {fold_csv_path} and {summary_csv_path}")

        results = {}
        for m in metrics_dict:
            results[m] = metrics_dict[m]
            results[f'{m}_mean'] = np.mean(metrics_dict[m])
            results[f'{m}_std'] = np.std(metrics_dict[m])
        results['train_time_mean'] = np.mean(train_times)
        results['inference_time_mean'] = np.mean([r.get('inference_time', 0) for r in
                                                  [self.evaluate(x_val, y_val) for x_val, y_val in
                                                   [(images[val_idx], labels[val_idx]) for _, val_idx in
                                                    skf.split(images, labels)]]])

        return results