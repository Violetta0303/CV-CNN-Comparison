import numpy as np
import seaborn as sns
import time
import os
import csv
from abc import ABC, abstractmethod
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class ModelBase(ABC):
    """
    Abstract base class for all models (traditional CV and deep learning)
    Provides common interface for training, evaluation and cross-validation
    """

    def __init__(self, name):
        self.name = name
        self.is_trained = False
        self.training_time = 0
        self.inference_time = 0
        self.dataset_info = None  # Store dataset information

    @abstractmethod
    def train(self, train_images, train_labels):
        pass

    @abstractmethod
    def predict(self, images):
        pass

    @abstractmethod
    def save(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass

    def set_dataset_info(self, dataset_info):
        self.dataset_info = dataset_info

    def _get_dataset_subdir(self, base_dir):
        """Generate sub-directory path based on dataset name."""
        dataset = self.dataset_info.get('dataset', 'generic') if self.dataset_info else 'generic'
        path = os.path.join(base_dir, dataset)
        os.makedirs(path, exist_ok=True)
        return path

    def evaluate(self, test_images, test_labels, class_names=None, plot_confusion=False):
        start_time = time.time()
        predictions = self.predict(test_images)
        self.inference_time = time.time() - start_time

        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, average='macro', zero_division=0)
        recall = recall_score(test_labels, predictions, average='macro', zero_division=0)
        f1 = f1_score(test_labels, predictions, average='macro', zero_division=0)

        if plot_confusion and class_names is not None:
            out_dir = self._get_dataset_subdir("figures")
            self._plot_confusion_matrix(test_labels, predictions, class_names, out_dir=out_dir)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions,
            'ground_truth': test_labels,
            'inference_time': self.inference_time
        }

    def cross_validate(self, train_images, train_labels, n_folds=5, class_names=None, export_csv=True):
        figure_dir = self._get_dataset_subdir("figures")
        results_dir = self._get_dataset_subdir("results")

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        fold_metrics = {metric: [] for metric in metrics}
        fold_times = {'train': [], 'inference': []}

        all_confusion_matrices = []
        all_predictions = []
        all_ground_truth = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_images), 1):
            print(f"\nFold {fold}/{n_folds}")

            if isinstance(train_images, list):
                X_train = [train_images[i] for i in train_idx]
                X_val = [train_images[i] for i in val_idx]
            else:
                X_train = train_images[train_idx]
                X_val = train_images[val_idx]
            y_train = train_labels[train_idx]
            y_val = train_labels[val_idx]

            start_time = time.time()
            self.train(X_train, y_train)
            train_time = time.time() - start_time
            fold_times['train'].append(train_time)

            # results = self.evaluate(X_val, y_val, class_names=class_names)
            results = self.evaluate(X_val, y_val, class_names=class_names, plot_confusion=True)
            for metric in metrics:
                fold_metrics[metric].append(results[metric])
            fold_times['inference'].append(results['inference_time'])

            all_predictions.extend(results['predictions'])
            all_ground_truth.extend(results['ground_truth'])
            cm = confusion_matrix(y_val, results['predictions'])
            all_confusion_matrices.append(cm)

            print(f"Fold {fold} - Accuracy: {results['accuracy']:.4f}, "
                  f"Precision: {results['precision']:.4f}, "
                  f"Recall: {results['recall']:.4f}, "
                  f"F1: {results['f1']:.4f}, "
                  f"Train time: {train_time:.2f}s")

        cv_results = {f'{m}_mean': np.mean(fold_metrics[m]) for m in metrics}
        cv_results.update({f'{m}_std': np.std(fold_metrics[m]) for m in metrics})
        cv_results['train_time_mean'] = np.mean(fold_times['train'])
        cv_results['inference_time_mean'] = np.mean(fold_times['inference'])

        print("\nCross-Validation Summary:")
        for m in metrics:
            print(f"{m.capitalize()}: {cv_results[f'{m}_mean']:.4f} ± {cv_results[f'{m}_std']:.4f}")
        print(f"Average Training Time: {cv_results['train_time_mean']:.2f}s")
        print(f"Average Inference Time: {cv_results['inference_time_mean']:.4f}s")

        if self.dataset_info:
            print("\nDataset Information:")
            for key, value in self.dataset_info.items():
                print(f"  {key}: {value}")

        self._plot_cv_metrics(fold_metrics, n_folds, figure_dir)
        self._plot_merged_confusion_matrix(all_confusion_matrices, class_names, figure_dir)
        self._plot_confusion_matrix(all_ground_truth, all_predictions, class_names, fold="all", out_dir=figure_dir)

        if export_csv:
            self._export_cv_results_to_csv(fold_metrics, cv_results, results_dir)

        return cv_results, fold_metrics

    def _plot_cv_metrics(self, metric_dict, n_folds, out_dir):
        metrics = list(metric_dict.keys())
        x = np.arange(1, n_folds + 1)

        plt.figure(figsize=(10, 6))
        for m in metrics:
            plt.plot(x, metric_dict[m], marker='o', label=m.capitalize())

        plt.title(f'{self.name} Cross-Validation Metrics per Fold', fontsize=14)
        plt.xlabel('Fold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(x, fontsize=11)
        plt.yticks(fontsize=11)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=11)
        # plt.tight_layout()
        filename = os.path.join(out_dir, f"{os.path.basename(self.name)}_cv_metrics.png")
        plt.savefig(filename, dpi=300)
        print(f"Saved CV metrics: {filename}")
        plt.close()

    def _plot_merged_confusion_matrix(self, confusion_matrices, class_names, out_dir):
        if not confusion_matrices:
            return

        merged_cm = sum(confusion_matrices)
        plt.figure(figsize=(10, 8))
        sns.heatmap(merged_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Merged Confusion Matrix - {self.name} (All Folds)', fontsize=14)
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("True", fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        # plt.tight_layout()
        filename = os.path.join(out_dir, f"{os.path.basename(self.name)}_confusion_foldall.png")
        plt.savefig(filename, dpi=300)
        print(f"Saved merged confusion matrix: {filename}")
        plt.close()

    def _plot_confusion_matrix(self, y_true, y_pred, class_names, fold=None, out_dir="figures"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        title = f'Confusion Matrix - {self.name}'
        if fold is not None:
            title += f' (Fold {fold})'
        plt.title(title, fontsize=14)
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("True", fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        # plt.tight_layout()
        filename = os.path.join(out_dir, f"{os.path.basename(self.name)}_confusion_fold{fold if fold else 'all'}.png")
        plt.savefig(filename, dpi=300)
        print(f"Saved confusion matrix: {filename}")
        plt.close()

    def _compute_metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return acc, prec, rec, f1

    def _plot_multiclass_roc(self, y_true, y_score, class_names, out_dir):
        n_classes = len(class_names)
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i],
                     label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'Multiclass ROC Curve - {self.name}', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)

        fig_path = os.path.join(out_dir, f"{self.name}_roc_curve.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Saved ROC curve: {fig_path}")

    def _export_cv_results_to_csv(self, fold_metrics, summary_metrics, out_dir):
        os.makedirs(out_dir, exist_ok=True)

        fold_csv_path = os.path.join(out_dir, f"{os.path.basename(self.name)}_cv_fold_metrics.csv")
        with open(fold_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['fold'] + list(fold_metrics.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for fold in range(len(fold_metrics['accuracy'])):
                row = {'fold': fold + 1}
                for metric in fold_metrics:
                    row[metric] = fold_metrics[metric][fold]
                writer.writerow(row)

        summary_csv_path = os.path.join(out_dir, f"{os.path.basename(self.name)}_cv_summary.csv")
        with open(summary_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['metric', 'mean', 'std'])
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                writer.writerow([
                    metric,
                    summary_metrics[f'{metric}_mean'],
                    summary_metrics[f'{metric}_std']
                ])
            writer.writerow(['train_time', summary_metrics['train_time_mean'], ''])
            writer.writerow(['inference_time', summary_metrics['inference_time_mean'], ''])

            if self.dataset_info:
                writer.writerow(['', '', ''])
                writer.writerow(['dataset_info', '', ''])
                for key, value in self.dataset_info.items():
                    writer.writerow([key, value, ''])

        print(f"Results exported to CSV: {fold_csv_path} and {summary_csv_path}")
