import cv2
import numpy as np
import os
import time
import csv
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import joblib
from model_base import ModelBase


# def extract_keypoints_descriptors(images, detector='sift'):
#     if detector == 'sift':
#         extractor = cv2.SIFT_create()
#     elif detector == 'orb':
#         extractor = cv2.ORB_create()
#     elif detector == 'brisk':
#         extractor = cv2.BRISK_create()
#     else:
#         raise ValueError(f"Unsupported detector: {detector}")
#
#     all_descriptors = []
#     image_descriptors = []
#
#     print(f"Extracting features using {detector}...")
#
#     for i, img in enumerate(images):
#         keypoints, descriptors = extractor.detectAndCompute(img, None)
#         if descriptors is not None:
#             descriptors = descriptors.astype(np.float32)
#             all_descriptors.extend(descriptors)
#             image_descriptors.append(descriptors)
#         else:
#             # If no features detected, add zeros to avoid crashes
#             image_descriptors.append(np.zeros((1, extractor.descriptorSize()), dtype=np.float32))
#
#         # Progress update
#         if (i + 1) % 100 == 0 or i + 1 == len(images):
#             print(f"Processed {i + 1}/{len(images)} images")
#
#     return np.array(all_descriptors), image_descriptors


def extract_keypoints_descriptors(images, detector='sift'):
    all_descriptors = []
    image_descriptors = []

    print(f"Extracting features using {detector}...")

    # Handle detector creation
    if detector == 'sift':
        extractor = cv2.SIFT_create()

    elif detector == 'orb':
        extractor = cv2.ORB_create()

    # elif detector == 'brisk':
    #     extractor = cv2.BRISK_create()

    elif detector == 'harris+brief':
        try:
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        except AttributeError:
            raise ImportError("OpenCV contrib modules required for BRIEF. Install with: pip install opencv-contrib-python")
    else:
        raise ValueError(f"Unsupported detector: {detector}")

    for i, img in enumerate(images):
        if detector == 'harris+brief':
            # Harris detection + BRIEF
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(
                gray, maxCorners=500, qualityLevel=0.01, minDistance=5, useHarrisDetector=True
            )

            if corners is not None:
                keypoints = [cv2.KeyPoint(float(x), float(y), 3) for [[x, y]] in corners]
                keypoints, descriptors = brief.compute(gray, keypoints)
            else:
                descriptors = None
        else:
            keypoints, descriptors = extractor.detectAndCompute(img, None)

        if descriptors is not None:
            descriptors = descriptors.astype(np.float32)
            all_descriptors.extend(descriptors)
            image_descriptors.append(descriptors)
        else:
            # If no features detected, use zeros
            if detector == 'harris+brief':
                dim = 32  # BRIEF descriptor size
            else:
                dim = extractor.descriptorSize()
            image_descriptors.append(np.zeros((1, dim), dtype=np.float32))

        if (i + 1) % 100 == 0 or i + 1 == len(images):
            print(f"Processed {i + 1}/{len(images)} images")

    return np.array(all_descriptors), image_descriptors


def compute_bow_histograms(image_descriptors, kmeans_model):
    histograms = []

    print(f"Computing BoW histograms for {len(image_descriptors)} images...")

    for i, descriptors in enumerate(image_descriptors):
        if descriptors is None or len(descriptors) == 0:
            hist = np.zeros(kmeans_model.n_clusters)
        else:
            # predictions = kmeans_model.predict(descriptors)
            predictions = kmeans_model.predict(descriptors.astype(np.float32))
            hist, _ = np.histogram(predictions, bins=np.arange(kmeans_model.n_clusters + 1))
        histograms.append(hist)

        # Progress update
        if (i + 1) % 100 == 0 or i + 1 == len(image_descriptors):
            print(f"Processed {i + 1}/{len(image_descriptors)} histograms")

    return np.array(histograms)


class BoWImageClassifier(ModelBase):
    def __init__(self, num_clusters=100, feature_detector='sift', classifier_type='svm', name='BoW_Model', dataset_name=''):
        # super().__init__(name)
        super().__init__(f"{name}_{dataset_name}")
        self.num_clusters = num_clusters
        self.feature_detector = feature_detector
        self.classifier_type = classifier_type
        self.classifier = None
        self.kmeans = None
        self.scaler = StandardScaler()
        self.vocabulary = None
        self.training_stats = {
            'feature_extraction_time': 0,
            'clustering_time': 0,
            'classification_time': 0,
            'descriptor_count': 0,
            'avg_descriptors_per_image': 0
        }

    # def train(self, images, labels):
    #     print(f"Training BoW classifier with {self.num_clusters} clusters and {self.feature_detector} detector...")
    #
    #     # Feature extraction timing
    #     start_time = time.time()
    #     all_desc, image_desc = extract_keypoints_descriptors(images, self.feature_detector)
    #     if len(all_desc) == 0:
    #         raise ValueError("No descriptors were extracted from training images. Cannot train KMeans.")
    #     feature_time = time.time() - start_time
    #     self.training_stats['feature_extraction_time'] = feature_time
    #     self.training_stats['descriptor_count'] = len(all_desc)
    #     self.training_stats['avg_descriptors_per_image'] = len(all_desc) / len(images)
    #
    #     if len(all_desc) < self.num_clusters:
    #         adjusted = max(2, len(all_desc) // 2)
    #         print(
    #             f"[Warning] Only {len(all_desc)} descriptors found, reducing num_clusters from {self.num_clusters} to {adjusted}")
    #         self.num_clusters = adjusted
    #
    #     print(f"Extracted {len(all_desc)} descriptors in {feature_time:.2f} seconds")
    #     print(f"Average of {self.training_stats['avg_descriptors_per_image']:.1f} descriptors per image")
    #
    #     # Clustering timing
    #     print(f"Training K-means with {len(all_desc)} descriptors...")
    #     start_time = time.time()
    #     self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
    #     self.kmeans.fit(all_desc)
    #     clustering_time = time.time() - start_time
    #     self.training_stats['clustering_time'] = clustering_time
    #
    #     self.vocabulary = self.kmeans.cluster_centers_
    #     print(f"Visual vocabulary created in {clustering_time:.2f} seconds")
    #
    #     # BoW feature computation
    #     print("Computing BoW features...")
    #     bow_features = compute_bow_histograms(image_desc, self.kmeans)
    #     X = self.scaler.fit_transform(bow_features)
    #
    #     # Classification timing
    #     print("Training classifier...")
    #     start_time = time.time()
    #     if self.classifier_type == 'svm':
    #         parameters = {'C': [1, 10, 100], 'gamma': ['scale', 'auto', 0.1, 0.01]}
    #         grid = GridSearchCV(SVC(kernel='rbf', probability=True), parameters, cv=3)
    #         grid.fit(X, labels)
    #         self.classifier = grid.best_estimator_
    #         print(f"Best SVM parameters: {grid.best_params_}")
    #     else:
    #         raise ValueError(f"Unsupported classifier: {self.classifier_type}")
    #
    #     classification_time = time.time() - start_time
    #     self.training_stats['classification_time'] = classification_time
    #     print(f"Classifier trained in {classification_time:.2f} seconds")
    #
    #     self.is_trained = True
    #     self.training_time = feature_time + clustering_time + classification_time
    #
    #     # Log training statistics
    #     self._log_training_stats()

    def train(self, images, labels):
        """
        Train the Bag‑of‑Words image classifier.

        Parameters
        ----------
        images : list | np.ndarray
            Input images.
        labels : np.ndarray
            Corresponding labels for `images`.
        """
        print(f"Training BoW classifier with {self.num_clusters} clusters and "
              f"{self.feature_detector} detector...")

        # Set default classifier to 'svm' if not specified
        if not self.classifier_type:
            print("[Info] No classifier type specified. Defaulting to 'svm'.")
            self.classifier_type = 'svm'

        # ---------- Feature Extraction ----------
        start_time = time.time()
        all_desc, image_desc = extract_keypoints_descriptors(images,
                                                             self.feature_detector)
        if len(all_desc) == 0:
            raise ValueError("No descriptors were extracted from training images.")
        feature_time = time.time() - start_time

        # Book‑keeping statistics
        self.training_stats['feature_extraction_time'] = feature_time
        self.training_stats['descriptor_count'] = len(all_desc)
        self.training_stats['avg_descriptors_per_image'] = len(all_desc) / len(images)

        # Adjust the number of clusters if we have too few descriptors
        if len(all_desc) < self.num_clusters:
            adjusted = max(2, len(all_desc) // 2)
            print(f"[Warning] Only {len(all_desc)} descriptors found; "
                  f"reducing num_clusters from {self.num_clusters} to {adjusted}")
            self.num_clusters = adjusted

        print(f"Extracted {len(all_desc)} descriptors "
              f"({self.training_stats['avg_descriptors_per_image']:.1f} per image) "
              f"in {feature_time:.2f}s")

        # ---------- K‑means Clustering ----------
        print(f"Training K‑means with {self.num_clusters} clusters...")
        start_time = time.time()
        self.kmeans = KMeans(n_clusters=self.num_clusters,
                             random_state=42,
                             n_init=10)
        self.kmeans.fit(all_desc)
        clustering_time = time.time() - start_time
        self.training_stats['clustering_time'] = clustering_time

        self.vocabulary = self.kmeans.cluster_centers_
        print(f"Visual vocabulary created in {clustering_time:.2f}s")

        # ---------- BoW Histogram Encoding ----------
        print("Computing BoW histograms...")
        bow_features = compute_bow_histograms(image_desc, self.kmeans)
        X = self.scaler.fit_transform(bow_features)  # Scale for distance‑based models

        # ---------- Classifier Training ----------
        print("Training classifier...")
        start_time = time.time()

        if self.classifier_type == 'svm':
            # Existing SVM branch with grid search
            param_grid = {'C': [1, 10, 100],
                          'gamma': ['scale', 'auto', 0.1, 0.01]}
            grid = GridSearchCV(SVC(kernel='rbf', probability=True),
                                param_grid, cv=3)
            grid.fit(X, labels)
            self.classifier = grid.best_estimator_
            print(f"Best SVM parameters: {grid.best_params_}")

        elif self.classifier_type == 'knn':
            # K‑Nearest‑Neighbours
            self.classifier = KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='minkowski')
            self.classifier.fit(X, labels)

        elif self.classifier_type == 'decision_tree':
            # CART Decision Tree
            self.classifier = DecisionTreeClassifier(
                max_depth=None,
                random_state=42)
            self.classifier.fit(X, labels)

        elif self.classifier_type == 'random_forest':
            # Random Forest (bagging of decision trees)
            self.classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=42,
                n_jobs=-1)
            self.classifier.fit(X, labels)

        else:
            raise ValueError(f"Unsupported classifier: {self.classifier_type}")

        classification_time = time.time() - start_time
        self.training_stats['classification_time'] = classification_time
        print(f"Classifier trained in {classification_time:.2f}s")

        # ---------- Finalise ----------
        self.is_trained = True
        self.training_time = (feature_time + clustering_time + classification_time)

        # Save all training statistics to CSV
        self._log_training_stats()

        # Save visual word distribution
        self.plot_vocabulary_distribution(images)


    def predict(self, images):
        start_time = time.time()
        _, image_desc = extract_keypoints_descriptors(images, self.feature_detector)
        X = compute_bow_histograms(image_desc, self.kmeans)
        X_scaled = self.scaler.transform(X)
        predictions = self.classifier.predict(X_scaled)
        self.inference_time = time.time() - start_time
        return predictions

    def evaluate(self, test_images, test_labels, class_names=None, plot_confusion=False):
        _, image_desc = extract_keypoints_descriptors(test_images, self.feature_detector)
        X = compute_bow_histograms(image_desc, self.kmeans)
        X_scaled = self.scaler.transform(X)
        predictions = self.classifier.predict(X_scaled)
        self.inference_time = time.time()

        acc = accuracy_score(test_labels, predictions)
        prec = precision_score(test_labels, predictions, average='macro', zero_division=0)
        rec = recall_score(test_labels, predictions, average='macro', zero_division=0)
        f1 = f1_score(test_labels, predictions, average='macro', zero_division=0)

        if plot_confusion and class_names is not None:
            out_dir = self._get_dataset_subdir("figures")
            self._plot_confusion_matrix(test_labels, predictions, class_names, out_dir=out_dir)
            if hasattr(self.classifier, "predict_proba"):
                probs = self.classifier.predict_proba(X_scaled)
                self._plot_multiclass_roc(test_labels, probs, class_names, out_dir=out_dir)

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "predictions": predictions,
            "ground_truth": test_labels,
            "inference_time": self.inference_time
        }

    def save(self, filename):
        joblib.dump({
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'classifier': self.classifier,
            'feature_detector': self.feature_detector,
            'num_clusters': self.num_clusters,
            'training_stats': self.training_stats
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        data = joblib.load(filename)
        self.kmeans = data['kmeans']
        self.scaler = data['scaler']
        self.classifier = data['classifier']
        self.feature_detector = data['feature_detector']
        self.num_clusters = data['num_clusters']
        if 'training_stats' in data:
            self.training_stats = data['training_stats']
        self.is_trained = True
        print(f"Model loaded from {filename}")

    def _log_training_stats(self):
        """Log training statistics to a CSV file"""
        os.makedirs("results", exist_ok=True)

        results_dir = self._get_dataset_subdir("results")
        stats_file = os.path.join(results_dir, f"{self.name}_training_stats.csv")

        with open(stats_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['Num Clusters', self.num_clusters])
            writer.writerow(['Feature Detector', self.feature_detector])
            writer.writerow(['Classifier Type', self.classifier_type])
            writer.writerow(['Total Training Time', f"{self.training_time:.2f} seconds"])
            writer.writerow(
                ['Feature Extraction Time', f"{self.training_stats['feature_extraction_time']:.2f} seconds"])
            writer.writerow(['Clustering Time', f"{self.training_stats['clustering_time']:.2f} seconds"])
            writer.writerow(['Classification Time', f"{self.training_stats['classification_time']:.2f} seconds"])
            writer.writerow(['Total Descriptors', self.training_stats['descriptor_count']])
            writer.writerow(['Avg Descriptors/Image', f"{self.training_stats['avg_descriptors_per_image']:.1f}"])

            # Add dataset info if available
            if self.dataset_info:
                writer.writerow([''])
                writer.writerow(['Dataset Information'])
                for key, value in self.dataset_info.items():
                    writer.writerow([key, value])

        print(f"Training statistics saved to {stats_file}")


    def plot_vocabulary_distribution(self, images):
        """Plot the distribution of visual words in the vocabulary"""
        os.makedirs("figures", exist_ok=True)

        _, image_desc = extract_keypoints_descriptors(images, self.feature_detector)
        bow_features = compute_bow_histograms(image_desc, self.kmeans)
        summed = np.sum(bow_features, axis=0)

        plt.figure(figsize=(12, 6))
        plt.bar(np.arange(len(summed)), summed)
        plt.title(f"Visual Word Distribution - {self.name}", fontsize=14)
        plt.xlabel("Visual Word Index", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        # plt.tight_layout()

        # Create subdirectories based on dataset
        figure_dir = self._get_dataset_subdir("figures")
        results_dir = self._get_dataset_subdir("results")

        # Save figure to dataset-specific figure folder
        filename = os.path.join(figure_dir, f"{self.name}_visual_word_distribution.png")
        plt.savefig(filename, dpi=300)
        plt.show()

        # Export distribution to dataset-specific results folder
        csv_path = os.path.join(results_dir, f"{self.name}_word_distribution.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['word_index', 'frequency'])
            for i, freq in enumerate(summed):
                writer.writerow([i, freq])

        print(f"Word distribution exported to {csv_path}")

    def plot_accuracy_vs_clusters(self, images, labels, cluster_range=(50, 100, 200, 300, 400), n_folds=5):
        """
        Plot accuracy as a function of vocabulary size (number of clusters)
        """
        os.makedirs("figures", exist_ok=True)

        accuracies = []
        training_times = []

        # Save original number of clusters to restore later
        original_clusters = self.num_clusters

        print("Evaluating performance vs. cluster size:")
        for n_clusters in cluster_range:
            print(f"\nTesting with {n_clusters} clusters...")
            self.num_clusters = n_clusters
            cv_results, _ = self.cross_validate(
                images=images,
                labels=labels,
                n_folds=n_folds
            )
            accuracies.append(cv_results['accuracy_mean'])
            training_times.append(cv_results['train_time_mean'])

        # Restore original setting
        self.num_clusters = original_clusters

        # Plot accuracy vs clusters
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(cluster_range, accuracies, 'bo-', linewidth=2)
        plt.title('Accuracy vs. Vocabulary Size', fontsize=14)
        plt.xlabel('Number of Visual Words (k)', fontsize=12)
        plt.ylabel('Cross-Validation Accuracy', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        # Plot training time vs clusters
        plt.subplot(1, 2, 2)
        plt.plot(cluster_range, training_times, 'ro-', linewidth=2)
        plt.title('Training Time vs. Vocabulary Size', fontsize=14)
        plt.xlabel('Number of Visual Words (k)', fontsize=12)
        plt.ylabel('Training Time (seconds)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        # plt.tight_layout()
        # Create subdirectories for dataset-specific outputs
        figure_dir = self._get_dataset_subdir("figures")
        results_dir = self._get_dataset_subdir("results")

        # Save figure to dataset-specific folder
        filename = os.path.join(figure_dir, f"{self.name}_accuracy_vs_clusters.png")
        plt.savefig(filename, dpi=300)
        plt.show()

        # Export results to CSV in dataset-specific folder
        csv_path = os.path.join(results_dir, f"{self.name}_clusters_analysis.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['n_clusters', 'accuracy', 'training_time'])
            for i, n_clusters in enumerate(cluster_range):
                writer.writerow([n_clusters, accuracies[i], training_times[i]])

        print(f"Cluster analysis exported to {csv_path}")


class LocalFeatureObjectRecognizer(ModelBase):
    def __init__(self, feature_type='orb', matcher_type='flann', name='LocalFeatureMatcher'):
        super().__init__(name)
        self.feature_type = feature_type
        self.matcher_type = matcher_type
        self.templates = []  # list of (label, keypoints, descriptors, image)
        self.extractor = self._create_feature_extractor()
        self.matcher = self._create_matcher()

    def _create_feature_extractor(self):
        if self.feature_type == 'sift':
            return cv2.SIFT_create()
        elif self.feature_type == 'orb':
            return cv2.ORB_create()
        elif self.feature_type == 'brisk':
            return cv2.BRISK_create()
        else:
            raise ValueError("Unsupported feature type")

    def _create_matcher(self):
        if self.matcher_type == 'flann':
            if self.feature_type in ['sift', 'brisk']:
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
                return cv2.FlannBasedMatcher(index_params, search_params)
            else:
                return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif self.matcher_type == 'bf':
            norm = cv2.NORM_L2 if self.feature_type == 'sift' else cv2.NORM_HAMMING
            return cv2.BFMatcher(norm, crossCheck=True)
        else:
            raise ValueError("Unsupported matcher type")

    def train(self, images, labels):
        start_time = time.time()
        self.templates = []
        for img, label in zip(images, labels):
            keypoints, descriptors = self.extractor.detectAndCompute(img, None)
            if descriptors is not None:
                self.templates.append((label, keypoints, descriptors, img))
        self.training_time = time.time() - start_time
        self.is_trained = True
        print(f"Stored {len(self.templates)} templates for matching in {self.training_time:.2f} seconds")

    def predict(self, images):
        start_time = time.time()
        predictions = []
        for img in images:
            keypoints, descriptors = self.extractor.detectAndCompute(img, None)
            best_score = 0
            best_label = -1

            for label, train_kp, train_desc, train_img in self.templates:
                if descriptors is None or train_desc is None:
                    continue

                matches = self.matcher.match(descriptors, train_desc)
                matches = sorted(matches, key=lambda x: x.distance)
                score = sum([1.0 / (m.distance + 1e-5) for m in matches[:10]])

                if score > best_score:
                    best_score = score
                    best_label = label

            predictions.append(best_label)
        self.inference_time = time.time() - start_time
        return np.array(predictions)

    def save(self, filename):
        joblib.dump({
            'templates': self.templates,
            'feature_type': self.feature_type,
            'matcher_type': self.matcher_type
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        data = joblib.load(filename)
        self.templates = data['templates']
        self.feature_type = data['feature_type']
        self.matcher_type = data['matcher_type']
        self.extractor = self._create_feature_extractor()
        self.matcher = self._create_matcher()
        print(f"Model loaded from {filename}")
        self.is_trained = True