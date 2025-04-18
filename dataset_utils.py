import os
import numpy as np
import cv2
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Attempt import of auto-downloader
try:
    from download_icubworld import download_icubworld
except ImportError:
    download_icubworld = None


def load_icub_world(dataset_path=None, categories=None, max_images_per_category=None, version='1.0'):
    """
    Load iCubWorld dataset. Supports version '1.0', 'transformations', or '28'.

    Modification: Ensure consistent image shapes for numpy array conversion
    """
    import cv2
    import numpy as np
    import os

    # Auto-download if no path provided (you might need to implement this)
    if dataset_path is None:
        raise ValueError("Dataset path not provided")

    # === 1.0 version ===
    if version == '1.0':
        root = dataset_path
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Expected folder not found: {root}")

        all_images = []
        all_labels = []
        class_names = []

        class_folders = sorted(os.listdir(root))
        if categories:
            class_folders = [c for c in class_folders if c in categories]

        # Consistent image size
        img_size = (64, 64)

        for class_index, class_name in enumerate(class_folders):
            class_names.append(class_name)
            folder = os.path.join(root, class_name)

            image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm'))]
            print(f"[{class_name}] Found {len(image_files)} images in {folder}")

            if max_images_per_category:
                image_files = image_files[:max_images_per_category]

            for fname in image_files:
                fpath = os.path.join(folder, fname)
                img = cv2.imread(fpath)

                if img is not None:
                    # Ensure 3 channels
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    elif img.shape[2] > 3:
                        img = img[:, :, :3]

                    # Resize to consistent size
                    img_resized = cv2.resize(img, img_size)

                    all_images.append(img_resized)
                    all_labels.append(class_index)
                else:
                    print(f"Failed to read image: {fpath}")

        # Use np.stack to ensure consistent shape
        if all_images:
            images_array = np.stack(all_images)
            labels_array = np.array(all_labels)
            print(f"Loaded {len(images_array)} images from {len(class_names)} classes.")
            return images_array, labels_array, class_names
        else:
            raise ValueError("No images could be loaded from the dataset!")

    # === iCubWorld28 version ===
    elif version == '28':
        train_root = os.path.join(dataset_path, 'train')
        test_root = os.path.join(dataset_path, 'test')

        if not os.path.isdir(train_root) or not os.path.isdir(test_root):
            raise FileNotFoundError(f"Expected 'train' and 'test' folders under {dataset_path}")

        all_images = []
        all_labels = []
        class_names_set = set()
        image_label_pairs = []

        # Consistent image size
        img_size = (64, 64)

        # Nested loader for train/test/day*/class/
        def load_nested(root):
            for day in os.listdir(root):
                day_path = os.path.join(root, day)
                if not os.path.isdir(day_path):
                    continue
                for class_name in os.listdir(day_path):
                    class_path = os.path.join(day_path, class_name)
                    if not os.path.isdir(class_path):
                        continue
                    if categories and class_name not in categories:
                        continue
                    class_names_set.add(class_name)
                    for fname in os.listdir(class_path):
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm')):
                            image_label_pairs.append((os.path.join(class_path, fname), class_name))

        load_nested(train_root)
        load_nested(test_root)

        class_names = sorted(list(class_names_set))
        name_to_index = {name: idx for idx, name in enumerate(class_names)}

        for fpath, cname in image_label_pairs:
            img = cv2.imread(fpath)
            if img is not None:
                # Ensure 3 channels
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] > 3:
                    img = img[:, :, :3]

                # Resize to consistent size
                img_resized = cv2.resize(img, img_size)

                all_images.append(img_resized)
                all_labels.append(name_to_index[cname])
            else:
                print(f"Failed to read image: {fpath}")

        # Use np.stack to ensure consistent shape
        if all_images:
            images_array = np.stack(all_images)
            labels_array = np.array(all_labels)
            print(f"Loaded {len(images_array)} images from {len(class_names)} classes.")
            return images_array, labels_array, class_names
        else:
            raise ValueError("No images could be loaded from the dataset!")


def load_cifar10(split='train', dataset_path='datasets/cifar-10-batches-py'):
    import pickle
    import os

    def load_batch(file):
        with open(file, 'rb') as f:
            entry = pickle.load(f, encoding='bytes')
            data = entry[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = entry[b'labels']
        return data, labels

    if split == 'train':
        images, labels = [], []
        for i in range(1, 6):
            file = os.path.join(dataset_path, f'data_batch_{i}')
            imgs, lbls = load_batch(file)
            images.append(imgs)
            labels += lbls
        images = np.concatenate(images)
    elif split == 'test':
        file = os.path.join(dataset_path, 'test_batch')
        images, labels = load_batch(file)
    else:
        raise ValueError("split must be 'train' or 'test'")

    # Load class names
    meta_path = os.path.join(dataset_path, 'batches.meta')
    with open(meta_path, 'rb') as f:
        class_names = pickle.load(f, encoding='bytes')[b'label_names']
        class_names = [x.decode('utf-8') for x in class_names]

    return images, np.array(labels), class_names


def create_balanced_subset(images, labels, n_per_class=100):
    """
    Create a balanced subset with fixed number of samples per class.

    Parameters:
        images (np.ndarray): Input images
        labels (np.ndarray): Corresponding labels
        n_per_class (int): Samples per class

    Returns:
        subset_images, subset_labels
    """
    classes = np.unique(labels)
    subset_indices = []

    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) >= n_per_class:
            chosen = np.random.choice(cls_indices, n_per_class, replace=False)
        else:
            chosen = np.random.choice(cls_indices, n_per_class, replace=True)
        subset_indices.extend(chosen)

    subset_images = images[subset_indices] if not isinstance(images, list) else [images[i] for i in subset_indices]
    subset_labels = labels[subset_indices]

    return subset_images, subset_labels


def split_dataset(images, labels, test_size=0.2, random_state=42):
    """
    Split images and labels into train/test sets.

    Parameters:
        test_size (float): Proportion for test split
        random_state (int): Seed for reproducibility

    Returns:
        x_train, x_test, y_train, y_test
    """
    return train_test_split(images, labels, test_size=test_size, random_state=random_state, stratify=labels)


def preprocess_for_cnn(images, normalize=True, expand_dims=False):
    """
    Preprocess images for CNN input.

    Parameters:
        normalize (bool): Scale to [0,1]
        expand_dims (bool): Add channel for grayscale images

    Returns:
        preprocessed images
    """
    images = images.astype(np.float32)
    if normalize and images.max() > 1.0:
        images /= 255.0
    if expand_dims and len(images.shape) == 3:
        images = np.expand_dims(images, axis=-1)
    return images


def visualize_dataset_samples(images, labels, class_names, n_samples=10, figsize=(15, 8)):
    """
    Plot random images from the dataset with labels.

    Parameters:
        n_samples (int): Number of samples to show
    """
    indices = np.random.choice(len(images), min(n_samples, len(images)), replace=False)
    grid_size = int(np.ceil(np.sqrt(n_samples)))
    plt.figure(figsize=figsize)

    for i, idx in enumerate(indices):
        plt.subplot(grid_size, grid_size, i + 1)
        img = images[idx]
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f"Class: {class_names[labels[idx]]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_predictions(model, images, labels, class_names, n_samples=10, figsize=(15, 10)):
    """
    Plot model predictions alongside ground-truth labels.

    Parameters:
        model: Must have a .predict() method
    """
    indices = np.random.choice(len(images), min(n_samples, len(images)), replace=False)
    preds = model.predict(images[indices])
    grid_size = int(np.ceil(np.sqrt(n_samples)))

    plt.figure(figsize=figsize)
    for i, idx in enumerate(range(len(indices))):
        plt.subplot(grid_size, grid_size, i + 1)
        true_idx = indices[idx]
        img = images[true_idx]
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        true_label = labels[true_idx]
        pred_label = preds[idx]
        colour = 'green' if pred_label == true_label else 'red'
        plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}", color=colour)
        plt.axis('off')
    plt.tight_layout()
    plt.show()