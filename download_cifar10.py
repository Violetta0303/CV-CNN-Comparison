import os
import requests
import tarfile
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def download_cifar10(dest_folder='datasets'):
    """
    Download and extract CIFAR-10 dataset (Python version).
    """

    dataset_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    dataset_folder = os.path.join(dest_folder, 'cifar-10-batches-py')
    tar_filename = os.path.join(dest_folder, 'cifar-10-python.tar.gz')

    # Check if dataset already exists
    if os.path.exists(dataset_folder):
        print(f"Dataset already exists at: {dataset_folder}")
        return dataset_folder

    os.makedirs(dest_folder, exist_ok=True)

    try:
        print(f"Downloading CIFAR-10 dataset from {dataset_url}...")

        headers = {'User-Agent': 'Mozilla/5.0'}
        with requests.get(dataset_url, headers=headers, stream=True, verify=False) as r:
            r.raise_for_status()
            with open(tar_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"Downloaded to {tar_filename}")
        print("Extracting dataset...")
        with tarfile.open(tar_filename, 'r:gz') as tar_ref:
            tar_ref.extractall(dest_folder)

        os.remove(tar_filename)
        print(f"Extraction completed. Dataset is available at: {dataset_folder}")

        return dataset_folder

    except Exception as e:
        print(f"Download or extraction failed: {e}")
        return None


if __name__ == '__main__':
    download_cifar10()