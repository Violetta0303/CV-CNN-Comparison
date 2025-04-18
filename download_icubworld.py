import os
import zipfile
import ssl
import requests

ssl._create_default_https_context = ssl._create_unverified_context


def download_icubworld(dest_folder='datasets', version='1.0'):
    """
    Download and extract iCubWorld dataset (including transformations).
    If 'transformations' version is requested, will use local folder if URL is None or download fails.
    """

    dataset_info = {
        '1.0': {
            'url': 'https://www.icub.org/download/datasets/iCubWorld1.0.zip',
            'folder_name': 'iCubWorld1.0'
        },
        '28': {
            'url': 'https://www.icub.org/download/datasets/iCubWorld28_20150708.zip',
            'folder_name': 'iCubWorld28'
        },
        'transformations': {
            'url': None,  # Disable downloading for transformations; use local folder instead
            'folder_name': 'iCubWorld_Transformations'
        }
    }

    if version not in dataset_info:
        print(f"Unsupported dataset version: {version}")
        return None

    dataset_url = dataset_info[version]['url']
    dataset_folder = os.path.join(dest_folder, dataset_info[version]['folder_name'])
    zip_filename = os.path.join(dest_folder, f"{dataset_info[version]['folder_name']}.zip")

    # If folder already exists, skip downloading
    if os.path.exists(dataset_folder):
        print(f"Dataset already exists at: {dataset_folder}")
        return dataset_folder

    # For transformations: skip download if URL is None
    if dataset_url is None:
        print(f"No download URL provided for version '{version}'.")
        if os.path.exists(dataset_folder):
            print(f"Using existing local dataset at {dataset_folder}")
            return dataset_folder
        else:
            print(f"Local dataset folder not found: {dataset_folder}")
            print("Please download manually from: https://data.mendeley.com/datasets/647wgpxs5d/1")
            print("   and extract to: ", dataset_folder)
            return None

    os.makedirs(dest_folder, exist_ok=True)

    try:
        print(f"Downloading iCubWorld dataset version '{version}' from {dataset_url}...")

        headers = {'User-Agent': 'Mozilla/5.0'}
        with requests.get(dataset_url, headers=headers, stream=True, verify=False) as r:
            r.raise_for_status()
            with open(zip_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"Downloaded to {zip_filename}")
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)

        os.remove(zip_filename)
        print(f"Extraction completed. Dataset is available at: {dataset_folder}")

        return dataset_folder

    except Exception as e:
        print(f"Download or extraction failed: {e}")
        return None

