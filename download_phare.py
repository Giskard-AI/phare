"""
Script to clone the Phare dataset from Hugging Face Hub.
Dataset: https://huggingface.co/datasets/giskardai/phare
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def fetch_phare_dataset(local_dir: str | Path = "./phare_data") -> Path:
    """
    Clone the Phare dataset from Hugging Face Hub to a local directory.

    Args:
        local_dir: Path where the dataset will be downloaded.

    Returns:
        Path to the downloaded dataset.
    """
    local_dir = Path(local_dir)

    print(f"Cloning giskardai/phare dataset to: {local_dir.absolute()}")

    downloaded_path = snapshot_download(
        repo_id="giskardai/phare",
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=["**/*.jsonl"],
    )

    print(f"Dataset successfully downloaded to: {downloaded_path}")
    return Path(downloaded_path)


def main():
    parser = argparse.ArgumentParser(
        description="Clone the Phare dataset from Hugging Face Hub"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./phare_data",
        help="Local directory to clone the dataset to (default: ./phare_data)",
    )

    args = parser.parse_args()

    fetch_phare_dataset(args.path)


if __name__ == "__main__":
    main()
