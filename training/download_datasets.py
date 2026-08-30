import subprocess
import os
import zipfile

DATASETS = {
    "chat": "abhayayare/multi-turn-chatbot-conversation-dataset",
    "reasoning": "tiyabk/reasoning-training-data",
    "math": "reyesenrique/llm-mathematical-reasoning-benchmark-dataset",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def download_all():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, dataset in DATASETS.items():
        print(f"[{name}] Downloading {dataset} ...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", DATA_DIR, "--unzip"],
            check=True,
        )
    print("All datasets downloaded.")


if __name__ == "__main__":
    download_all()
