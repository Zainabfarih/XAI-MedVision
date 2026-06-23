"""Download the model checkpoints from Hugging Face into the expected folders.

Usage:
    pip install huggingface_hub
    python scripts/download_models.py

Files land in:
    checkpoints/ijepa/ijepa_best.pth
    checkpoints/probe/probe_best.pth
which is where both the notebooks and the web demo look for them.
"""
import os
import shutil

from huggingface_hub import hf_hub_download

# TODO: replace with your Hugging Face model repo id.
REPO_ID = "zainabFarih/lung-ct-nodule-ijepa-vit-small"

# Map: file name on the Hub -> local destination (relative to project root).
FILES = {
    "ijepa_best.pth": os.path.join("checkpoints", "ijepa", "ijepa_best.pth"),
    "probe_best.pth": os.path.join("checkpoints", "probe", "probe_best.pth"),
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    for filename, rel_dest in FILES.items():
        dest = os.path.join(ROOT, rel_dest)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        cached = hf_hub_download(repo_id=REPO_ID, filename=filename)
        shutil.copy(cached, dest)
        print(f"OK  {filename}  ->  {rel_dest}")
    print("Done. Checkpoints ready.")


if __name__ == "__main__":
    main()
