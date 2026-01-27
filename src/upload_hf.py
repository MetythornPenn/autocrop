import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, login


def main():
    repo_id = "metythorn/autocrop"
    models_dir = Path("models")

    if not models_dir.exists():
        raise FileNotFoundError("models directory not found")

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(models_dir),
        repo_type="model",
        path_in_repo="",
        commit_message="Upload models from local models/ directory",
    )

    readme_content = """---
language: en
library_name: autocrop_kh
tags:
  - onnx
  - document-segmentation
  - image-processing
---

# autocrop_kh models

ONNX model files for the autocrop_kh library.

Default model:
- autocrop_model_v2.onnx
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".md") as tmp:
        tmp.write(readme_content)
        tmp_path = tmp.name

    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=tmp_path,
        path_in_repo="README.md",
        commit_message="Add README",
    )


if __name__ == "__main__":
    main()
