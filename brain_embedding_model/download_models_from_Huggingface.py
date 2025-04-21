import shutil
from huggingface_hub import hf_hub_download
import os


for file in [
    "openclip_embedding_model.safetensors",
    "imagebind_embedding_model.safetensors",
]:
    cached_path = hf_hub_download(
        repo_id="openbrain-bv/brainvivo-openbrain",
        filename=file,
    )

    # Define the target path in the current working directory
    target_path = os.path.join(os.getcwd(), file)

    # Copy it
    shutil.copy(cached_path, target_path)

    print(f"File copied to: {target_path}")
