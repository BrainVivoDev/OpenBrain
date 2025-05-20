from huggingface_hub import snapshot_download
import os

repo_id   = "openbrain-bv/brainvivo-openbrain"
lancedb_folder = "unsplash.lance"

# Ensure current working directory is set to the script's directory
path_of_embedding_explorer = os.path.dirname(os.path.abspath(__file__))

# Define the target path in the current working directory
target_path = path_of_embedding_explorer + "/resources/lancedb_data"

snapshot_download(
    repo_id=repo_id,
    allow_patterns=f"{lancedb_folder}/*",
    local_dir=target_path,
    local_dir_use_symlinks=False,
    resume_download=True
)

print(f"File copied to: {target_path}")