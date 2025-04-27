from pathlib import Path
import lancedb
import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from image_embedding_model.imagebind_embedding import img2imagebind
from brain_embedding_model.brain_embedding import calc_brain_embedding

# Paths to resources
project_dir = Path(__file__).resolve().parent.parent
BRAIN_EMBEDDING_MAT = project_dir / "brain_embedding_model" / "imagebind_embedding_model.safetensors"
LANCE_DB_PATH = project_dir / "brainTwins_embedding_explorer" / "resources" / "lancedb_data"


def create_lancedb_table(images_folder_path:str, table_name:str, lancedb_data_path:str) -> str:
    """ 
    Create a LanceDB table with the computed embeddings for images of your choice.
    The table is created with the cosine metric for the vector column.

    Call example:
    >>> lancedb_new_table_path = create_lancedb_table("path/to/images", "table_name", "path/to/lancedb/folder")
    
    Parameters
    ----------
    - images_folder_path: the path to the folder containing the images. Images should be in jpg or jpeg formats.
    - table_name: the name of the table to be created
    - lancedb_data_path: the path to the folder where the LanceDB database is stored.

    Returns
    -------
    - a path to the LanceDB table
    """
    
    # Check if the images folder exists
    if not os.path.exists(images_folder_path):
        raise FileNotFoundError(f"Images folder not found at {images_folder_path}. Please provide a valid path.")
    # Check if the LanceDB data path exists
    if not os.path.exists(lancedb_data_path):
        raise FileNotFoundError(f"LanceDB data path not found at {lancedb_data_path}. Please provide a valid path.")
    # Check if the table name is valid
    if not table_name:
        raise ValueError("Table name cannot be empty. Please provide a valid table name.")
    # Check if another table with the same name already exists
    db = lancedb.connect(lancedb_data_path)
    if table_name in db.table_names():
        raise ValueError(f"A table with the name {table_name} already exists. Please provide a different name.")
    
    # For each image in the folder, get the brain embedding and save it to a list
    data = []
    image_paths = [image_path for image_path in Path(images_folder_path).iterdir() 
               if image_path.suffix.lower() in {'.jpg', '.jpeg'}]
    for image_path in image_paths:
        image_name = image_path.stem
        
        # Get the brain embedding of the image
        image_embeddings, _ = img2imagebind(str(image_path))
        brain_embedding_list = calc_brain_embedding(image_embeddings, BRAIN_EMBEDDING_MAT)
        brain_embedding = brain_embedding_list[0]["brain_embedding"]
    
        # Append the record to the list
        data.append(
            {
                "image_path": str(image_path).split('/')[-1], # not using the full image path to avoid local paths
                "filename": image_name,
                "brain_embedding": brain_embedding,
            }
        )

    # Create the LanceDB table
    try:
        # Convert the data to a list of dictionaries.
        formatted_data = [
            {
                "image_path": record["image_path"],
                "filename": record["filename"],
                "vector": record["brain_embedding"].tolist(),
            }
            for record in data
        ]
        # Create the LanceDB table
        table = db.create_table(name=table_name, data=formatted_data)
        if len(formatted_data) < 256:
            print(
                "Warning: duplicating the data in order to be able to set an index (cosine metric)"
            )
            table.add(formatted_data)
        table.create_index(metric="cosine", vector_column_name="vector")
    
    except Exception as e:
        print(e)
    
    return f"{lancedb_data_path}/{table_name}"
 

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a new DB from a set of images"
    )
    parser.add_argument(
        "images_path",
        help="Path to the folder that contains the images",
    )
    parser.add_argument(
        "table_name",
        help="Name of the table to be created",
    )
    args = parser.parse_args()

    # Check if the required files exist
    if not BRAIN_EMBEDDING_MAT.exists():
        raise FileNotFoundError(f"Brain embedding model not found at {BRAIN_EMBEDDING_MAT}. Please download the model. See README.md for instructions.")
    if not LANCE_DB_PATH.exists():
        os.makedirs(LANCE_DB_PATH, exist_ok=True)
        print(f"Created directory {LANCE_DB_PATH} for LanceDB data.")
    if not args.images_path:
        raise ValueError("Please provide a path to the images folder.")
    elif not os.path.exists(args.images_path):
        raise FileNotFoundError(f"Images folder not found at {args.images_path}. Please provide a valid path.")
    if not args.table_name:
        raise ValueError("Please provide a name for the table.")

    new_lancedb_table_path = create_lancedb_table(args.images_path, args.table_name, LANCE_DB_PATH)
    print(f"Created {args.table_name} DB at {new_lancedb_table_path}.")

if __name__ == "__main__":
    main()