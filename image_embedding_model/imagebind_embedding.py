import os
import torch
import numpy as np

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


class ImagebindSingleton:
    """
    Singleton class to manage a single instance of the ImageBind model.

    Attributes
    ----------
    _instance : ImagebindSingleton or None
        The single instance of the class.
    device : str
        The device on which the model is loaded (e.g., "cpu" or "cuda").
    model : torch.nn.Module
        The ImageBind model instance.
    """
    _instance = None

    def __new__(cls, device: str = "cpu"):
        """
        Create or return the single instance of the class.

        Parameters
        ----------
        device : str, optional
            The device on which to load the model (default is "cpu").

        Returns
        -------
        ImagebindSingleton
            The single instance of the class.
        """
        if cls._instance is None:
            cls._instance = super(ImagebindSingleton, cls).__new__(cls)
            cls._instance.device = device
            # Initialize the model once
            cls._instance.model = imagebind_model.imagebind_huge(
                pretrained=True
            )
            cls._instance.model.eval()
            cls._instance.model.to(device)
        return cls._instance

    def get_model(self) -> torch.nn.Module:
        """
        Get the ImageBind model instance.

        Returns
        -------
        torch.nn.Module
            The ImageBind model instance.
        """
        return self.model


def img2imagebind(images_path: str) -> tuple[list[dict[str, any]], np.ndarray]:
    """
    Convert images into ImageBind embeddings.

    Parameters
    ----------
    images_path : str
        Path to a directory containing images or a single image file.

    Returns
    -------
    out_data : list of dict[str, any]
        A list of dictionaries containing the image path, filename, and
        embedding.
    emb : np.ndarray
        A NumPy array containing all embeddings.
    """
    print("ImageBind input:", images_path)

    # Determine the device to use (MPS for macOS if available)
    device = "cpu"

    # Collect image file paths
    if (
        images_path.lower().endswith(".jpg") or
        images_path.lower().endswith(".jpeg")
    ):
        files_list = [images_path]
    else:
        files_list = []
        for root, _, files in os.walk(images_path):
            for file in files:
                if (
                    file.lower().endswith(".jpg") or
                    file.lower().endswith(".jpeg")
                ):
                    file_path = os.path.join(root, file)
                    files_list.append(file_path)

    # Use the singleton to get the ImageBind model instance
    imagebind_singleton = ImagebindSingleton(device=device)
    model = imagebind_singleton.get_model()
    model.eval()
    model.to(device)

    # Prepare inputs for the model
    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data(
            files_list, device
        )
    }

    # Generate embeddings using the model
    with torch.no_grad():
        embeddings = model(inputs)

    embeddings_array = embeddings[ModalityType.VISION].numpy()

    # Construct output data with metadata and embeddings
    output_data = []
    for i, image_path in enumerate(files_list):
        output_data.append(
            {
                "image_path": image_path,
                "filename": os.path.basename(image_path),
                "image_embedding": embeddings_array[i, :],
                "image_embedding_type": "imagebind",
            }
        )

    return output_data, embeddings_array


def calc_imagebind_embedding(images_path: str) -> list[dict[str, any]]:
    """
    Calculate ImageBind embeddings for a given set of images.

    Parameters
    ----------
    images_path : str
        Path to a directory containing images or a single image file.

    Returns
    -------
    embeddings : list of dict[str, any]
        A list of dictionaries containing the image path, filename,
        and embedding for each processed image.
    """
    embeddings, _ = img2imagebind(images_path)
    return embeddings
