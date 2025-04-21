import os
import glob
import torch
import open_clip
import numpy as np
from PIL import Image


class OpenClipSingleton:
    """
    Singleton class to manage a single instance of the OpenCLIP model.

    Attributes
    ----------
    _instance : OpenClipSingleton or None
        The single instance of the class.
    device : str
        The device on which the model is loaded (e.g., "cpu" or "cuda").
    model : torch.nn.Module
        The OpenCLIP model instance.
    preprocess : callable
        The preprocessing pipeline for the model.
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
        OpenClipSingleton
            The single instance of the class.
        """
        if cls._instance is None:
            cls._instance = super(OpenClipSingleton, cls).__new__(cls)
            cls._instance.device = device
            # Load the model and preprocessing pipeline only once.
            cls._instance.model, cls._instance.preprocess = (
                open_clip.create_model_from_pretrained(
                    "hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
                )
            )
            cls._instance.model.to(device)
        return cls._instance

    def get_model(self) -> torch.nn.Module:
        """
        Get the OpenCLIP model instance.

        Returns
        -------
        torch.nn.Module
            The OpenCLIP model instance.
        """
        return self.model


def calculate_image_embedding(image: Image.Image) -> torch.Tensor:
    """
    Compute and return the normalized image embedding for a given PIL Image.

    Parameters
    ----------
    image : PIL.Image.Image
        The input image in PIL format.

    Returns
    -------
    torch.Tensor
        The normalized image embedding as a PyTorch tensor.
    """
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Get the singleton instance of OpenCLIP.
    clip_instance = OpenClipSingleton(device=device)

    # Preprocess the image, add a batch dimension, and move to device.
    image_input = clip_instance.preprocess(image).unsqueeze(0).to(
        clip_instance.device
    )

    # Compute the embedding with no gradient updates.
    with torch.no_grad():
        image_features = clip_instance.model.encode_image(image_input)
        # Normalize the feature vector.
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
            )

    return image_features


def img2openclip(images_path: str) -> tuple[list[dict[str, any]], np.ndarray]:
    """
    Convert images into OpenCLIP embeddings.

    Parameters
    ----------
    images_path : str
        Path to a directory containing images or a single image file.

    Returns
    -------
    out_data : list of dict[str, any]
        A list of dictionaries containing metadata and embeddings for each
        image.
    emb : np.ndarray
        A NumPy array containing all embeddings.
    """
    print("OpenCLIP input:", images_path)

    embedding_list = []
    file_names = []
    image_paths = []

    # Check if the path is a directory or a single file.
    if os.path.isdir(images_path):
        # Find all image files with common extensions in the directory.
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(images_path, ext)))
    elif os.path.isfile(images_path):
        files = [images_path]
    else:
        raise ValueError(f"{images_path} is not a valid file or directory.")

    for file_path in files:
        try:
            image_paths.append(file_path)
            # Open the image and ensure it is in RGB mode.
            image = Image.open(file_path).convert("RGB")
            embedding = calculate_image_embedding(image)
            # Remove batch dimension and move to CPU for NumPy conversion.
            embedding = embedding.squeeze(0).cpu().numpy()
            embedding_list.append(embedding)
            file_names.append(os.path.basename(file_path))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not embedding_list:
        raise ValueError(
            "No images were processed. "
            "Check the provided path and image files."
        )

    embeddings_array = np.vstack(embedding_list)

    output_data = []
    for i, image_name in enumerate(file_names):
        output_data.append(
            {
                "image_path": image_paths[i],
                "filename": image_name,
                "image_embedding": embeddings_array[i, :],
                "image_embedding_type": "openclip",
            }
        )

    return output_data, embeddings_array


def calc_openclip_embedding(images_path: str) -> list[dict[str, any]]:
    """
    Calculate OpenCLIP embeddings for a given set of images.

    Parameters
    ----------
    images_path : str
        Path to a directory containing images or a single image file.

    Returns
    -------
    embeddings : list of dict[str, any]
        A list of dictionaries containing metadata and embeddings for each
        processed image.
    """
    
    # Generate embeddings using img2openclip function.
    embeddings, _ = img2openclip(images_path)
    return embeddings

