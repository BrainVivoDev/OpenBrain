import numpy as np
import pandas as pd
from safetensors.numpy import load_file


def brain_embed(
    model: np.ndarray | pd.DataFrame,
    imagebind_embedding: np.ndarray | pd.DataFrame
) -> np.ndarray | pd.DataFrame:
    """
    Convert imagebind embedding to brain embedding using the model.

    Parameters
    ----------
    model : np.ndarray or pd.DataFrame
        The model weights for converting the imagebind embedding to brain
        embedding.
        If a DataFrame, its index will be used as column names for the output
        DataFrame.
    imagebind_embedding : np.ndarray or pd.DataFrame
        The imagebind embedding(s) to convert to brain embedding. If a
        DataFrame,
        its index will be used as row names for the output DataFrame.

    Returns
    -------
    brain_embedding : np.ndarray or pd.DataFrame
        The brain embedding(s). If either input was a DataFrame, the output
        will be a
        DataFrame with appropriate indices and columns.
    """
    embedding_names: pd.Index | None = None
    if isinstance(model, pd.DataFrame):
        embedding_names = model.index
        model = model.values
        
    sample_names: pd.Index | None = None
    if isinstance(imagebind_embedding, pd.DataFrame):
        sample_names = imagebind_embedding.index
        imagebind_embedding = imagebind_embedding.values

    if imagebind_embedding.ndim == 1:
        imagebind_embedding = imagebind_embedding.reshape(1, -1)
    imagebind_embedding = np.concatenate(
        (imagebind_embedding, np.ones((imagebind_embedding.shape[0], 1))),
        axis=1
    )
    
    brain_embedding: np.ndarray | pd.DataFrame = np.dot(
        imagebind_embedding, model.T
    )
    brain_embedding = np.clip(brain_embedding, 0, 1)

    if (embedding_names is not None) or (sample_names is not None):
        brain_embedding = pd.DataFrame(brain_embedding)
        if embedding_names is not None:
            brain_embedding.columns = embedding_names
        if sample_names is not None:
            brain_embedding.index = sample_names
    return brain_embedding


def calc_brain_embedding(
    image_embeddings: list[dict[str, any]],
    brain_model_file: str
) -> list[dict[str, any]]:
    """
    Calculate brain embeddings for a list of image embeddings using a model
    loaded from a safetensors file.

    Parameters
    ----------
    image_embeddings : list of dict[str, any]
        A list of dictionaries, each containing at least the key 'image_embedding'
        with its embedding as a NumPy array.
    brain_model_file : str
        Path to the safetensors file containing the model weights.

    Returns
    -------
    image_embeddings : list of dict[str, any]
        The input list, where each dictionary is updated with a new key
        'brain_embedding'
        containing the computed brain embedding as a NumPy array.
    """
    # Load the model from the safetensors file
    brain_model = load_model(brain_model_file)

    # Extract imagebind embeddings from the input dictionaries
    imagebind_features = np.array([
        embedding_dict['image_embedding']
        for embedding_dict in image_embeddings
    ])
        
    # Compute brain embeddings using the loaded model
    predicted_brain_embeddings = brain_embed(brain_model, imagebind_features)

    # Convert DataFrame to NumPy array if necessary
    if isinstance(predicted_brain_embeddings, pd.DataFrame):
        predicted_brain_embeddings = predicted_brain_embeddings.to_numpy()

    # Update each dictionary in the input list with the brain embedding
    for index, sample in enumerate(image_embeddings):
        sample["brain_embedding"] = predicted_brain_embeddings[index, :]

    return image_embeddings


def load_model(brain_model_file: str) -> pd.DataFrame:
    """
    Load the brain embedding model from a file.

    Parameters
    ----------
    brain_model_file : str
        Path to the file containing the brain embedding model.

    Returns
    -------
    pd.DataFrame
        The loaded brain embedding model as a DataFrame.
    """
    # Load the model from the file
    brain_model_dict = load_file(brain_model_file)
    
    model = pd.DataFrame(
        brain_model_dict['weights'],
    )
    
    #Check that '_x' '_y' and '_z' are in the model
    if  all(key in brain_model_dict for key in ['_x', '_y', '_z']):
        coords = [
            f"[{', '.join([str(a) for a in list(c)])}]" 
            for c in np.vstack((
                brain_model_dict['_x'],
                brain_model_dict['_y'],
                brain_model_dict['_z']
            )).T]
        model.index = coords
    
    if '_f' in brain_model_dict:
        model.columns = brain_model_dict['_f']
    
    return model
