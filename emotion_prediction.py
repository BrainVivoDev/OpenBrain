import os
import shutil
import tempfile

import pickle
import pandas as pd
from typing import Tuple


from image_embedding_model.imagebind_embedding import calc_imagebind_embedding
from image_embedding_model.openclip_embedding import calc_openclip_embedding
from brain_embedding_model.brain_embedding import calc_brain_embedding

BRAIN_EMBEDDING_MAT_IMAGEBIND = (
    "./brain_embedding_model/imagebind_embedding_model.safetensors"
)


def copy_image_to_temp(src_image_path: str) -> str:
    """
    Copies the image at src_image_path into a newly created temporary directory.

    Args:
        src_image_path (str): The file path of the source image.

    Returns:
        str: The file path of the copied image in the temporary directory.
    """
    # Create a new temporary directory
    temp_dir = tempfile.mkdtemp(prefix="img_copy_")

    # Ensure the source file exists
    if not os.path.isfile(src_image_path):
        raise FileNotFoundError(f"Source image not found: {src_image_path}")

    # Define the destination path in the temporary directory
    image_name = os.path.basename(src_image_path)
    dest_image_path = os.path.join(temp_dir, image_name)

    # Copy the image
    shutil.copy2(src_image_path, dest_image_path)

    return temp_dir


def get_closest_emotion(
    valence: float, arousal: float, emotions_df: pd.DataFrame
) -> Tuple[str, str, str]:
    """
    Find the closest emotion in the 2D valence–arousal space.

    Args:
        valence (float): Valence value to classify.
        arousal (float): Arousal value to classify.

    Returns:
        Tuple[str, str, str]: (emotion name, color name, hex code)
    """
    best = None
    best_dist = float("inf")
    for index, row in emotions_df.iterrows():
        # Euclidean distance squared
        dv = valence - row["Valence"]
        da = arousal - row["Arousal"]
        dist = dv * dv + da * da
        if dist < best_dist:
            best_dist = dist
            best = row
    return best["Emotion"]


def predict_emotion(in_image_path):

    conversion_table = pd.read_csv(
        "./brain_embedding_model/va_transformation_table.csv"
    )

    new_image_temp_folder = copy_image_to_temp(in_image_path)

    image_emb_imagebind = calc_imagebind_embedding(new_image_temp_folder)
    data_imagebind = calc_brain_embedding(
        image_emb_imagebind,
        brain_model_file=BRAIN_EMBEDDING_MAT_IMAGEBIND,
    )

    with open("./valence_prediction.pickle", "rb") as f:
        valence_model = pickle.load(f)

    with open("./arousal_prediction.pickle", "rb") as f:
        arousal_model = pickle.load(f)

    valence_pred = valence_model.predict(
        data_imagebind[0]["brain_embedding"].reshape(1, -1)
    )[0]

    arousal_pred = arousal_model.predict(
        data_imagebind[0]["brain_embedding"].reshape(1, -1)
    )[0]

    emotion = get_closest_emotion(valence_pred, arousal_pred, conversion_table)
    print(f"Emotion: {emotion}, Valence:{valence_pred}, Arousal:{arousal_pred}")
    return emotion, valence_pred, arousal_pred


in_image_path = "/Users/shlomilifshits/Documents/OpenBrain/oasis/Images/Shark 2.jpg"
predict_emotion(in_image_path)
