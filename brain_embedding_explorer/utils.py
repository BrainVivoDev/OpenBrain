import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from brain_embedding_model.brain_embedding import brain_embed, load_model

'''
This script contains utility functions for processing brain embeddings and
activating parcels. It includes functions to:
1. Get a summary of the highest activated brain regions for a given image.
2. Create a new LanceDB table with computed embeddings for images of your choice.
'''

# Paths to resources
project_dir = Path(__file__).resolve().parent.parent
BRAIN_EMBEDDING_MAT = project_dir / "brain_embedding_model" / "imagebind_embedding_model.safetensors"
TRAINSET_EMBEDDINGS_PATH = project_dir / "brain_embedding_explorer" / "resources" / "full_trainset_imagebind_embeddings.csv"
PARCELS_MAP_PATH = project_dir / "brain_embedding_explorer" / "resources" / "parcels_seg_Schaefer2018_200Parcels_Kong2022_17Networks_order.nii"
PERCEPTION_ATLAS_PATH = project_dir / "brain_embedding_explorer" / "resources" / "Perception_atlas.csv"


def activated_parcels(imagebind_image_embedding, image_name: str, space="1024") -> tuple[dict[str, str], pd.DataFrame]:
    """ 
    Get the most activated brain regions in a brain embedding based on a given imagebind image embedding.
    The activation is defined as the percentile of activation for each voxel relative to the activation in that voxel
    in response to a set of images. Averaging is done across all voxels in a given brain region (parcel).
    The brain regions are defined by the Schaefer atlas (200 parcels, 17 networks).
    
    Parameters
    ----------
    - imagebind_image_embedding: the imagebind embedding of the image
    - image_name: the name of the image
    - space: the space of the brain embedding (1024 or 60k), for future functionality

    Returns
    -------
    - a dictionary with the top 15 brain regions, and their role in perception
    - the brain embedding of the image
    """
    
    # Load the training data
    train_data = pd.read_csv(TRAINSET_EMBEDDINGS_PATH, index_col=0)

    # Future functionality - provide the results rom all the gray matter
    # if space == "60k":
        # brain_model_dict = load_file(BRAIN_EMBEDDING_FULL_GM_MAT)
    
    # Load the brain embedding model
    brain_model = load_model(BRAIN_EMBEDDING_MAT)
    # Get the brain embedding of the training data
    brain_embeddings = brain_embed(brain_model, train_data)

    # get the mean and std activation of each voxel from the training data
    voxel_distribution = pd.DataFrame(
        index=["mean", "std"],
        columns=brain_embeddings.columns,
    )
    for i in brain_embeddings.columns:
        # get the mean and std of each voxel
        voxel_distribution.loc["mean", i] = np.mean(brain_embeddings.loc[:, i])
        voxel_distribution.loc["std", i] = np.std(brain_embeddings.loc[:, i])

    # Get the brain embedding of the input image
    imagebind_embedding = pd.DataFrame(
        [imagebind_image_embedding], index=[image_name], columns=range(1024)
    )
    image_full_br = brain_embed(brain_model, imagebind_embedding)
    # fourth, get the mean voxel precentile of each parcel
    nii_data = nib.load(PARCELS_MAP_PATH)
    brain_map = nii_data.get_fdata()  # a 3D numpy array of shape (102, 102, 64).
    parcels = dict()
    for i in image_full_br.columns:
        [x1, x2, x3] = eval(i)
        parcel = brain_map[x1, x2, x3]
        if voxel_distribution.loc["std", i] == 0:
            # setting the percentile to 0 if the std is 0 across all training data in this voxel
            percentile = 0
        else:
            percentile = (
                image_full_br.loc[:, i].values[0] - voxel_distribution.loc["mean", i]
            ) / voxel_distribution.loc["std", i]
        if parcel not in parcels:
            parcels[parcel] = [percentile]
        else:
            parcels[parcel].append(percentile)
    # get the mean percentile of each parcel
    parcel_mean = dict()
    for parcel in parcels:
        if parcel != 0:
            parcel_mean[parcel] = np.nanmean(parcels[parcel])
    # get the top 10 parcels
    parcel_mean = sorted(parcel_mean.items(), key=lambda x: x[1], reverse=True)
    top_parcels = parcel_mean[:15]
    # get the words/roles of each parcel from the atlas csv
    atlas_csv = pd.read_csv(PERCEPTION_ATLAS_PATH)
    parcels_terms = dict()
    for parcel in top_parcels:
        parcel_id = parcel[0].round(0)
        parcel_name = f"{atlas_csv[atlas_csv['Parcel Value'] == parcel_id]['Region name'].values[0]}, {atlas_csv[atlas_csv['Parcel Value'] == parcel_id]['Network'].values[0]} network"
        parcel_terms = atlas_csv[atlas_csv["Parcel Value"] == parcel_id][
            "Perception"
        ].values[0]
        parcels_terms[parcel_name] = parcel_terms
    return parcels_terms, image_full_br




