import lancedb
import numpy as np
import os
import sys
import pandas as pd
import streamlit as st
import tempfile
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from image_embedding_model.imagebind_embedding import img2imagebind
from brain_embedding_model.brain_embedding import calc_brain_embedding

from PIL import Image
from utils import activated_parcels
from dotenv import load_dotenv

load_dotenv()


# -----------------------------
# Constants
project_dir = Path(__file__).resolve().parent.parent
st.session_state.brain_embedding_mat = (
    project_dir / "brain_embedding_model" / "imagebind_embedding_model.safetensors"
)
st.session_state.tbl = "oasis_1024embeddings_5"
st.session_state.lance_db_path = (
    project_dir / "brain_embedding_explorer" / "resources" / "lancedb_data"
)
st.session_state.k = 5
st.session_state.valence = (
    project_dir / "brain_embedding_explorer" / "resources" / "diff_embedding.csv"
)
st.session_state.images_path = os.getenv("OASIS_IMAGE_DIR")

if os.getenv("OASIS_IMAGE_DIR") is None:
    st.error(
        "Please set the OASIS_IMAGE_DIR environment variable to the path of the images. See README file for more details."
    )


# -----------------------------
# Streamlit App
logo_path = project_dir / "graphics" / "OB-Logo.svg"
# st.image(str(logo_path.resolve()), use_container_width=False)
st.image(logo_path, use_container_width=False)
st.title("Brain Embedding Explorer")

# Initialize keys in session state if they don't exist
if "image_brain_embedding" not in st.session_state:
    st.session_state["image_brain_embedding"] = None
    st.session_state["new_brain_embedding"] = None
if "uploaded_image_name" not in st.session_state:
    st.session_state["uploaded_image_name"] = None

# if st.session_state.brain_embedding_mat does not exist, create it
if not os.path.exists(st.session_state.brain_embedding_mat):
    st.error(
        f"Missing DB file: {st.session_state.brain_embedding_mat}. Please check README file and download resources using download_models_from_Huggingface.py."
    )

# PART 1: Open an existing LanceDB table
st.header("1. Connect to Existing LanceDB Table")
db_dir = st.text_input("Enter LanceDB directory:", value=st.session_state.lance_db_path)
table_name = st.text_input("Enter table name:", value=st.session_state.tbl)
top_terms = None

if st.button("Connect to DB"):
    try:
        db = lancedb.connect(db_dir)
        table = db.open_table(table_name)
        st.session_state.db = db
        st.session_state.table = table
        st.success(f"Connected! Table '{table_name}' opened.")

    except Exception as e:
        st.error(f"Error connecting to DB: {e}")

# PART 2: Query via perturbation of a selected image
st.header("2. Modify the emotion induced by an image")
st.write(
    "Upload an image, compute its brain embedding, tweak valence and/or arousal, and retrieve similar images from the DB."
)

uploaded_image = st.file_uploader(
    "Upload your image", type=["jpg", "jpeg"], key="image1"
)

if uploaded_image:
    # Display the uploaded image
    st.image(
        Image.open(uploaded_image), caption="Query Image", use_container_width=True
    )
    # Check if this is a *new* image or we can use the embedding from the session state
    if uploaded_image.name != st.session_state["uploaded_image_name"]:
        # Update session state with the new filename
        st.session_state["uploaded_image_name"] = uploaded_image.name
        # Recompute the embedding (new image)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_image.getvalue())
                temp_path1 = tmp.name

                # Get the imagebind embedding of the image
                image_embeddings, emb_mat = img2imagebind(temp_path1)
                st.session_state.imagebind_embedding = image_embeddings[0][
                    "image_embedding"
                ]
                # Get the brain embedding of the image
                brain_embeddings = calc_brain_embedding(
                    image_embeddings, st.session_state.brain_embedding_mat
                )
                st.session_state.image_brain_embedding = brain_embeddings[0][
                    "brain_embedding"
                ]
                st.write("image embedding updated")
        except Exception as e:
            st.error(f"Error during image embedding extraction: {e}")

st.write("How much would you like to modify the emotions of this image?")

# Create three columns for the slider and labels
col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    st.write("More negative")

with col2:
    valence_slider = st.slider(
        "Emotion scaler",
        min_value=-10,
        max_value=10,
        value=0,
        step=2,
        label_visibility="collapsed",
    )

with col3:
    st.write("More positive")

if st.button("Modify emotion"):
    if "table" not in st.session_state:
        st.error("Please connect to the DB first.")
    elif not uploaded_image:
        st.warning("Please upload an image.")
    else:
        try:
            if "image_embedding" not in st.session_state:
                # Compute image embeddings
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(uploaded_image.getvalue())
                    temp_path1 = tmp.name

                # Get the imagebind embedding of the image
                image_embeddings, emb_mat = img2imagebind(temp_path1)
                st.session_state.imagebind_embedding = image_embeddings[0][
                    "image_embedding"
                ]
                # Get the brain embedding of the image
                brain_embeddings = calc_brain_embedding(
                    image_embeddings, st.session_state.brain_embedding_mat
                )
                st.session_state.image_brain_embedding = brain_embeddings[0][
                    "brain_embedding"
                ]

            # open st.session_state.valence
            valence_diff_embedding = np.loadtxt(st.session_state.valence, delimiter=',')

            emotional_supporting_voxels = np.where(
                (valence_diff_embedding > 0.02) | (valence_diff_embedding < -0.02)
            )[0]
            valence_diff_embedding = valence_diff_embedding[emotional_supporting_voxels]
            # modify valence by valence_slider*2*diff_embedding/10 so it reaches -2 to 2
            st.session_state.new_brain_embedding = (
                st.session_state.image_brain_embedding.copy()
            )
            st.session_state.new_brain_embedding[emotional_supporting_voxels] += (
                valence_slider * 0.2
            ) * valence_diff_embedding
            if (st.session_state.new_brain_embedding.max() > 1).any() or (
                st.session_state.new_brain_embedding.min() < 0
            ).any():
                st.write("Perturbed embedding is out of bounds. Clipping to [0, 1].")
                st.session_state.new_brain_embedding = np.clip(
                    st.session_state.new_brain_embedding, 0, 1
                )  # make sure no value is above 1 or below 0
            results_df = (
                st.session_state.table.search(
                    st.session_state.new_brain_embedding.tolist()
                )
                .limit(st.session_state.k)
                .to_pandas()
            )
            st.write("### Images with modified valence:")
            cols = st.columns(st.session_state.k)
            for idx, row in results_df.iterrows():
                img_path = row.get("image_path").replace(
                    "/Users/jasminebv/OASIS_database_2016/images",
                    os.getenv("OASIS_IMAGE_DIR"),
                )
                if img_path and os.path.exists(img_path):
                    img = Image.open(img_path)
                    cols[idx % st.session_state.k].image(
                        img, caption=row["filename"], use_container_width=True
                    )
                else:
                    cols[idx % st.session_state.k].write(row["filename"])
                    st.error(
                        f"Image not found at path: {img_path}. Please export oasis/Images as instructed. See README file for more details."
                    )
        except Exception as e:
            st.error(f"Error during valence modification: {e}")

# PART 3: Explore brain response induced by an image
st.header("3. Explore brain response induced by an image")
st.write(
    "Explore the brain response induced by the image, based on the brain embeddings of the uploaded image above."
)

if st.button("Compute brain response"):
    if not uploaded_image:
        st.warning("Please upload an image.")

    # Check if this is a *new* image or we can use the embedding from the session state
    elif uploaded_image.name != st.session_state["uploaded_image_name"] or (
        "image_embedding" not in st.session_state
    ):
        # Update session state with the new filename
        st.session_state["uploaded_image_name"] = uploaded_image.name

        # Recompute the embedding (new image)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_image.getvalue())
                temp_path1 = tmp.name
            # Get the imagebind embedding of the image
            image_embeddings, emb_mat = img2imagebind(temp_path1)
            st.session_state.imagebind_embedding = image_embeddings[0][
                "image_embedding"
            ]
        except Exception as e:
            st.error(f"Error during image embedding extraction: {e}")

    # Check if the image embedding is available
    if uploaded_image and st.session_state.imagebind_embedding is not None:
        st.write("### Activated regions:")
        top_terms, brain_response = activated_parcels(
            st.session_state.imagebind_embedding, uploaded_image.name, "1024"
        )
        for idx, row in enumerate(top_terms):
            st.write(f"**{row}**:\n {top_terms[row]}")


# PRINT top_terms as a table that can be downloaded as csv
if top_terms is not None:
    st.write("")
    parcels_df = pd.DataFrame(list(top_terms.items()), columns=["Region", "Perception"])

    # Align the download button to the right
    col1, col2 = st.columns([2, 1])
    with col2:
        st.download_button(
            label="Download results as CSV",
            data=parcels_df.to_csv(),
            file_name="activated_regions.csv",
            mime="text/csv",
        )


# -----------------------------
# To do: Allow arousal transformation as well. Allow combined transformation (valence & arousal)
