from image_embedding_model.imagebind_embedding import calc_imagebind_embedding
from image_embedding_model.openclip_embedding import calc_openclip_embedding
from brain_embedding_model.brain_embedding import calc_brain_embedding

BRAIN_EMBEDDING_MAT_IMAGEBIND = (
    "./brain_embedding_model/imagebind_embedding_model.safetensors"
)


BRAIN_EMBEDDING_MAT_OPENCLIP = (
    "./brain_embedding_model/openclip_embedding_model.safetensors"
)


OASIS_IMAGE_DIR = "./examples/sample_OASIS_images"

## imagebind
image_emb_imagebind = calc_imagebind_embedding(OASIS_IMAGE_DIR)
data_imagebind = calc_brain_embedding(
    image_emb_imagebind,
    brain_model_file=BRAIN_EMBEDDING_MAT_IMAGEBIND,
)

## openclip
image_emb_openclip = calc_openclip_embedding(OASIS_IMAGE_DIR)
data_openclip = calc_brain_embedding(
    image_emb_openclip,
    brain_model_file=BRAIN_EMBEDDING_MAT_OPENCLIP,
)
