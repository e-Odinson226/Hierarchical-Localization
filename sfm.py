import tqdm, tqdm.notebook

tqdm.tqdm = tqdm.notebook.tqdm  # notebook-friendly progress bars
from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
)
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
import pycolmap
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster


DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"

INPUT_DIR = DATA_DIR / "extracted_faces"
OUTPUT_DIR = DATA_DIR / "face_model"
print(INPUT_DIR)

for dir in INPUT_DIR.glob("*/"):
    # print(f"DIR: {str(dir).split('/')[-1]}")
    mirror_output_dir = OUTPUT_DIR / str(dir).split("/")[-1]
    mirror_output_dir.mkdir(parents=True, exist_ok=True)
    for img in dir.glob("**/*"):
        pass

outputs = DATA_DIR / "output"
sfm_pairs = outputs / "pairs-sfm.txt"
loc_pairs = outputs / "pairs-loc.txt"
sfm_dir = outputs / "sfm"
features = outputs / "features.h5"
matches = outputs / "matches.h5"

feature_conf = extract_features.confs["superpoint_aachen"]
matcher_conf = match_features.confs["superglue"]


references = [
    p.relative_to(INPUT_DIR).as_posix() for p in (INPUT_DIR / "erfan/").iterdir()
]
print(len(references), "mapping images")
# plot_images([read_image(DATA_DIR / r) for r in references[:4]], dpi=50)

# print(f"INPUT_DIR:{INPUT_DIR}, references:{references}")
extract_features.main(
    feature_conf, INPUT_DIR, image_list=references, feature_path=features
)
pairs_from_exhaustive.main(sfm_pairs, image_list=references)
match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

model = reconstruction.main(
    sfm_dir, INPUT_DIR, sfm_pairs, features, matches, image_list=references
)
fig = viz_3d.init_figure()

viz_3d.plot_reconstruction(fig, model, color="rgba(255,0,0,0.5)", name="mapping")
fig.show()

visualization.visualize_sfm_2d(model, INPUT_DIR, color_by="visibility", n=2)

model.export_PLY(str(OUTPUT_DIR / "erfan/model.ply"))
