import shutil
import tempfile
import os

# Make a temporary directory to work in
workdir = tempfile.mkdtemp()
SEARCH_PATH = os.path.join(
    workdir, *"data/{dataset}/{dataset}.zarr/recon-1/{name}".split("/")
)
PREDICTION_PATH = os.path.join(
    workdir, *"data/predictions/{dataset}.zarr/{crop}".split("/")
)
PROCESSED_PATH = os.path.join(
    workdir, *"data/processed/{dataset}.zarr/{crop}".split("/")
)
SUBMISSION_PATH = os.path.join(workdir, "submission.zarr")

# %%
# def test_fetch_data():
from cellmap_segmentation_challenge.cli import fetch_data_cli

os.makedirs(os.path.join(workdir, "data"), exist_ok=True)
fetch_data_cli(
    crops="116,117,118,119",
    raw_padding=0,
    dest="data",
    access_mode="a",
    fetch_all_em_resolutions=False,
    batch_size=256,
    num_workers=32,
)

# %%
# def test_train()
shutil.copy("test_train_config.py", workdir)
shutil.copy("test_process_config.py", workdir)
os.chdir(workdir)

from cellmap_segmentation_challenge.cli import train_cli
from cellmap_segmentation_challenge.utils import make_datasplit_csv

make_datasplit_csv(
    classes=["mito", "er"],
    search_path="data/{dataset}/{dataset}.zarr/recon-1/{name}",
    validation_prob=0.5,
)

train_cli("test_train_config.py")

# %%
# def test_predict()
from cellmap_segmentation_challenge.cli import predict_cli

predict_cli(
    "test_predict_config.py",
    crops="116",
    output_path=PREDICTION_PATH,
    do_orthoplanes=False,
    overwrite=True,
)

# %%
# def test_process()
from cellmap_segmentation_challenge.cli import process_cli

process_cli(
    "test_process_config.py",
    crops="116",
    overwrite=True,
    input_path=PREDICTION_PATH,
    output_path=PROCESSED_PATH,
)

# %%
# def test_pack_results()
from cellmap_segmentation_challenge.cli import package_submission_cli

package_submission_cli(PROCESSED_PATH, SUBMISSION_PATH, overwrite=True)

# %%
# def test_evaluate()
from cellmap_segmentation_challenge.cli import evaluate_cli
from cellmap_segmentation_challenge.evaluate import INSTANCE_CLASSES

evaluate_cli(
    SUBMISSION_PATH.replace(".zarr", ".zip"),
    result_file=os.path.join(workdir, "perfect_result.json"),
    truth_path=SUBMISSION_PATH,
    instance_classes=INSTANCE_CLASSES,
)

# %%
# Now test evaluation with up/downsampling of perfect predictions

# Save upsampled processed data
...

# Evaluate upsampled data
...

# Save downsampled processed data
...

# Evaluate downsampled data
...

# Compare results
...

# %%
# Now test evaluation with up/downsampling of imperfect predictions

# Make imperfect predictions
...

# Save upsampled processed data
...

# Evaluate upsampled data
...

# Save downsampled processed data
...

# Evaluate downsampled data
...

# Compare results
...

# %%
# Cleanup
os.chdir("..")
shutil.rmtree(workdir)
