import lazy_loader as lazy

# Lazy-load submodules
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "predict": ["predict"],
        "process": ["process"],
        "train": ["train"],
        "visualize": ["visualize"],
        "evaluate": [
            "package_submission",
            "save_numpy_class_arrays_to_zarr",
            "save_numpy_class_labels_to_zarr",
            "score_submission",
        ],
    },
)

from .config import (
    CROP_NAME,
    PREDICTIONS_PATH,
    PROCESSED_PATH,
    RAW_NAME,
    REPO_ROOT,
    BASE_DATA_PATH,
    SEARCH_PATH,
    SUBMISSION_PATH,
    TRUTH_PATH,
)

from . import utils
