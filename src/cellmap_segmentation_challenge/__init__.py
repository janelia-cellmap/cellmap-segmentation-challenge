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
            "score_submission",
            "match_crop_space",
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
