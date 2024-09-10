# %%
# Example usage ===========================================================
from cellmap_segmentation_challenge.evaluate import (
    score,
    get_leaderboard_stat,
    print_leaderboard_stats,
)

label_dict = {"mito": 0}
test_datasets = {
    "jrc_hela-2": "/nrs/cellmap/bennettd/data/jrc_hela-2/jrc_hela-2.zarr/recon-1/labels/groundtruth/crop155/mito/s0"
}
resolution = 8
distance_max = 80.0
norm_slope = 1.0
truth_datasets = {
    "jrc_hela-2": "/nrs/cellmap/bennettd/data/jrc_hela-2/jrc_hela-2.zarr/recon-1/labels/groundtruth/crop155/{label}/{resolution_level}"
}
overall_score, summary_results, dataset_results = score(
    label_dict, test_datasets, resolution, distance_max, norm_slope, truth_datasets
)

# %%
# Example usage ===========================================================
# NOTE: adding custom datasets to the leaderboard for this example
CLASS_DATASETS = {"mito": ["jrc_hela-2"]}
leaderboard_stats = {k: {} for k in label_dict.keys()}
for label in label_dict.keys():
    leaderboard_stats[label]["mean"], leaderboard_stats[label]["std"] = (
        get_leaderboard_stat(label, dataset_results)
    )
print_leaderboard_stats(leaderboard_stats)

# %%
