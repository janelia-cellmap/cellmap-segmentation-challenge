import shutil
import sys
from time import time
import numpy as np
import requests
from tqdm import tqdm
from cellmap_segmentation_challenge.utils import get_tested_classes
from cellmap_segmentation_challenge import TRUTH_PATH
import zarr
import git

from upath import UPath
from scipy import ndimage as ndi


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 1.0


def _disk(radius: int) -> np.ndarray:
    """Binary disk structuring element."""
    r = int(radius)
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= r * r


def _boundary_bands(G: np.ndarray, band: int, connectivity: int = 1):
    """
    Returns:
      inner_band: pixels in G near boundary (preferred FN removal candidates)
      outer_band: pixels outside G near boundary (preferred FP addition candidates)
    """
    G = G.astype(bool)
    if band <= 0:
        inner = G.copy()
        outer = (~G).copy()
        return inner, outer

    # Use disk SE for spatial realism
    se = _disk(band)

    # Inner band = G minus an eroded version of G
    G_er = ndi.binary_erosion(G, structure=se, iterations=1, border_value=0)
    inner_band = np.logical_and(G, ~G_er)

    # Outer band = dilated G minus G
    G_di = ndi.binary_dilation(G, structure=se, iterations=1)
    outer_band = np.logical_and(G_di, ~G)

    return inner_band, outer_band


def _sample_indices(mask: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly sample k True locations (flat indices) without replacement."""
    idx = np.flatnonzero(mask)
    if k <= 0 or idx.size == 0:
        return np.array([], dtype=np.int64)
    k = min(k, idx.size)
    return rng.choice(idx, size=k, replace=False)


def _add_blob(
    P: np.ndarray,
    candidate: np.ndarray,
    add_k: int,
    rng: np.random.Generator,
    blob_radius: int = 8,
    n_seeds: int = 6,
) -> int:
    """
    Adds 'blob-like' FP pixels by planting random seeds and dilating them.
    Returns how many pixels were actually added.
    """
    if add_k <= 0:
        return 0
    cand_idx = np.flatnonzero(candidate)
    if cand_idx.size == 0:
        return 0

    # Create seed map
    seeds = np.zeros_like(P, dtype=bool)
    seeds_n = min(n_seeds, cand_idx.size)
    seed_idx = rng.choice(cand_idx, size=seeds_n, replace=False)
    seeds.flat[seed_idx] = True

    # Grow seeds into blobs
    se = _disk(max(1, blob_radius))
    blob = ndi.binary_dilation(seeds, structure=se, iterations=1)

    # Restrict to candidate region + not already in P
    blob = np.logical_and(blob, candidate)
    blob = np.logical_and(blob, ~P)

    # If blob too big, randomly subsample to exact add_k
    blob_idx = np.flatnonzero(blob)
    if blob_idx.size == 0:
        return 0

    choose = (
        blob_idx
        if blob_idx.size <= add_k
        else rng.choice(blob_idx, size=add_k, replace=False)
    )
    before = P.sum()
    P.flat[choose] = True
    return int(P.sum() - before)


def _remove_blob(
    P: np.ndarray,
    candidate: np.ndarray,
    rem_k: int,
    rng: np.random.Generator,
    blob_radius: int = 8,
    n_seeds: int = 6,
) -> int:
    """
    Removes 'blob-like' FN pixels by planting seeds inside candidate and dilating them.
    Returns how many pixels were actually removed.
    """
    if rem_k <= 0:
        return 0
    cand_idx = np.flatnonzero(candidate)
    if cand_idx.size == 0:
        return 0

    seeds = np.zeros_like(P, dtype=bool)
    seeds_n = min(n_seeds, cand_idx.size)
    seed_idx = rng.choice(cand_idx, size=seeds_n, replace=False)
    seeds.flat[seed_idx] = True

    se = _disk(max(1, blob_radius))
    blob = ndi.binary_dilation(seeds, structure=se, iterations=1)

    # Restrict to candidate region + currently in P
    blob = np.logical_and(blob, candidate)
    blob = np.logical_and(blob, P)

    blob_idx = np.flatnonzero(blob)
    if blob_idx.size == 0:
        return 0

    choose = (
        blob_idx
        if blob_idx.size <= rem_k
        else rng.choice(blob_idx, size=rem_k, replace=False)
    )
    before = P.sum()
    P.flat[choose] = False
    return int(before - P.sum())


def perturb_mask_realistic(
    G: np.ndarray,
    target_iou: float,
    p_fn: float = 0.5,
    band: int = 3,
    style: str = "ring",  # "ring" or "blob"
    blob_radius: int = 8,
    blob_seeds: int = 6,
    rng: np.random.Generator | None = None,
    max_tries: int = 4000,
):
    """
    Make a perturbed mask P from ground-truth G with approx target IoU, using spatially realistic errors.

    Construction:
      Start P = G
      Remove r pixels (FN) mostly from inner boundary band
      Add a pixels (FP) mostly from outer boundary band

    Exact-count IoU relationship (counts):
      IoU = (g - r) / (g + a)
    """

    if rng is None:
        rng = np.random.default_rng()

    G = G.astype(bool)
    g = int(G.sum())
    if g == 0:
        raise ValueError("Ground-truth mask is empty; cannot target IoU reliably.")
    if not (0.0 < target_iou <= 1.0):
        raise ValueError("target_iou must be in (0, 1].")
    if target_iou == 1.0:
        return G.copy()

    bg = ~G
    b = int(bg.sum())

    # Choose r then compute a = (g - r)/t - g
    def a_from_r(r: int) -> int:
        return int(round((g - r) / target_iou - g))

    # Prefer both FP and FN: try to find r such that a>0 and feasible
    r0 = int(np.clip(round(p_fn * g * (1 - target_iou)), 1, g - 1))
    best = None
    jitter = max(5, g // 80)

    for _ in range(max_tries):
        r = int(np.clip(r0 + rng.integers(-jitter, jitter + 1), 0, g))
        a = a_from_r(r)
        if 0 <= a <= b:
            if r > 0 and a > 0:
                best = (r, a)
                break
            if best is None:
                best = (r, a)

    if best is None:
        raise RuntimeError(
            "Could not find feasible (FN removals r, FP adds a). Try different target_iou."
        )

    r, a = best

    P = G.copy()

    # Candidate bands near boundary
    inner_band, outer_band = _boundary_bands(G, band=band)

    # If band degenerates (tiny objects), fall back to whole regions
    inner_cand = inner_band if inner_band.any() else G
    outer_cand = outer_band if outer_band.any() else (~G)

    # --- Apply FN removals ---
    if style == "ring":
        # remove from inner band first
        rem_idx = _sample_indices(inner_cand & P, r, rng)
        P.flat[rem_idx] = False

        # If not enough (rare), remove remaining from anywhere inside
        remaining = r - rem_idx.size
        if remaining > 0:
            rem2 = _sample_indices(G & P, remaining, rng)
            P.flat[rem2] = False

    elif style == "blob":
        removed = _remove_blob(
            P, inner_cand, r, rng, blob_radius=blob_radius, n_seeds=blob_seeds
        )
        remaining = r - removed
        if remaining > 0:
            # finish by ring-like sampling if blobs didn't hit exact count
            rem_idx = _sample_indices((G & P), remaining, rng)
            P.flat[rem_idx] = False
    else:
        raise ValueError("style must be 'ring' or 'blob'")

    # --- Apply FP additions ---
    if style == "ring":
        add_idx = _sample_indices(outer_cand & (~P), a, rng)
        P.flat[add_idx] = True

        remaining = a - add_idx.size
        if remaining > 0:
            add2 = _sample_indices((~G) & (~P), remaining, rng)
            P.flat[add2] = True

    elif style == "blob":
        added = _add_blob(
            P, outer_cand, a, rng, blob_radius=blob_radius, n_seeds=blob_seeds
        )
        remaining = a - added
        if remaining > 0:
            add_idx = _sample_indices(((~G) & (~P)), remaining, rng)
            P.flat[add_idx] = True

    # (Optional) return counts for debugging
    achieved = iou(G, P)
    info = {
        "target_iou": target_iou,
        "achieved_iou": achieved,
        "g": g,
        "fn_removed_r": r,
        "fp_added_a": a,
        "band": band,
        "style": style,
    }
    return P, info


def format_coordinates(coordinates):
    """
    Format the coordinates to a string.

    Parameters
    ----------
    coordinates : list
        List of coordinates.

    Returns
    -------
    str
        Formatted string.
    """
    return f"[{';'.join([str(c) for c in coordinates])}]"


def construct_test_crop_manifest(
    path_root: str,
    search_path: str = "{path_root}/{dataset}/groundtruth.zarr/{crop}/{label}",
    write_path: str | None = (UPath(__file__).parent / "test_crop_manifest.csv").path,
    verbose: bool = False,
) -> None | list[str]:
    """
    Construct a manifest file for testing crops from a given path.

    Parameters
    ----------
    path_root : str
        Path to the directory containing the datasets.
    search_path : str, optional
        Format string to search for the crops. The default is "{path_root}/{dataset}/groundtruth.zarr/{crop}/{label}". The function assumes that the keys appear in the file tree in the following order: 1) "path_root", 2) "dataset", 3) "crop", 4) "label"
    write_path : str, optional
        Path to write the manifest file. The default is "test_crop_manifest.csv".
    verbose : bool, optional
        Print verbose output. The default is False.
    """
    # Get the tested classes
    tested_classes = set(get_tested_classes())

    # Construct the manifest
    manifest = [
        "crop_name,dataset,class_label,voxel_size,translation,shape",
    ]

    # Get datasets
    datasets = [
        d.name
        for d in UPath(
            search_path.split("{dataset}")[0].format(path_root=path_root)
        ).iterdir()
        if d.is_dir()
    ]

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")

        # Get crops
        crops = [
            d.name
            for d in UPath(
                search_path.split("{crop}")[0].format(
                    path_root=path_root, dataset=dataset
                )
            ).iterdir()
            if d.is_dir()
        ]
        for crop in crops:
            if verbose:
                print(f"\tProcessing crop: {crop}")

            # Get labels in crop
            had_classes = set(
                [
                    d.name
                    for d in UPath(
                        search_path.split("{label}")[0].format(
                            path_root=path_root, dataset=dataset, crop=crop
                        )
                    ).iterdir()
                    if d.is_dir()
                ]
            )

            # Filter for tested classes
            labels = list(had_classes.intersection(tested_classes))

            for label in labels:
                if verbose:
                    print(f"\t\tProcessing label: {label}")
                # Get the zarr file
                zarr_file = zarr.open(
                    search_path.format(
                        path_root=path_root, dataset=dataset, crop=crop, label=label
                    ),
                    mode="r",
                )

                # Get the metadata
                metadata = zarr_file.attrs.asdict()["multiscales"][0]["datasets"][0][
                    "coordinateTransformations"
                ]
                for meta in metadata:
                    if meta["type"] == "translation":
                        translation = format_coordinates(meta["translation"])
                    elif meta["type"] == "scale":
                        voxel_size = format_coordinates(meta["scale"])
                shape = format_coordinates(zarr_file["s0"].shape)
                manifest.append(
                    f"{crop.replace('crop', '')},{dataset},{label},{voxel_size},{translation},{shape}"
                )
                if verbose:
                    print(
                        f"\t\t\tScale: {voxel_size}\n\t\t\tTranslation: {translation}\n\t\t\tShape: {shape}"
                    )
    if write_path is None:
        return manifest

    # Write the manifest
    with open(write_path, "w") as f:
        f.write("\n".join(manifest))

    print(f"Manifest written to: {write_path}")


def construct_truth_dataset(
    path_root: str,
    search_path: str = "{path_root}/{dataset}/groundtruth.zarr/{crop}/{label}",
    destination: str = TRUTH_PATH,
    write_path: str = "{crop}/{label}",
):
    """
    Construct a consolidated Zarr file for the groundtruth datasets, to use for evaluation.

    Parameters
    ----------
    path_root : str
        Path to the directory containing the datasets.
    search_path : str, optional
        Format string to search for the crops. The default is "{path_root}/{dataset}/groundtruth.zarr/{crop}/{label}". The function assumes that the keys appear in the file tree in the following order: 1) "path_root", 2) "dataset", 3) "crop", 4) "label"
    destination : str, optional
        Path to write the consolidated Zarr file. The default is "cellmap-segmentation-challenge/data/ground_truth.zarr".
    write_path : str, optional
        Format string to write the crops to within the destination Zarr. The default is "{crop}/{label}".
    """
    start_time = time()

    # Get the test crop manifested
    manifest = construct_test_crop_manifest(path_root, search_path, write_path=None)

    # Open the destination Zarr folder
    if UPath(destination).exists():
        print(f"Removing existing ground truth dataset at: {destination}")
        shutil.rmtree(destination)
    # ground_truth = zarr.open_group(destination, mode="w")
    ground_truth = zarr.open_group(destination, mode="a")

    # Make a pool for parallel processing
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pool = ThreadPoolExecutor()

    # Copy the ground truth datasets
    futures = []
    crops_started = set()
    for line in tqdm(manifest[1:], desc="Formatting ground truth..."):
        crop = line.split(",")[0]
        if crop not in crops_started:
            crops_started.add(crop)
            ground_truth.create_group(f"crop{crop}")
        futures.append(
            pool.submit(copy_gt, line, search_path, path_root, write_path, ground_truth)
        )

    for future in tqdm(as_completed(futures), total=len(futures), desc="Copying..."):
        future.result()

    print(f"Ground truth dataset written to: {destination}")
    print(f"Done in {time() - start_time}!")


def copy_gt(line, search_path, path_root, write_path, ground_truth):
    # Get the metadata from the manifest
    crop, dataset, class_label, voxel_size, translation, shape = line.split(",")
    crop_name = f"crop{crop}"
    voxel_size = eval(voxel_size.replace(";", ","))
    translation = eval(translation.replace(";", ","))
    shape = eval(shape.replace(";", ","))

    # Open the source ground truth zarr file
    path = search_path.format(
        path_root=path_root, dataset=dataset, crop=crop_name, label=class_label
    )
    zarr_file = zarr.open(path, mode="r")

    # Write the dataset to the destination Zarr
    print(f"Writing {write_path.format(crop=crop_name, label=class_label)}")
    dataset = ground_truth.create_dataset(
        write_path.format(crop=crop_name, label=class_label),
        data=zarr_file["s0"],
        shape=shape,
        dtype=zarr_file["s0"].dtype,
        overwrite=True,
        # fill_value=0,
        dimension_separator="/",
    )
    dataset.attrs["voxel_size"] = voxel_size
    dataset.attrs["translation"] = translation
    dataset.attrs["shape"] = shape


def download_file(url, dest):
    response = requests.get(url)
    response.raise_for_status()
    with open(dest, "wb") as f:
        f.write(response.content)


def format_string(string: str, format_kwargs: dict) -> str:
    """
    Convenience function to format a string with only the keys present in both the stringand in the `format_kwargs`. When all keys in the `format_kwargs` are present in `string` (in brackets), the function will return `string.format(**format_kwargs)` exactly. When none of the keys in the `format_kwargs` are present in the string, the function will return the original string, without error.

    Parameters
    ----------
    string : str
        The string to format.
    format_kwargs : dict
        The dictionary of key-value pairs to format the string with.

    Returns
    -------
    str
        The formatted string

    Examples
    --------
    format_string("this/{thing}", {})  # returns "this/{thing}"
    format_string("this/{thing}", {"thing":"that", "but":"not this"}) # returns "this/that"
    """
    new_kwargs = {}
    # Find the keys that are present in the string
    for key_chunk in string.split("{")[1:]:
        key = key_chunk.split("}")[0]
        if key in format_kwargs:
            new_kwargs[key] = format_kwargs[key]
        else:
            new_kwargs[key] = "{" + key + "}"
    string = string.format(**new_kwargs)
    return string


def get_git_hash() -> str:
    """
    Get the current git hash of the repository.

    Returns
    -------
    str
        The current git hash, or "unknown" if not running from a git repository.
    """
    try:
        repo = git.Repo(UPath(__file__).parent, search_parent_directories=True)
        sha = repo.head.object.hexsha
        return sha
    except git.exc.InvalidGitRepositoryError:
        return "unknown"


def get_singleton_dim(shape: tuple) -> int | None:
    """Return the index of the first size-1 spatial dimension, or None."""
    arr = np.where([s == 1 for s in shape])[0]
    return int(arr[0]) if arr.size > 0 else None


def squeeze_singleton_dim(data, dim: int):
    """Squeeze *dim* from a tensor, or from each tensor value in a dict."""
    if isinstance(data, dict):
        return {k: v.squeeze(dim=dim) for k, v in data.items()}
    return data.squeeze(dim=dim)


def unsqueeze_singleton_dim(data, dim: int):
    """Insert *dim* into a tensor, or into each tensor value in a dict."""
    if isinstance(data, dict):
        return {k: v.unsqueeze(dim=dim) for k, v in data.items()}
    return data.unsqueeze(dim=dim)


def structure_model_output(
    outputs,
    classes: list,
    num_channels_per_class: int | None = None,
) -> dict:
    """
    Convert raw model output into the nested dict expected by CellMapDatasetWriter.

    The returned structure maps array-name → class-name → tensor (or tensor when
    there is one channel per class).

    Parameters
    ----------
    outputs:
        Either a plain tensor of shape ``(B, C, ...)`` or a dict produced by the
        model.  Dict keys may be the class names themselves *or* arbitrary array
        names (e.g. resolution levels); each dict value is a tensor of shape
        ``(B, C, ...)``.
    classes:
        Ordered list of class names.
    num_channels_per_class:
        When ``> 1``, each class occupies this many consecutive channels in the
        channel dimension.  ``None`` means one channel per class.

    Returns
    -------
    dict
        Nested structure suitable for writing via ``dataset_writer[idx] = ...``.
    """
    if isinstance(outputs, dict) and set(outputs.keys()) == set(classes):
        # Dict keys are already the class names; values are per-class tensors
        return {"output": outputs}
    elif isinstance(outputs, dict):
        # Dict with non-class keys (e.g. resolution levels): split each value
        structured = {}
        for k, v in outputs.items():
            if num_channels_per_class is not None:
                expected_channels = len(classes) * num_channels_per_class
                if v.shape[1] != expected_channels:
                    raise ValueError(
                        f"Number of output channels ({v.shape[1]}) in '{k}' does not match "
                        f"expected ({expected_channels} = {len(classes)} classes × "
                        f"{num_channels_per_class} channels per class)."
                    )
                structured[k] = {
                    class_name: v[
                        :,
                        i * num_channels_per_class : (i + 1) * num_channels_per_class,
                    ]
                    for i, class_name in enumerate(classes)
                }
            elif v.shape[1] == len(classes):
                structured[k] = v
            else:
                raise ValueError(
                    f"Number of output channels ({v.shape[1]}) does not match number of "
                    f"classes ({len(classes)}). Should be a multiple of the number of classes."
                )
        return structured
    elif num_channels_per_class is not None:
        expected_channels = len(classes) * num_channels_per_class
        if outputs.shape[1] != expected_channels:
            raise ValueError(
                f"Number of output channels ({outputs.shape[1]}) does not match "
                f"expected ({expected_channels} = {len(classes)} classes × "
                f"{num_channels_per_class} channels per class)."
            )
        return {
            "output": {
                class_name: outputs[
                    :,
                    i * num_channels_per_class : (i + 1) * num_channels_per_class,
                ]
                for i, class_name in enumerate(classes)
            }
        }
    elif outputs.shape[1] == len(classes):
        return {"output": outputs}
    else:
        raise ValueError(
            f"Number of output channels ({outputs.shape[1]}) does not match number of "
            f"classes ({len(classes)}). Should be a multiple of the number of classes."
        )


def get_data_from_batch(batch, keys, device):
    if len(keys) > 1:
        inputs = {key: batch[key].to(device) for key in keys}
    else:
        # Assumes the model output is a single tensor
        inputs = batch[keys[0]].to(device)
    return inputs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python utils.py <path_root>")
        sys.exit(1)
    elif len(sys.argv) == 2 or sys.argv[2] == "dataset":
        construct_truth_dataset(
            sys.argv[1],
        )
    elif sys.argv[2] == "manifest":
        construct_test_crop_manifest(sys.argv[1], verbose=True)
