"""Submission processing and main evaluation entry point."""

import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time

import numpy as np
import zarr
from tqdm import tqdm
from upath import UPath
from zarr.errors import PathNotFoundError

from ...config import SUBMISSION_PATH, TRUTH_PATH, INSTANCE_CLASSES
from .aggregation import update_scores
from .config import EvaluationConfig
from .scoring import score_label, empty_label_score
from .zip_utils import unzip_file


def ensure_zgroup(path: UPath) -> zarr.Group:
    """
    Ensure that the given path can be opened as a zarr Group. If a .zgroup is not present, add it.
    """
    try:
        return zarr.open(path.path, mode="r")
    except PathNotFoundError:
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory.")
        # Add a .zgroup file to force Zarr-2 format
        (path / ".zgroup").write_text('{"zarr_format": 2}')
        return zarr.open(path.path, mode="r")


def get_evaluation_args(
    volumes,
    submission_path,
    truth_path=TRUTH_PATH,
    instance_classes=INSTANCE_CLASSES,
) -> list[tuple]:
    """
    Get the arguments for scoring each label in the submission.
    Args:
        volumes (list): A list of volumes to score.
        submission_path (str): The path to the submission volume.
        truth_path (str): The path to the ground truth volume.
        instance_classes (list): A list of instance classes.
    Returns:
        A list of tuples containing the arguments for each label to be scored.
    """
    if not isinstance(volumes, (tuple, list)):
        volumes = [volumes]
    score_label_arglist = []
    for volume in volumes:
        submission_path = UPath(submission_path)
        pred_volume_path = submission_path / volume
        logging.info(f"Scoring {pred_volume_path}...")
        truth_path = UPath(truth_path)

        # Find labels to score
        pred_labels = [a for a in ensure_zgroup(pred_volume_path).array_keys()]

        crop_name = pred_volume_path.name
        truth_labels = [a for a in ensure_zgroup(truth_path / crop_name).array_keys()]

        found_labels = list(set(pred_labels) & set(truth_labels))
        missing_labels = list(set(truth_labels) - set(pred_labels))

        # Score_label arguments for each label
        score_label_arglist.extend(
            [
                (
                    pred_volume_path / label if label in found_labels else None,
                    label,
                    crop_name,
                    truth_path,
                    instance_classes,
                )
                for label in truth_labels
            ]
        )
        logging.info(f"Missing labels: {missing_labels}")

    return score_label_arglist


def missing_volume_score(
    truth_path, volume, instance_classes=INSTANCE_CLASSES
) -> list[tuple]:
    """
    Score a missing volume as 0's, congruent with the score_volume function.

    Args:
        truth_path (str): The path to the ground truth volume.
        volume (str): The name of the volume.
        instance_classes (list): A list of instance classes.

    Returns:
        dict: A dictionary of scores for the volume.

    Example usage:
        scores = missing_volume_score('truth.zarr/test_volume')
    """
    logging.info(f"Scoring missing volume {volume}...")
    truth_path = UPath(truth_path)
    truth_volume_path = truth_path / volume

    # Find labels to score
    truth_labels = [a for a in ensure_zgroup(truth_volume_path).array_keys()]

    # Score each label
    scores = {
        label: empty_label_score(label, volume, instance_classes, truth_path)
        for label in truth_labels
    }

    return scores


def ensure_valid_submission(submission_path: UPath):
    """
    Ensure that the unzipped submission path is a valid Zarr-2 file.

    Args:
        submission_path (str): The path to the unzipped submission Zarr-2 file.

    Raises:
        ValueError: If the submission is not a valid unzipped Zarr-2 file.
    """
    # See if a Zarr was incorrectly zipped inside other folder(s)
    # If so, move contents from .zarr folder to submission_path and warn user
    zarr_folders = list(submission_path.glob("**/*.zarr"))
    if len(zarr_folders) == 0:
        # Try forcing Zarr-2 format by adding .zgroup if missing
        try:
            ensure_zgroup(submission_path)
            logging.warning(
                f"Submission at {submission_path} did not contain a .zgroup file. Added one to force Zarr-2 format."
            )
        except Exception as e:
            raise ValueError(
                f"Submission at {submission_path} is not a valid unzipped Zarr-2 file."
            ) from e
    elif len(zarr_folders) == 1:
        zarr_folder = zarr_folders[0]
        logging.warning(
            f"Submission at {submission_path} contains a Zarr folder inside subfolder(s) at {zarr_folder}. Moving contents to the root submission folder."
        )
        # Move contents of zarr_folder to submission_path
        for item in zarr_folder.iterdir():
            target = submission_path / item.name
            if target.exists():
                if target.is_file():
                    target.unlink()
                else:
                    shutil.rmtree(target)
            shutil.move(str(item), str(submission_path))
        # Remove empty folders
        for parent in zarr_folder.parents:
            if parent == submission_path:
                break
            try:
                parent.rmdir()
            except OSError as e:
                logging.warning(
                    "Failed to remove directory %s while cleaning nested Zarr submission: %s",
                    parent,
                    e,
                )
        # Try opening again
        try:
            ensure_zgroup(submission_path)
            logging.warning(
                f"Submission at {submission_path} did not contain a .zgroup file. Added one to force Zarr-2 format."
            )
        except Exception as e:
            raise ValueError(
                f"Submission at {submission_path} is not a valid unzipped Zarr-2 file."
            ) from e
    elif len(zarr_folders) > 1:
        raise ValueError(
            f"Submission at {submission_path} contains multiple Zarr folders. Please ensure only one Zarr-2 file is submitted."
        )


def _prepare_submission(submission_path: UPath | str) -> UPath:
    """Unzip and validate submission.

    Args:
        submission_path: Path to zipped submission

    Returns:
        Path to unzipped, validated submission
    """
    unzipped_path = unzip_file(submission_path)
    ensure_valid_submission(UPath(unzipped_path))
    return UPath(unzipped_path)


def _discover_volumes(
    submission_path: UPath, truth_path: UPath
) -> tuple[list[str], list[str]]:
    """Discover volumes to score and missing volumes.

    Args:
        submission_path: Path to submission
        truth_path: Path to ground truth

    Returns:
        Tuple of (found_volumes, missing_volumes)

    Raises:
        ValueError: If no volumes found to score
    """
    pred_volumes = [d.name for d in submission_path.glob("*") if d.is_dir()]
    truth_volumes = [d.name for d in truth_path.glob("*") if d.is_dir()]

    found_volumes = list(set(pred_volumes) & set(truth_volumes))
    missing_volumes = list(set(truth_volumes) - set(pred_volumes))

    if len(found_volumes) == 0:
        # Check if "crop" prefixes are missing
        prefixed_pred_volumes = [f"crop{v}" for v in pred_volumes]
        found_volumes = list(set(prefixed_pred_volumes) & set(truth_volumes))

        if len(found_volumes) == 0:
            raise ValueError(
                "No volumes found to score. Make sure the submission is formatted correctly."
            )

        missing_volumes = list(set(truth_volumes) - set(prefixed_pred_volumes))

        # Rename predicted volumes to have "crop" prefix
        for v in pred_volumes:
            old_path = submission_path / v
            new_path = submission_path / f"crop{v}"
            try:
                old_path.rename(new_path)
            except Exception as exc:
                msg = (
                    f"Failed to rename predicted volume directory '{old_path}' to "
                    f"'{new_path}'. This may be due to missing files, insufficient "
                    "permissions, or an existing destination directory. Cannot "
                    "continue evaluation."
                )
                logging.error(msg)
                raise RuntimeError(msg) from exc

    return found_volumes, missing_volumes


def _estimate_max_task_memory_gb(
    evaluation_args: list[tuple],
    config: EvaluationConfig,
    per_instance_threads: int | None = None,
) -> tuple[float, int]:
    """Estimate peak per-worker memory from GT target shapes (metadata only).

    The estimate covers the two dominant contributors to worker RSS:
    (1) the fully-materialized truth + prediction arrays plus cc3d/bool
    intermediates that live for the duration of ``score_label``, and (2) the
    transient float64 distance-transform scratch allocated by the nested
    ``PER_INSTANCE_THREADS`` worker pool inside Hausdorff computation. The
    scratch dominates on instance tasks because each concurrent thread holds
    two ``float64`` arrays sized to the instance ROI (worst case: the full
    volume). Reads zarr shape/dtype without loading data.

    Returns:
        ``(peak_gb, n_voxels)`` for the most expensive task. ``n_voxels`` is
        the voxel count of the volume that pinned the peak, so callers can
        recompute the estimate if they throttle ``per_instance_threads``.
    """
    if per_instance_threads is None:
        per_instance_threads = config.per_instance_threads

    max_bytes = 0
    max_voxels = 0
    instance_classes = set(config.instance_classes)
    for args in evaluation_args:
        _, label_name, crop_name, truth_path, _ = args[:5]
        try:
            truth_arr = zarr.open(
                str(UPath(truth_path) / crop_name / label_name), mode="r"
            )
            n_voxels = int(np.prod(truth_arr.shape, dtype=np.int64))
            if label_name in instance_classes:
                # Base: truth + pred + cc3d relabel + binary intermediates ≈ 10 B/voxel.
                # Hausdorff scratch: up to per_instance_threads concurrent threads,
                # each allocating 2× float64 distance transforms sized to the ROI
                # (bounded by volume size).
                bytes_per_voxel = 10 + per_instance_threads * 2 * 8
            else:
                bytes_per_voxel = 5
            task_bytes = n_voxels * bytes_per_voxel
            if task_bytes > max_bytes:
                max_bytes = task_bytes
                max_voxels = n_voxels
        except Exception:
            continue
    return max_bytes / (1024**3), max_voxels


def _execute_parallel_scoring(
    evaluation_args: list[tuple],
    config: EvaluationConfig,
) -> list[tuple]:
    """Execute evaluations in parallel using process pools.

    Automatically scales worker count based on available system memory
    to prevent OOM errors.

    Args:
        evaluation_args: List of arguments for score_label
        config: Evaluation configuration

    Returns:
        List of (crop_name, label_name, result) tuples
    """
    effective_workers = config.max_workers
    effective_threads = config.per_instance_threads
    min_per_instance_threads = 4
    safety_factor = 1.5
    peak_gb = 0.0

    try:
        import psutil

        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024**3)
        reserve_gb = max(16.0, 0.05 * total_gb)
        budget_gb = max(total_gb - reserve_gb, 1.0)

        peak_gb, peak_voxels = _estimate_max_task_memory_gb(
            evaluation_args, config, per_instance_threads=effective_threads
        )
        if peak_gb > 0:
            safe_workers = max(
                1, int(budget_gb / (safety_factor * peak_gb))
            )
            if safe_workers < effective_workers:
                logging.warning(
                    f"Reducing workers from {effective_workers} to {safe_workers} "
                    f"based on total memory budget ({budget_gb:.0f}/{total_gb:.0f} GB) "
                    f"and estimated peak per worker ({peak_gb:.1f} GB, "
                    f"per_instance_threads={effective_threads})"
                )
                effective_workers = safe_workers

            # If a single task still exceeds per-worker share, throttle nested
            # Hausdorff threads so the whole fleet fits. Children read
            # PER_INSTANCE_THREADS from env at import time.
            per_worker_budget_gb = budget_gb / effective_workers
            if peak_voxels > 0 and peak_gb > per_worker_budget_gb:
                base_bytes_per_voxel = 10  # truth + pred + cc3d + bools
                target_gb = per_worker_budget_gb / safety_factor
                target_bytes_per_voxel = (target_gb * (1024**3)) / peak_voxels
                remaining = target_bytes_per_voxel - base_bytes_per_voxel
                if remaining > 0:
                    new_threads = int(remaining // 16)  # 2× float64 per thread
                else:
                    new_threads = min_per_instance_threads
                new_threads = max(
                    min_per_instance_threads,
                    min(effective_threads, new_threads),
                )
                if new_threads < effective_threads:
                    logging.warning(
                        f"Reducing per_instance_threads from {effective_threads} to "
                        f"{new_threads} to fit Hausdorff scratch in per-worker budget "
                        f"({per_worker_budget_gb:.1f} GB/worker)"
                    )
                    effective_threads = new_threads
                    # Recompute peak for logging; children pick this up via env.
                    peak_gb, _ = _estimate_max_task_memory_gb(
                        evaluation_args,
                        config,
                        per_instance_threads=effective_threads,
                    )
    except Exception as e:
        logging.warning(f"Could not estimate memory, using configured workers: {e}")

    # Propagate the (possibly throttled) per-instance thread count to worker
    # processes. distance.py reads PER_INSTANCE_THREADS at import time.
    os.environ["PER_INSTANCE_THREADS"] = str(effective_threads)

    logging.info(
        f"Scoring volumes in parallel, using {effective_workers} workers "
        f"(config.max_workers={config.max_workers}, "
        f"per_instance_threads={effective_threads}, peak~{peak_gb:.1f} GB/worker)..."
    )

    with ProcessPoolExecutor(effective_workers) as pool:
        futures = [pool.submit(score_label, *args) for args in evaluation_args]

        results = []
        for future in tqdm(
            as_completed(futures),
            desc="Scoring volumes",
            total=len(futures),
            dynamic_ncols=True,
            leave=True,
        ):
            results.append(future.result())

    return results


def _aggregate_and_save_results(
    results: list[tuple],
    missing_scores: dict,
    result_file: str | None,
    config: EvaluationConfig,
) -> dict:
    """Aggregate results and optionally save to file.

    Args:
        results: List of (crop_name, label_name, result) tuples
        missing_scores: Scores for missing volumes
        result_file: Path to save results (None to skip saving)
        config: Evaluation configuration

    Returns:
        Dictionary of aggregated scores
    """
    scores = missing_scores.copy()

    # Process all results and update incrementally
    all_scores, found_scores = update_scores(
        scores, results, result_file, instance_classes=config.instance_classes
    )

    logging.info("Scores combined across all test volumes:")
    logging.info(
        f"\tOverall Instance Score: {all_scores['overall_instance_score']:.4f}"
    )
    logging.info(
        f"\tOverall Semantic Score: {all_scores['overall_semantic_score']:.4f}"
    )
    logging.info(f"\tOverall Score: {all_scores['overall_score']:.4f}")

    logging.info("Scores combined across test volumes with data submitted:")
    logging.info(
        f"\tOverall Instance Score: {found_scores['overall_instance_score']:.4f}"
    )
    logging.info(
        f"\tOverall Semantic Score: {found_scores['overall_semantic_score']:.4f}"
    )
    logging.info(f"\tOverall Score: {found_scores['overall_score']:.4f}")

    return all_scores


def score_submission(
    submission_path=UPath(SUBMISSION_PATH).with_suffix(".zip").path,
    result_file=None,
    truth_path=TRUTH_PATH,
    instance_classes=INSTANCE_CLASSES,
    config: EvaluationConfig | None = None,
):
    """Score a submission against the ground truth data.

    This is the main entry point for evaluating a submission. It unzips,
    validates, scores, and aggregates results for all volumes.

    Args:
        submission_path: Path to the zipped submission Zarr-2 file
        result_file: Path to save the scores (None to skip saving)
        truth_path: Path to the ground truth Zarr-2 file
        instance_classes: List of instance segmentation classes
        config: Evaluation configuration (uses defaults if None)

    Returns:
        Dictionary of aggregated scores across all volumes

    Raises:
        ValueError: If submission format is invalid
        RuntimeError: If volume renaming fails

    Example:
        >>> scores = score_submission('submission.zip', 'results.json')
        >>> print(f"Overall score: {scores['overall_score']:.4f}")

    Results structure:
        {
            "cropN": {  # Per-volume scores
                "label_name": {
                    # Instance segmentation
                    "f1": float,
                    "hausdorff_distance": float,
                    "combined_score": float,
                    # OR semantic segmentation
                    "iou": float,
                    "dice_score": float,
                }
            },
            "label_scores": {  # Aggregated per-label
                "label_name": {...}
            },
            "overall_instance_score": float,
            "overall_semantic_score": float,
            "overall_score": float,
        }
    """
    if config is None:
        config = EvaluationConfig.from_env()
        config.validate()

    # Override config with explicit parameters if provided
    if truth_path != TRUTH_PATH:
        config.truth_path = UPath(truth_path)
    if instance_classes != INSTANCE_CLASSES:
        config.instance_classes = list(instance_classes)

    logging.info(f"Scoring {submission_path}...")
    start_time = time()

    # Step 1: Prepare submission
    submission_path = _prepare_submission(submission_path)

    # Step 2: Discover volumes
    logging.info(f"Discovering volumes in {submission_path}...")
    found_volumes, missing_volumes = _discover_volumes(
        submission_path, config.truth_path
    )

    logging.info(f"Scoring volumes: {found_volumes}")
    if len(missing_volumes) > 0:
        logging.info(f"Missing volumes: {missing_volumes}")
        logging.info("Scoring missing volumes as 0's")

    # Step 3: Score missing volumes
    scores = {
        volume: missing_volume_score(
            config.truth_path, volume, instance_classes=config.instance_classes
        )
        for volume in missing_volumes
    }

    # Step 4: Get evaluation arguments
    evaluation_args = get_evaluation_args(
        found_volumes,
        submission_path=submission_path,
        truth_path=config.truth_path,
        instance_classes=config.instance_classes,
    )

    # Step 5: Execute parallel scoring
    results = _execute_parallel_scoring(evaluation_args, config)

    # Step 6: Aggregate and save results
    all_scores = _aggregate_and_save_results(results, scores, result_file, config)

    logging.info(f"Submission scored in {time() - start_time:.2f} seconds")

    if result_file is None:
        logging.info("Final combined scores:")
        logging.info(all_scores)

    return all_scores
