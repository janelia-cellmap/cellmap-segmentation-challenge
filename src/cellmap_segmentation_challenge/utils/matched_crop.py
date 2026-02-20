import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import os

import numpy as np
import zarr
from skimage.transform import rescale
from upath import UPath

from cellmap_data import CellMapImage


logger = logging.getLogger(__name__)

# Maximum allowed size ratio between source and target arrays
# Can be set via environment variable for flexibility
# Default is 4x in each dimension (64x total volume size ratio)
# This limits memory usage to reasonable levels (e.g., 256MB for typical crops)
MAX_VOLUME_SIZE_RATIO = float(os.environ.get("MAX_VOLUME_SIZE_RATIO", 4**3))


def _get_attr_any(attrs, keys):
    for k in keys:
        if k in attrs:
            return attrs[k]
    return None


def _parse_voxel_size(attrs) -> Optional[Tuple[float, ...]]:
    vs = _get_attr_any(attrs, ["voxel_size", "resolution", "scale"])
    if vs is None:
        return None
    return tuple(float(x) for x in vs)


def _parse_translation(attrs) -> Optional[Tuple[float, ...]]:
    tr = _get_attr_any(attrs, ["translation", "offset"])
    if tr is None:
        return None
    return tuple(float(x) for x in tr)


def _resize_pad_crop(
    image: np.ndarray, target_shape: Tuple[int, ...], pad_value=0
) -> np.ndarray:
    # center pad/crop like your resize_array
    arr_shape = image.shape
    resized = image

    pad_width = []
    for i in range(len(target_shape)):
        if arr_shape[i] < target_shape[i]:
            pad_before = (target_shape[i] - arr_shape[i]) // 2
            pad_after = target_shape[i] - arr_shape[i] - pad_before
            pad_width.append((pad_before, pad_after))
        else:
            pad_width.append((0, 0))

    if any(p > 0 for pads in pad_width for p in pads):
        resized = np.pad(resized, pad_width, mode="constant", constant_values=pad_value)

    slices = []
    for i in range(len(target_shape)):
        if resized.shape[i] > target_shape[i]:
            start = (resized.shape[i] - target_shape[i]) // 2
            end = start + target_shape[i]
            slices.append(slice(start, end))
        else:
            slices.append(slice(None))
    return resized[tuple(slices)]


@dataclass
class MatchedCrop:
    path: str | UPath
    class_label: str
    target_voxel_size: Sequence[float]
    target_shape: Sequence[int]
    target_translation: Sequence[float]
    instance_classes: Sequence[str]
    semantic_threshold: float = 0.5
    pad_value: float | int = 0

    def _is_instance(self) -> bool:
        return self.class_label in set(self.instance_classes)

    def _select_non_ome_level(
        self, grp: zarr.Group
    ) -> Tuple[str, Optional[Tuple[float, ...]], Optional[Tuple[float, ...]]]:
        """
        Heuristic: choose among arrays like s0/s1/... (or any array keys) by voxel_size attrs.
        Preference: pick the level whose voxel_size is <= target_voxel_size (finer or equal to target resolution)
        and closest to target; else closest overall.
        Returns (array_key, voxel_size, translation)
        """
        keys = list(grp.array_keys())
        if not keys:
            raise ValueError(f"No arrays found under {self.path}")

        tgt = np.asarray(self.target_voxel_size, dtype=float)

        best_key = None
        best_vs = None
        best_tr = None
        best_score = None

        for k in keys:
            arr = grp[k]
            vs = _parse_voxel_size(arr.attrs)
            tr = _parse_translation(arr.attrs)

            if vs is None:
                # treat as unknown; deprioritize
                score = 1e18
            else:
                v = np.asarray(vs, dtype=float)
                if v.size != tgt.size:
                    v = v[-tgt.size :]
                # prefer v <= tgt
                # Here: voxel_size larger => coarser. We want not coarser than target => v <= tgt
                # If your convention is reversed, adjust this rule.
                not_coarser = np.all(v <= tgt)
                dist = float(np.linalg.norm(v - tgt))
                score = dist + (0.0 if not_coarser else 1e6)

            if best_score is None or score < best_score:
                best_score = score
                best_key = k
                best_vs = vs
                best_tr = tr

        assert best_key is not None
        return best_key, best_vs, best_tr

    def _check_size_ratio(self, source_shape: Tuple[int, ...]):
        tgt_size = np.prod(self.target_shape)
        if tgt_size == 0:
            raise ValueError(
                f"Invalid target shape {tuple(self.target_shape)}: product of dimensions is zero."
            )
        src_size = np.prod(source_shape)
        ratio = src_size / tgt_size
        
        # Estimate memory usage (assuming float32, 4 bytes per voxel)
        estimated_memory_mb = (src_size * 4) / (1024 * 1024)
        
        if ratio > MAX_VOLUME_SIZE_RATIO:
            raise ValueError(
                f"Source array at {self.path} is too large compared to target shape: "
                f"source size {src_size} voxels, target size {tgt_size} voxels, "
                f"ratio {ratio:.1f}x > {MAX_VOLUME_SIZE_RATIO}x limit. "
                f"Estimated memory: {estimated_memory_mb:.1f} MB. "
                f"This will cause memory allocation issues. "
                f"Please downsample your predictions to a resolution closer to the target."
            )
        
        # Warn if loading will use significant memory
        if estimated_memory_mb > 500:
            logger.warning(
                f"Loading large array from {self.path}: {source_shape} = {src_size} voxels. "
                f"Estimated memory: {estimated_memory_mb:.1f} MB. "
                f"Consider downsampling your predictions to reduce memory usage."
            )

    def _load_source_array(self):
        """
        Returns (image, input_voxel_size, input_translation)
        where image is a numpy array.
        """
        try:
            ds = zarr.open(str(self.path), mode="r")
        except Exception as e:
            raise ValueError(
                f"Failed to open zarr at {self.path}. "
                f"Ensure the path points to a valid zarr array or group. "
                f"Error: {e}"
            )

        logger.info(f"Loading from {self.path}, type: {type(ds).__name__}")

        # OME-NGFF multiscale
        if isinstance(ds, zarr.Group) and "multiscales" in ds.attrs:
            logger.info(f"Detected OME-NGFF multiscale format at {self.path}")
            try:
                img = CellMapImage(
                    path=str(self.path),
                    target_class=self.class_label,
                    target_scale=self.target_voxel_size,
                    target_voxel_shape=self.target_shape,
                    pad=True,
                    pad_value=self.pad_value,
                    interpolation="nearest" if self._is_instance() else "linear",
                )
                level = img.scale_level
                level_path = UPath(self.path) / level

                # Extract input voxel size and translation from multiscales metadata
                input_voxel_size = None
                input_translation = None
                for d in ds.attrs["multiscales"][0]["datasets"]:
                    if d["path"] == level:
                        for t in d.get("coordinateTransformations", []):
                            if t.get("type") == "scale":
                                input_voxel_size = tuple(t["scale"])
                            elif t.get("type") == "translation":
                                input_translation = tuple(t["translation"])
                        break

                arr = zarr.open(str(level_path), mode="r")
                self._check_size_ratio(arr.shape)
                image = arr[:]
                return image, input_voxel_size, input_translation
            except Exception as e:
                raise ValueError(
                    f"Failed to load OME-NGFF multiscale data from {self.path}. "
                    f"Error: {e}"
                )

        # Non-OME group multiscale OR single-scale array with attrs
        if isinstance(ds, zarr.Group):
            logger.info(f"Detected zarr Group (non-OME) at {self.path}")
            # If this group directly contains the label array (common): path points at an array node
            # zarr.open on an array path usually returns an Array, not Group. If we got Group, pick a level.
            try:
                key, vs, tr = self._select_non_ome_level(ds)
                arr = ds[key]
                self._check_size_ratio(arr.shape)
                image = arr[:]
                return image, vs, tr
            except Exception as e:
                raise ValueError(
                    f"Failed to load from non-OME zarr Group at {self.path}. "
                    f"Expected to find resolution levels (e.g., 's0', 's1') with voxel_size metadata, "
                    f"but encountered an error: {e}"
                )

        # Single-scale zarr array
        if isinstance(ds, zarr.Array):
            logger.info(f"Detected single-scale zarr Array at {self.path}")
            self._check_size_ratio(ds.shape)
            image = ds[:]
            vs = _parse_voxel_size(ds.attrs)
            tr = _parse_translation(ds.attrs)
            if vs is None:
                logger.warning(
                    f"No voxel_size metadata found at {self.path}. "
                    f"Will attempt to match by shape only, which may produce incorrect alignment."
                )
            if tr is None:
                logger.warning(
                    f"No translation metadata found at {self.path}. "
                    f"Assuming zero offset, which may produce incorrect alignment."
                )
            return image, vs, tr

        raise ValueError(f"Unsupported zarr node type at {self.path}: {type(ds)}")

    def load_aligned(self) -> np.ndarray:
        """
        Return full aligned volume in target space (target_shape).
        """
        tgt_vs = np.asarray(self.target_voxel_size, dtype=float)
        tgt_shape = tuple(int(x) for x in self.target_shape)
        tgt_tr = np.asarray(self.target_translation, dtype=float)

        image, input_voxel_size, input_translation = self._load_source_array()

        # Resample if needed
        if input_voxel_size is not None:
            in_vs = np.asarray(input_voxel_size, dtype=float)
            if in_vs.size != tgt_vs.size:
                in_vs = in_vs[-tgt_vs.size :]

            if not np.allclose(in_vs, tgt_vs):
                scale_factors = in_vs / tgt_vs
                if self._is_instance():
                    image = rescale(
                        image,
                        scale_factors,
                        order=0,
                        mode="constant",
                        preserve_range=True,
                    ).astype(image.dtype)
                else:
                    if image.dtype == bool:
                        image = image.astype(np.float32)
                    imgf = rescale(
                        image,
                        scale_factors,
                        order=1,
                        mode="constant",
                        preserve_range=True,
                    )
                    image = imgf > self.semantic_threshold

        elif image.shape != tgt_shape:
            # If no voxel size info, fall back to center crop/pad
            image = _resize_pad_crop(image, tgt_shape, pad_value=self.pad_value)

        # Compute relative offset in voxel units
        if input_translation is not None:
            in_tr = np.asarray(input_translation, dtype=float)
            if in_tr.size != tgt_tr.size:
                in_tr = in_tr[-tgt_tr.size :]

            # snap to voxel grid
            adjusted_in_tr = (in_tr // tgt_vs) * tgt_vs
            rel = (np.abs(adjusted_in_tr - tgt_tr) // tgt_vs) * np.sign(
                adjusted_in_tr - tgt_tr
            )
            rel = rel.astype(int)
        else:
            rel = np.zeros(len(tgt_shape), dtype=int)

        # Translate + crop/pad into destination
        if any(rel != 0) or image.shape != tgt_shape:
            result = np.zeros(tgt_shape, dtype=image.dtype)
            input_slices = []
            output_slices = []
            for d in range(len(tgt_shape)):
                if rel[d] < 0:
                    input_start = abs(rel[d])
                    output_start = 0
                    input_end = min(input_start + tgt_shape[d], image.shape[d])
                    length = input_end - input_start
                    output_end = output_start + length
                else:
                    input_start = 0
                    output_start = rel[d]
                    output_end = min(tgt_shape[d], image.shape[d] + output_start)
                    length = output_end - output_start
                    input_end = input_start + length

                if length <= 0:
                    return result

                input_slices.append(slice(int(input_start), int(input_end)))
                output_slices.append(slice(int(output_start), int(output_end)))

            result[tuple(output_slices)] = image[tuple(input_slices)]
            return result

        return image
