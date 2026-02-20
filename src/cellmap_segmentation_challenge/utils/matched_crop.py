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

# Memory estimation constants
BYTES_PER_FLOAT32 = 4  # 4 bytes per float32 voxel

# Maximum allowed size ratio between source and target arrays
# Can be set via environment variable for flexibility
# Default is 16x in each dimension (4096x total volume size ratio)
# With chunked loading, we can handle larger arrays without loading all into memory at once
MAX_VOLUME_SIZE_RATIO = float(os.environ.get("MAX_VOLUME_SIZE_RATIO", 16**3))


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

        # Estimate memory usage (assuming float32)
        estimated_memory_mb = (src_size * BYTES_PER_FLOAT32) / (1024 * 1024)

        # Return whether we should use chunked loading
        return ratio, estimated_memory_mb

    def _should_use_chunked_loading(self, ratio: float, estimated_memory_mb: float) -> bool:
        """
        Determine if we should use chunked loading based on size ratio.
        Chunked loading is used when the array is large but within acceptable limits.
        """
        # Use chunked loading if ratio is high but not exceeding the limit
        # This allows processing of larger arrays without exceeding memory
        if ratio > MAX_VOLUME_SIZE_RATIO:
            raise ValueError(
                f"Source array at {self.path} is too large compared to target shape: "
                f"ratio {ratio:.1f}x > {MAX_VOLUME_SIZE_RATIO}x limit. "
                f"Estimated memory: {estimated_memory_mb:.1f} MB. "
                f"Please downsample your predictions to a resolution closer to the target."
            )

        # Use chunked loading for arrays that would use significant memory
        return estimated_memory_mb > 500

    def _load_array_chunked(self, arr: zarr.Array, scale_factors: Tuple[float, ...]) -> np.ndarray:
        """
        Load and downsample a zarr array in chunks to reduce memory usage.
        
        Args:
            arr: The zarr array to load
            scale_factors: Scale factors for rescaling (in_vs / tgt_vs).
                When input has finer resolution than target (in_vs < tgt_vs),
                scale_factors < 1.0, causing rescale() to downsample.
                Example: in_vs=2nm, tgt_vs=8nm → scale_factors=0.25 → output is 0.25x input size
        
        Returns:
            The downsampled array as a numpy array
        """
        logger.info(f"Loading and downsampling array in chunks with scale factors: {scale_factors}")
        
        # Calculate output shape after downsampling
        # scale_factors < 1.0 means downsampling, so output is smaller
        output_shape = tuple(int(np.ceil(s * sf)) for s, sf in zip(arr.shape, scale_factors))
        output = np.zeros(output_shape, dtype=arr.dtype if self._is_instance() else np.float32)
        
        # Determine chunk size based on memory constraints
        # Target ~100MB per chunk in memory
        target_chunk_memory_mb = 100
        chunk_voxels = int((target_chunk_memory_mb * 1024 * 1024) / BYTES_PER_FLOAT32)
        chunk_size_per_dim = max(32, int(chunk_voxels ** (1/3)))  # At least 32 voxels per dimension
        
        logger.info(f"Processing with chunk size: {chunk_size_per_dim} voxels per dimension")
        
        # Process array in chunks
        for z_start in range(0, arr.shape[0], chunk_size_per_dim):
            z_end = min(z_start + chunk_size_per_dim, arr.shape[0])
            for y_start in range(0, arr.shape[1], chunk_size_per_dim):
                y_end = min(y_start + chunk_size_per_dim, arr.shape[1])
                for x_start in range(0, arr.shape[2], chunk_size_per_dim):
                    x_end = min(x_start + chunk_size_per_dim, arr.shape[2])
                    
                    # Load chunk
                    chunk = arr[z_start:z_end, y_start:y_end, x_start:x_end]
                    
                    # Downsample chunk
                    if self._is_instance():
                        chunk_downsampled = rescale(
                            chunk,
                            scale_factors,
                            order=0,
                            mode="constant",
                            preserve_range=True,
                        ).astype(chunk.dtype)
                    else:
                        if chunk.dtype == bool:
                            chunk = chunk.astype(np.float32)
                        chunk_downsampled = rescale(
                            chunk,
                            scale_factors,
                            order=1,
                            mode="constant",
                            preserve_range=True,
                        )
                        # Don't threshold here, will be done at the end
                    
                    # Calculate output position
                    # scale_factors represents the ratio of dimensions (output_size / input_size)
                    # When downsampling (in_vs < tgt_vs), scale_factors < 1.0
                    out_z_start = int(z_start * scale_factors[0])
                    out_z_end = min(int(np.ceil(z_end * scale_factors[0])), output_shape[0])
                    out_y_start = int(y_start * scale_factors[1])
                    out_y_end = min(int(np.ceil(y_end * scale_factors[1])), output_shape[1])
                    out_x_start = int(x_start * scale_factors[2])
                    out_x_end = min(int(np.ceil(x_end * scale_factors[2])), output_shape[2])
                    
                    # Get actual downsampled chunk shape
                    chunk_out_shape = (
                        out_z_end - out_z_start,
                        out_y_end - out_y_start,
                        out_x_end - out_x_start
                    )
                    
                    # Ensure chunk_downsampled matches expected output region
                    if chunk_downsampled.shape != chunk_out_shape:
                        # Crop or pad to match
                        chunk_downsampled = chunk_downsampled[:chunk_out_shape[0], :chunk_out_shape[1], :chunk_out_shape[2]]
                    
                    # Place downsampled chunk in output
                    output[out_z_start:out_z_end, out_y_start:out_y_end, out_x_start:out_x_end] = chunk_downsampled
        
        # Convert back to bool if semantic (threshold once at the end)
        if not self._is_instance():
            output = output > self.semantic_threshold
        
        return output

    def _load_source_array(self):
        """
        Returns (image, input_voxel_size, input_translation, already_downsampled)
        where image is a numpy array and already_downsampled indicates if chunked downsampling was applied.
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
                ratio, estimated_memory_mb = self._check_size_ratio(arr.shape)
                use_chunked = self._should_use_chunked_loading(ratio, estimated_memory_mb)
                
                if use_chunked:
                    logger.warning(
                        f"Large OME-NGFF array detected ({estimated_memory_mb:.1f} MB). "
                        f"Loading entire array - CellMapImage should have already selected appropriate resolution level. "
                        f"If memory issues occur, ensure predictions are saved at appropriate resolution."
                    )
                
                image = arr[:]
                return image, input_voxel_size, input_translation, False  # Not downsampled in chunks
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
                ratio, estimated_memory_mb = self._check_size_ratio(arr.shape)
                use_chunked = self._should_use_chunked_loading(ratio, estimated_memory_mb)
                
                if use_chunked and vs is not None:
                    logger.info(f"Using chunked loading for large non-OME array ({estimated_memory_mb:.1f} MB)")
                    # Calculate scale factors for downsampling
                    in_vs = np.asarray(vs, dtype=float)
                    tgt_vs = np.asarray(self.target_voxel_size, dtype=float)
                    if in_vs.size != tgt_vs.size:
                        in_vs = in_vs[-tgt_vs.size:]
                    
                    if not np.allclose(in_vs, tgt_vs):
                        # Downsampling needed - use chunked approach
                        # scale_factors = in_vs / tgt_vs
                        # When in_vs < tgt_vs (fine→coarse), scale_factors < 1.0, causing rescale to downsample
                        scale_factors = in_vs / tgt_vs
                        image = self._load_array_chunked(arr, scale_factors)
                        return image, self.target_voxel_size, tr, True  # Already downsampled
                    else:
                        # No downsampling needed, load normally
                        image = arr[:]
                        return image, vs, tr, False
                else:
                    image = arr[:]
                    return image, vs, tr, False
            except Exception as e:
                raise ValueError(
                    f"Failed to load from non-OME zarr Group at {self.path}. "
                    f"Expected to find resolution levels (e.g., 's0', 's1') with voxel_size metadata, "
                    f"but encountered an error: {e}"
                )

        # Single-scale zarr array
        if isinstance(ds, zarr.Array):
            logger.info(f"Detected single-scale zarr Array at {self.path}")
            ratio, estimated_memory_mb = self._check_size_ratio(ds.shape)
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
            
            use_chunked = self._should_use_chunked_loading(ratio, estimated_memory_mb)
            
            if use_chunked and vs is not None:
                logger.info(f"Using chunked loading for large single-scale array ({estimated_memory_mb:.1f} MB)")
                # Calculate scale factors for downsampling
                in_vs = np.asarray(vs, dtype=float)
                tgt_vs = np.asarray(self.target_voxel_size, dtype=float)
                if in_vs.size != tgt_vs.size:
                    in_vs = in_vs[-tgt_vs.size:]
                
                if not np.allclose(in_vs, tgt_vs):
                    # Downsampling needed - use chunked approach
                    # scale_factors = in_vs / tgt_vs
                    # When in_vs < tgt_vs (fine→coarse), scale_factors < 1.0, causing rescale to downsample
                    scale_factors = in_vs / tgt_vs
                    image = self._load_array_chunked(ds, scale_factors)
                    return image, self.target_voxel_size, tr, True  # Already downsampled
                else:
                    # No downsampling needed, load normally
                    image = ds[:]
                    return image, vs, tr, False
            else:
                image = ds[:]
                return image, vs, tr, False

        raise ValueError(f"Unsupported zarr node type at {self.path}: {type(ds)}")

    def load_aligned(self) -> np.ndarray:
        """
        Return full aligned volume in target space (target_shape).
        """
        tgt_vs = np.asarray(self.target_voxel_size, dtype=float)
        tgt_shape = tuple(int(x) for x in self.target_shape)
        tgt_tr = np.asarray(self.target_translation, dtype=float)

        image, input_voxel_size, input_translation, already_downsampled = self._load_source_array()

        # Resample if needed (skip if already downsampled in chunks)
        if not already_downsampled and input_voxel_size is not None:
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

        elif not already_downsampled and image.shape != tgt_shape:
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
