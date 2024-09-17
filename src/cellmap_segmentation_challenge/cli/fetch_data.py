from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
import click
from xarray import DataArray
from xarray_ome_ngff import read_multiscale_group
from xarray_ome_ngff.v04.multiscale import transforms_from_coords
from cellmap_segmentation_challenge.fetch_data import _resolve_em_source_url, _resolve_gt_source_url,_resolve_gt_dest_url, subset_to_slice, partition_copy_store, read_group, get_chunk_keys
from ..utils.crops import CHALLENGE_CROPS, Crop
import structlog
from yarl import URL
import zarr
from pathlib import Path
import numpy as np
from zarr.storage import FSStore
from pydantic_zarr.v2 import GroupSpec

# SOURCE_URL = URL('s3://janelia-cosem-datasets')
CROP_SOURCE_URL = URL('file:///nrs/cellmap/bennettd/data/crop_tests')
EM_SOURCE_URL = URL('s3://janelia-cosem-datasets')
NUM_WORKERS=32


@click.command
@click.option('--crops', type=click.STRING, required=True, default='all')
@click.option('--raw-padding', type=click.INT, default=0)
@click.option('--dest-dir', type=click.STRING, required=True)
@click.option("--access-mode", type=click.STRING, default="append")
@click.option("--fetch-all-em-resolutions", type=click.BOOL, is_flag=True, default=False)
def fetch_crops_cli(crops: str, raw_padding: str, dest_dir: str, access_mode: str, fetch_all_em_resolutions):

    if access_mode == 'overwrite':
        mode= 'w'
    elif access_mode == 'create':
        mode='w-'
    elif access_mode == 'append':
        mode = 'a'

    fetch_save_start = time.time()
    pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
    dest_path_abs = Path(dest_dir).absolute()

    log = structlog.get_logger()
    crops_parsed: tuple[Crop, ...]
    if crops == 'all':
        crops_parsed = CHALLENGE_CROPS
    else:
        crops_split = tuple(int(x) for x in crops.split(','))
        crops_parsed = tuple(filter(lambda v: v.id in crops_split, CHALLENGE_CROPS))
    crop_ids = tuple(c.id for c in crops_parsed)
    if len(crops_parsed) == 0:
        log.info(f'No crops found matching {crops}. Doing nothing.')
        return
    
    log.info(f'Preparing to copy the following crops: {crop_ids}')
    log.info(f'Data will be saved to {dest_path_abs}')
    
    for crop in crops_parsed:
        log = log.bind(crop=crop)
        gt_save_start = time.time()
        gt_source_url = _resolve_gt_source_url(CROP_SOURCE_URL, crop)
        try:
            gt_source_group = read_group(str(gt_source_url), mode='r')
            log.info(f'Found a Zarr group at {gt_source_url}.')
        except zarr.errors.GroupNotFoundError:
            log.info(f'No Zarr group was found at {gt_source_url}. This crop will be skipped.')
            continue

        # gt_dest_url = URL.build(scheme='file', path=str(dest_path_abs))        
        gt_dest_url = _resolve_gt_dest_url(URL.build(scheme='file', path=str(dest_path_abs)), crop)
        fs = gt_source_group.store.fs
        
        store_path = gt_source_group.store.path
        # Using fs.find is a performance hack until we fix the slowness of traversing the zarr hierarchy to 
        # build the list of files
        gt_files = fs.find(gt_source_group.store.path)
        crop_group_inventory = tuple(fn.split(store_path)[-1] for fn in gt_files)
        log.info(f'Preparing to fetch {len(crop_group_inventory)} files from {gt_source_url}.')
        
        dest_crop_group = zarr.open_group(str(gt_dest_url), mode=mode)
        
        partition_copy_store(
            keys=crop_group_inventory, 
            source_store=gt_source_group.store, 
            dest_store=dest_crop_group.store, 
            batch_size=256, 
            pool=pool,
            log=log)
                    
        log.info(f'Finished saving crop to local directory after {time.time() - gt_save_start:0.3f}s')

        em_dest_urls = _resolve_em_source_url(EM_SOURCE_URL, crop)
        em_dest_url : URL | None = None
        em_source_group: None | zarr.Group = None

        # todo: functionalize this
        for maybe_em_source_url in em_dest_urls:
            log.info(f'Checking for EM data at {maybe_em_source_url}')
            try:
                em_source_group = read_group(str(maybe_em_source_url))
                em_source_url = maybe_em_source_url
                em_dest_url = URL.build(scheme='file', path=str(dest_path_abs)).joinpath(maybe_em_source_url.path.lstrip('/'))
                log.info(f'Found EM data at {em_source_url}')
                break
            except zarr.errors.GroupNotFoundError:
                log.info(f'No EM data found at {maybe_em_source_url}')

        if em_source_group is None:
            log.info(f'No EM data found at any of the possible URLs. No EM data will be fetched for this crop.')        
            continue 

        if em_source_group is not None:         
            # model the em group locally
            em_dest_group = GroupSpec.from_zarr(em_source_group).to_zarr(FSStore(str(em_dest_url)), path='', overwrite=(mode =='w'))

            # get the multiscale model of the source em group           
            array_wrapper = {'name': 'dask_array', 'config': {'chunks': 'auto'}}
            em_source_arrays = read_multiscale_group(em_source_group, array_wrapper=array_wrapper)
            padding = raw_padding

            # get the overlapping region between the crop and the full array, in array coordinates.
            crop_multiscale_group: dict[str, DataArray] | None = None
            for _, group in gt_source_group.groups():
                try:
                    crop_multiscale_group = read_multiscale_group(group, array_wrapper=array_wrapper)
                    break
                except (ValueError, TypeError):
                    continue

            if crop_multiscale_group is None:
                log.info(f'No multiscale groups found in {gt_source_url}. No EM data can be fetched.')
            else:
                em_group_inventory = ()
                em_source_arrays_sorted = sorted(em_source_arrays.items(), key=lambda kv: np.prod(kv[1].shape), reverse=True)
                gt_source_arrays_sorted = sorted(crop_multiscale_group.items(), key=lambda kv: np.prod(kv[1].shape), reverse=True)
                
                # apply padding in a resolution-aware way
                _, (base_em_scale, _) = transforms_from_coords(em_source_arrays_sorted[0][1].coords, transform_precision=4)
                _, (base_gt_scale, _) = transforms_from_coords(gt_source_arrays_sorted[0][1].coords, transform_precision=4)

                for key, array in em_source_arrays_sorted:
                    _, (current_scale, _) = transforms_from_coords(array.coords, transform_precision=4)
                    if fetch_all_em_resolutions: ratio_threshold = 0
                    else: ratio_threshold = 0.9
                    scale_ratios = tuple(s_current / s_gt for s_current, s_gt in zip(current_scale.scale, base_gt_scale.scale))
                    if all(tuple(x > ratio_threshold for x in scale_ratios)):
                        
                        relative_scale = base_em_scale.scale[0] / current_scale.scale[0]
                        current_pad = int(padding * relative_scale)
                        slices = subset_to_slice(array, crop_multiscale_group['s0'])
                        slices_padded = tuple(slice(max(sl.start - current_pad, 0), min(sl.stop + current_pad, shape), sl.step) for sl, shape in zip(slices, array.shape))
                        new_chunks = tuple(map(lambda v: '/'.join([key, v]), get_chunk_keys(em_source_group[key], slices_padded)))
                        log.info(f'Gathering {len(new_chunks)} chunks from level {key}.')
                        em_group_inventory += new_chunks
                    else:
                        log.info(f'Skipping scale level {key} because it is sampled more densely than the groundtruth data')

                log.info(f'Preparing to fetch {len(em_group_inventory)} files from {em_source_url}.')
                partition_copy_store(
                    keys=em_group_inventory, 
                    source_store=em_source_group.store, 
                    dest_store=em_dest_group.store, 
                    batch_size=256, 
                    pool=pool,
                    log=log)

    log = log.unbind('crop')
    log.info(f'Done after {time.time() - fetch_save_start:0.3f}s')