from concurrent.futures import ThreadPoolExecutor
import time
import click

from cellmap_segmentation_challenge.fetch_data import _resolve_em, _resolve_gt, get_group_objects, partition_copy_store, read_group
from ..utils.crops import CHALLENGE_CROPS, Crop
import structlog
from yarl import URL
import zarr
from pathlib import Path

SOURCE_URL = URL('s3://janelia-cosem-datasets')
NUM_WORKERS=32

@click.command
@click.option('--crops', type=click.STRING, required=True, default='all')
@click.option('--raw-padding', type=click.STRING)
@click.option('--dest-dir', type=click.STRING, required=True)
@click.option("--access-mode", type=click.STRING, default="append")
def fetch_crops_cli(crops: str, raw_padding: str, dest_dir: str, access_mode: str):
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
        gt_source_url = _resolve_gt(SOURCE_URL, crop)
        try:
            gt_source_group = read_group(str(gt_source_url), mode='r')
            log.info(f'Found a Zarr group at {gt_source_url}.')
        except zarr.errors.GroupNotFoundError:
            log.info(f'No Zarr group was found at {gt_source_url}. This crop will be skipped.')
            continue
        gt_dest_url = gt_source_url.with_host(str(dest_path_abs)).with_scheme('file')        
        fs = gt_source_group.store.fs
        
        store_path = gt_source_group.store.path
        crop_group_inventory = tuple(fn.split(store_path)[-1] for fn in fs.find(gt_source_group.store.path))
        log.info(f'Preparing to fetch {len(crop_group_inventory)} files from {gt_source_url}.')
        
        if access_mode == 'overwrite':
            mode= 'w'
        elif access_mode == 'create':
            mode='w-'
        elif access_mode == 'append':
            mode = 'a'

        dest_crop_group = zarr.open_group(str(gt_dest_url), mode=mode)
        
        partition_copy_store(
            keys=crop_group_inventory, 
            source_store=gt_source_group.store, 
            dest_store=dest_crop_group.store, batch_size=256, 
            pool=pool)            
        log.info(f'Finished saving crop to local directory after {time.time() - gt_save_start:0.3f}s')

        maybe_em_groups = _resolve_em(SOURCE_URL, crop)
        for _maybe_em_group in maybe_em_groups:
            try:
                read_group(str(_maybe_em_group))
            except zarr.errors.GroupNotFoundError:
                continue
            finally:
                log.info() 

        log.info(f'No EM data found.')

    log = log.unbind('crop')
    log.info(f'Done after {time.time() - fetch_save_start:0.3f}s')