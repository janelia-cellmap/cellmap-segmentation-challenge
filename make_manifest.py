from cellmap_segmentation_challenge.fetch_data import _resolve_gt_source_url, _resolve_em_source_url, get_url, resolve_em_url
from cellmap_segmentation_challenge.utils.crops import CHALLENGE_CROPS, DEFAULT_CROP_URL, DEFAULT_EM_URL
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# get constants from environment, falling back to defaults as needed
crop_url = os.environ.get('CSC_FETCH_DATA_CROP_URL', DEFAULT_CROP_URL)
em_url = os.environ.get('CSC_FETCH_DATA_EM_URL', DEFAULT_EM_URL)

out_fname = 'manifest.csv'
lines = ("crop_name, crop_url, em_url\n",)
for crop in CHALLENGE_CROPS:
    crop_id = crop.id
    gt_url = _resolve_gt_source_url(crop_url, crop)
    em_root, em_paths = _resolve_em_source_url(em_url, crop)
    em_group = resolve_em_url(em_root, em_paths)

    lines += (f'{crop.id},{gt_url!s},{get_url(em_group)!s}\n',)

with open(out_fname, 'w') as out_fo:
    out_fo.writelines(lines)

