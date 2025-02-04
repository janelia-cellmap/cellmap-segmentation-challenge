#!/bin/bash

# Determine fetch options and zip name
# Read CLI arguments
RES_MODE=$1  # "matched" or "all"
PAD=$2       # "0" for no padding or e.g. "128" for 128-voxel padding

if [ "$RES_MODE" == "matched" ]; then
    FETCH_ARG=""
    ZIP_PREFIX="matched_res"
elif [ "$RES_MODE" == "all" ]; then
    FETCH_ARG="-all-res"
    ZIP_PREFIX="all_res"
else
    echo "Invalid resolution mode. Use 'matched' or 'all'."
    exit 1
fi

if [ "$PAD" -eq 0 ]; then
    PAD_ARG=""
    ZIP_SUFFIX="no_pad"
else
    PAD_ARG="-p $PAD"
    ZIP_SUFFIX="${PAD}_pad"
fi

ZIP_NAME="${ZIP_PREFIX}_${ZIP_SUFFIX}.zip"

# Create a new data folder
DATA_DIR="data_${ZIP_PREFIX}_${ZIP_SUFFIX}"
mkdir -p "$DATA_DIR"

# Remove any existing data in the folder
rm -rf "$DATA_DIR/*"
rm -f "$ZIP_NAME" &

# Fetch, zip, and clean up
echo "Zipping ${ZIP_NAME}..."
cd "$DATA_DIR"
csc fetch-data -d . $FETCH_ARG $PAD_ARG #-m overwrite
cd ..
zip -r "$ZIP_NAME" "$DATA_DIR"

# Remove the data folder after zipping
rm -rf "$DATA_DIR"