TRUTH_DATASETS = {
    "truth_1": "s3://truth_1.zarr/{label}/{resolution_level}",
    "truth_2": "s3://truth_2.zarr/{label}/{resolution_level}",
}

RESOLUTION_LEVELS = {
    8: "s0",
    16: "s1",
    32: "s2",
    64: "s3",
    128: "s4",
    256: "s5",
    512: "s6",
    1024: "s7",
    2048: "s8",
}

CLASS_DATASETS = {
    "class_1": ["truth_1", "truth_2"],
    "class_2": ["truth_2", "truth_3"],
}
