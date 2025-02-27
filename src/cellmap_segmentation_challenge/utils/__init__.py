import lazy_loader as lazy

# Lazy-load submodules
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "crops": [
            "CropsRow",
            "fetch_crop_manifest",
            "TestCropRow",
            "fetch_test_crop_manifest",
            "TEST_CROPS",
            "TEST_CROPS_DICT",
            "get_test_crops",
        ],
        "dataloader": ["get_dataloader"],
        "datasplit": [
            "make_datasplit_csv",
            "make_s3_datasplit_csv",
            "get_class_relations",
            "get_tested_classes",
            "get_formatted_fields",
        ],
        "loss": ["CellMapLossWrapper"],
        "security": ["analyze_script", "load_safe_config"],
        "utils": [
            "construct_test_crop_manifest",
            "construct_truth_dataset",
            "simulate_predictions_iou",
            "simulate_predictions_iou_binary",
            "simulate_predictions_accuracy",
            "perturb_instance_mask",
            "download_file",
            "format_string",
        ],
        "submission": [
            "package_submission",
            "save_numpy_class_arrays_to_zarr",
            "save_numpy_class_labels_to_zarr",
        ],
    },
)
