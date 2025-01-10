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
        ],
        "loss": ["CellMapLossWrapper"],
        "security": ["analyze_script", "load_safe_config"],
    },
)
