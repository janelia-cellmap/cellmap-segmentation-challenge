from cellmap_segmentation_challenge.utils import load_safe_config

# Load the configuration file
config_path = "train_config.py"
config = load_safe_config(config_path)

# Bring the required configurations into the global namespace
batch_size = getattr(config, "batch_size", 1)
input_array_info = getattr(
    config, "input_array_info", {"shape": (1, 64, 64), "scale": (8, 8, 8)}
)
target_array_info = getattr(config, "target_array_info", input_array_info)
classes = config.classes


# Define the process function, which takes a numpy array as input and returns a numpy array as output
def process_func(x):
    # Simple thresholding function
    return x > 0.5


if __name__ == "__main__":
    from cellmap_segmentation_challenge import process

    # Call the process function with the configuration file
    process(__file__, overwrite=True)
