import ast
import importlib
from importlib.machinery import SourceFileLoader
import os

from upath import UPath

# Define restricted imports and functions
DISALLOWED_IMPORTS = {"os", "subprocess", "sys"}
# DISALLOWED_FUNCTIONS = {"eval", "exec", "open", "compile", "__import__"}
DISALLOWED_FUNCTIONS = {"eval", "exec", "compile", "__import__"}


def analyze_script(filepath):
    """
    Analyzes the script at `filepath` using `ast` for potentially unsafe imports and function calls.
    Returns a boolean indicating whether the script is safe and a list of detected issues.
    """
    issues = []
    with open(filepath, "r") as file:
        source_code = file.read()

    # Parse the code into an AST
    tree = ast.parse(source_code, filename=filepath)

    # Traverse the AST and analyze nodes
    for node in ast.walk(tree):
        # Check for disallowed imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in DISALLOWED_IMPORTS:
                    issues.append(f"Disallowed import detected: {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            if node.module in DISALLOWED_IMPORTS:
                issues.append(f"Disallowed import detected: {node.module}")

        # Check for disallowed function calls
        elif isinstance(node, ast.Call):
            # If function is a direct name (e.g., `eval()`)
            if isinstance(node.func, ast.Name) and node.func.id in DISALLOWED_FUNCTIONS:
                issues.append(f"Disallowed function call detected: {node.func.id}")
            # If function is an attribute call (e.g., `os.system()`)
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in DISALLOWED_FUNCTIONS
            ):
                issues.append(f"Disallowed function call detected: {node.func.attr}")

    # Return whether the script is safe (no issues found) and the list of issues
    is_safe = len(issues) == 0
    return is_safe, issues


def load_safe_config(config_path, force_safe=os.getenv("FORCE_SAFE_CONFIG", True)):
    """
    Loads the configuration script at `config_path` after verifying its safety.
    If `force_safe` is True, raises an error if the script is deemed unsafe.
    """
    # print(f"Analyzing script for obvious security liabilities:\n\t{config_path}")
    # print(
    #     "Keep in mind that this is not a foolproof security measure. Use caution using code from untrusted sources."
    # )
    is_safe, issues = analyze_script(config_path)
    if not is_safe:
        print("Script contains unsafe elements:")
        for issue in issues:
            print(f" - {issue}")
        if force_safe:
            raise ValueError(
                "Unsafe script detected; loading aborted. You can set the environment variable FORCE_SAFE_CONFIG=False or pass force_safe=False to override."
            )

    # Load the config module if script is safe
    config_path = UPath(config_path)
    # Create a dedicated namespace for the config
    config_namespace = {}
    try:
        with open(config_path, "r") as config_file:
            code = config_file.read()
            if "__file__" in code:
                code = code.replace("__file__", f'"{config_path}"')
            exec(code, config_namespace)
        # Extract the config object from the namespace
        config = Config(**config_namespace)
    except Exception as e:
        print(e)
        raise RuntimeError(
            f"Failed to execute configuration file: {config_path}"
        ) from e

    return config


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
