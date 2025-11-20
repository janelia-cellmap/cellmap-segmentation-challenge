"""Unit tests for security functions in cellmap_segmentation_challenge.utils.security"""

import tempfile
import os

from cellmap_segmentation_challenge.utils.security import (
    analyze_script,
    Config,
)


class TestAnalyzeScript:
    """Tests for analyze_script function"""

    def test_analyze_safe_script(self):
        """Test analysis of a safe script"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
import numpy as np
import torch

def safe_function():
    x = np.array([1, 2, 3])
    return x * 2
"""
            )
            f.flush()
            try:
                is_safe, issues = analyze_script(f.name)
                assert is_safe is True
                assert len(issues) == 0
            finally:
                os.unlink(f.name)

    def test_analyze_script_with_disallowed_import(self):
        """Test detection of disallowed imports"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
import os
import sys

def unsafe_function():
    os.system("echo test")
"""
            )
            f.flush()
            try:
                is_safe, issues = analyze_script(f.name)
                assert is_safe is False
                assert len(issues) > 0
                assert any("os" in issue for issue in issues)
            finally:
                os.unlink(f.name)

    def test_analyze_script_with_disallowed_function(self):
        """Test detection of disallowed function calls"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
def unsafe_function():
    exec("print('hello')")
    compile("x = 1", "", "exec")
"""
            )
            f.flush()
            try:
                is_safe, issues = analyze_script(f.name)
                assert is_safe is False
                assert len(issues) > 0
                assert any("exec" in issue.lower() for issue in issues)
            finally:
                os.unlink(f.name)

    def test_analyze_script_with_from_import(self):
        """Test detection of disallowed from imports"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
from subprocess import run

def unsafe_function():
    run(["echo", "test"])
"""
            )
            f.flush()
            try:
                is_safe, issues = analyze_script(f.name)
                assert is_safe is False
                assert len(issues) > 0
                assert any("subprocess" in issue for issue in issues)
            finally:
                os.unlink(f.name)


class TestConfig:
    """Tests for Config class"""

    def test_config_initialization(self):
        """Test Config initialization with kwargs"""
        config = Config(learning_rate=0.001, batch_size=32, epochs=10)
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.epochs == 10

    def test_config_to_dict(self):
        """Test Config to_dict method"""
        config = Config(learning_rate=0.001, batch_size=32)
        config_dict = config.to_dict()
        assert config_dict == {"learning_rate": 0.001, "batch_size": 32}

    def test_config_get_existing_key(self):
        """Test Config get method with existing key"""
        config = Config(learning_rate=0.001, batch_size=32)
        assert config.get("learning_rate") == 0.001
        assert config.get("batch_size") == 32

    def test_config_get_missing_key_with_default(self):
        """Test Config get method with missing key and default"""
        config = Config(learning_rate=0.001)
        assert config.get("batch_size", 16) == 16
        assert config.get("epochs", 10) == 10

    def test_config_get_missing_key_no_default(self):
        """Test Config get method with missing key and no default"""
        config = Config(learning_rate=0.001)
        assert config.get("batch_size") is None

    def test_config_serialize_simple_types(self):
        """Test Config serialize with simple data types"""
        config = Config(
            learning_rate=0.001, batch_size=32, model_name="unet", use_cuda=True
        )
        serialized = config.serialize()
        assert serialized["learning_rate"] == 0.001
        assert serialized["batch_size"] == 32
        assert serialized["model_name"] == "unet"
        assert serialized["use_cuda"] is True

    def test_config_serialize_skips_modules_and_functions(self):
        """Test that serialize skips modules, classes, and functions"""
        import torch

        def my_function():
            pass

        config = Config(learning_rate=0.001, module=torch, function=my_function)
        serialized = config.serialize()

        # Should only include simple types
        assert "learning_rate" in serialized
        assert "module" not in serialized
        assert "function" not in serialized

    def test_config_serialize_skips_private_attributes(self):
        """Test that serialize skips private attributes (with __)"""
        config = Config(learning_rate=0.001, __private_value=42)
        serialized = config.serialize()
        assert "learning_rate" in serialized
        assert "__private_value" not in serialized

    def test_config_serialize_converts_complex_types_to_string(self):
        """Test that serialize converts complex types to strings"""
        config = Config(learning_rate=0.001, shape=(64, 64, 64), data=[1, 2, 3])
        serialized = config.serialize()
        assert serialized["learning_rate"] == 0.001
        assert isinstance(serialized["shape"], str)
        assert isinstance(serialized["data"], str)
