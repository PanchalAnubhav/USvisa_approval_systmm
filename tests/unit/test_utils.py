"""Tests for us_visa.utils.main_utils module."""

import os
import tempfile
import pytest
import numpy as np
import yaml

from us_visa.utils.main_utils import (
    read_yaml_file,
    write_yaml_file,
    save_object,
    load_object,
    save_numpy_array_data,
    load_numpy_array_data,
)


class TestYamlUtils:
    """Test suite for YAML read/write utilities."""

    def test_read_yaml_file(self, tmp_path):
        """read_yaml_file should correctly parse a YAML file."""
        yaml_content = {"key": "value", "nested": {"a": 1}}
        file_path = str(tmp_path / "test.yaml")
        with open(file_path, "w") as f:
            yaml.dump(yaml_content, f)

        result = read_yaml_file(file_path)
        assert result["key"] == "value"
        assert result["nested"]["a"] == 1

    def test_write_yaml_file(self, tmp_path):
        """write_yaml_file should create a valid YAML file."""
        content = {"status": True, "message": "test"}
        file_path = str(tmp_path / "subdir" / "output.yaml")

        write_yaml_file(file_path=file_path, content=content)

        assert os.path.exists(file_path)
        with open(file_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["status"] is True

    def test_write_yaml_file_replace(self, tmp_path):
        """write_yaml_file with replace=True should overwrite existing file."""
        file_path = str(tmp_path / "replace.yaml")
        write_yaml_file(file_path=file_path, content={"v": 1})
        write_yaml_file(file_path=file_path, content={"v": 2}, replace=True)

        result = read_yaml_file(file_path)
        assert result["v"] == 2


class TestObjectSerialization:
    """Test suite for object save/load utilities."""

    def test_save_and_load_object(self, tmp_path):
        """save_object and load_object should roundtrip correctly."""
        obj = {"model": "test", "params": [1, 2, 3]}
        file_path = str(tmp_path / "test.pkl")

        save_object(file_path=file_path, obj=obj)
        loaded = load_object(file_path=file_path)

        assert loaded == obj

    def test_save_and_load_numpy_array(self, tmp_path):
        """save_numpy_array_data and load_numpy_array_data should roundtrip correctly."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        file_path = str(tmp_path / "test.npy")

        save_numpy_array_data(file_path=file_path, array=arr)
        loaded = load_numpy_array_data(file_path=file_path)

        np.testing.assert_array_equal(loaded, arr)
