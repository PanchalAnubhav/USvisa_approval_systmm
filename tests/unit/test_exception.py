"""Tests for us_visa.exception module."""

import pytest
from us_visa.exception import USvisaException, error_message_detail


class TestUSvisaException:
    """Test suite for the custom exception class."""

    def test_exception_creates_with_message(self):
        """USvisaException should be instantiable with an error message."""
        try:
            raise ValueError("test error")
        except ValueError as e:
            import sys
            exc = USvisaException(e, sys)
            assert "test error" in str(exc)
            assert "line number" in str(exc)

    def test_exception_is_instance_of_exception(self):
        """USvisaException should be a subclass of Exception."""
        try:
            raise ValueError("test")
        except ValueError as e:
            import sys
            exc = USvisaException(e, sys)
            assert isinstance(exc, Exception)

    def test_error_message_detail_contains_filename(self):
        """error_message_detail should include the script filename."""
        try:
            raise ValueError("detail test")
        except ValueError as e:
            import sys
            msg = error_message_detail(e, sys)
            assert "test_exception.py" in msg
