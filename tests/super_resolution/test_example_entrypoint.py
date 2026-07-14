"""Testing the entrypoint."""

import logging
import sys
from unittest.mock import patch

import pytest


@patch.object(sys, "argv", ["super_resolution", "--integer_a", "4", "--integer_b", "5"])
def test_main(caplog: pytest.LogCaptureFixture) -> None:
    """
    Test main.

    Parameters
    ----------
    caplog : pytest.LogCaptureFixture
        The caplog fixture

    """
    from super_resolution.example_entrypoint import app

    with caplog.at_level(logging.INFO):
        app(result_action="return_value")
        assert "Bad Calculator Output" in caplog.records[0].message
        assert "9" in caplog.records[0].message
