"""Testing the entrypoint."""

import logging
import sys
from unittest.mock import patch

from pytest import LogCaptureFixture


@patch.object(sys, "argv", ["use_case_3", "--integer_a", "4", "--integer_b", "5"])
def test_main(caplog: LogCaptureFixture) -> None:
    """Test main.

    Parameters
    ----------
    caplog : LogCaptureFixture
        The caplog fixture

    """
    from use_case_3.example_entrypoint_a import main

    with caplog.at_level(logging.INFO):
        main()
        assert "Bad Calculator Output" in caplog.records[0].message
        assert "9" in caplog.records[0].message
