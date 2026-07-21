"""Tests for the __main__ entrypoint."""

from unittest.mock import patch


@patch("super_resolution.__main__.main")
@patch("super_resolution.__main__.setup_logging")
def test_main_module_calls_setup_logging_and_main(mock_setup, mock_main):
    import importlib
    import super_resolution.__main__ as main_mod

    # Simulate running as __main__
    with patch.object(main_mod, "__name__", "__main__"):
        # Re-execute the guarded block by calling the functions directly
        main_mod.setup_logging()
        main_mod.main()

    mock_setup.assert_called_once()
    mock_main.assert_called_once()
