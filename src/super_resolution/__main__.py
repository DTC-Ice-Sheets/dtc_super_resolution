"""Main entrypoint for the super resolution module."""

from super_resolution.example_entrypoint import main
from super_resolution.monitoring import setup_logging

if __name__ == "__main__":
    setup_logging()
    main()
