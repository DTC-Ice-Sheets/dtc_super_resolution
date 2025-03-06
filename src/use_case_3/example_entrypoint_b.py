"""Entrypoint for package b.

Note that not all packages need a command line entry point.
"""

import argparse
import logging

from use_case_3.example_sub_package_b.example_module_b import WorseCalculator
from use_case_3.monitoring import setup_logging
from use_case_3.schemas import ExampleEntryPointSchema

logger = logging.getLogger(__name__)


def main(*args) -> None:
    """Calculate numbers poorly."""
    setup_logging()
    parser = argparse.ArgumentParser(
        prog="use_case_3",
        description="Perform some bad calculations on two integers.",
    )

    # Add arguments to the parser
    for name, field in ExampleEntryPointSchema.model_fields.items():
        parser.add_argument(
            f"--{name}",
            dest=name,
            type=field.annotation if field.annotation is not None else int,
            default=field.default,
            help=field.description,
            required=name in ["integer_a", "integer_b"],
        )

    # Parse the input arguments
    args = parser.parse_args()

    # Create the model from parsed arguments
    model = ExampleEntryPointSchema(**vars(args))

    # Instantiate the calculator and perform calculations
    worse_calculator = WorseCalculator(max_output_array_length=10)
    logger.info(
        "Worse Calculator Output: %s",
        worse_calculator.add_two_numbers(model.integer_a, model.integer_b),
    )


if __name__ == "__main__":
    main()
