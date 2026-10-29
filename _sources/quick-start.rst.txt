Quick Start
===========

The package provides two command-line entrypoints for running the RCM emulator
against prepared GCM and RCM datasets. Both run preprocessing, model inference,
and postprocessing, then save the result as a Zarr dataset.

Choose an entrypoint
--------------------

``super_resolution.rcm_emulator_loc_entrypoint`` runs the emulator for one
requested location. Supply the location as a tuple of latitude and longitude:

.. code-block:: console

   poetry run python -m super_resolution.rcm_emulator_loc_entrypoint \
       --gcm_zarr_path /path/to/gcm_data.zarr \
       --rcm_zarr_path /path/to/rcm_data.zarr \
       --location 75.0 -20.0 \
       --output_zarr_path /path/to/location_predictions.zarr \
       --model-name model_baseline_unet

The location coordinates are decimal degrees in ``(latitude, longitude)``
order. Use the location-specific entrypoint when you only need a prediction at
one point rather than the domain-wide output.

``super_resolution.rcm_emulator_entrypoint`` runs the emulator over the
entire region:

.. code-block:: console

   poetry run python -m super_resolution.rcm_emulator_entrypoint \
       --gcm_zarr_path /path/to/gcm_data.zarr \
       --rcm_zarr_path /path/to/rcm_data.zarr \
       --output_zarr_path /path/to/predictions.zarr \
       --model-name model_baseline_unet

.. warning::

   Running inference across the full region requires substantial compute
   resources and may take a long time. We recommend that the rcm_emulator_loc_entrypoint is used where possible.


Arguments
---------

``--gcm_zarr_path``
   Path to the GCM input dataset. It must include the ERA5 variables
   ``d2m``, ``msl``, ``sp``, ``sst``, ``t2m``, ``tp``, ``u10``, and ``v10``,
   with non-empty ``x``, ``y``, and ``time`` dimensions.

``--rcm_zarr_path``
   Path to the RCM input dataset in consolidated Zarr format.

``--location`` (location-specific entrypoint only)
   Latitude and longitude in decimal degrees, in that order. For example,
   ``--location 75.0 -20.0``.

``--output_zarr_path``
   Destination path for the output Zarr dataset. The entrypoint writes with
   overwrite mode, so an existing dataset at this path will be replaced.

``--model-name``
   Model to use for inference. Choose ``model_baseline_unet`` or
   ``model_attention_unet``.

Getting command help
--------------------

Each entrypoint accepts ``--help`` to show its command-line options:

.. code-block:: console

   poetry run python -m super_resolution.rcm_emulator_entrypoint --help
   poetry run python -m super_resolution.rcm_emulator_loc_entrypoint --help

