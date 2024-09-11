# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Exports EE ImageCollections to Zarr using Xarray-Beam."""


import logging
from typing import Dict, List

from absl import app
from absl import flags
from absl.flags import argparse_flags
import apache_beam as beam
import xarray as xr
import xarray_beam as xbeam
import xee

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


_INPUT = flags.DEFINE_string(
    'input', '', help='The input Earth Engine ImageCollection.'
)
_CRS = flags.DEFINE_string(
    'crs',
    'EPSG:4326',
    help='Coordinate Reference System for output Zarr.',
)
_SCALE = flags.DEFINE_float('scale', 0.01, help='Scale factor for output Zarr.')
_OUTPUT = flags.DEFINE_string('output', '', help='The output zarr path.')
_RUNNER = flags.DEFINE_string('runner', None, help='beam.runners.Runner')


# pylint: disable=unused-argument
def _parse_dataflow_flags(argv: List[str]) -> List[str]:
  parser = argparse_flags.ArgumentParser(
      description='parser for dataflow flags',
      allow_abbrev=False,
  )


def main(argv: list[str]) -> None:
  assert _INPUT.value, 'Must specify --input'
  assert _OUTPUT.value, 'Must specify --output'

  target_chunks = {'lon': 1024, 'lat': 1024}

  init_params = dict(
    project='YOUR-CLOUD-PROJECT', # update to your cloud project
    opt_url=ee.data.HIGH_VOLUME_API_BASE_URL
  )

  ee.Initialize(**init_params)

  # need to update to an asset that is public
  clip_fc = ee.FeatureCollection(
    'ASSET_ID'
  )
  clip_geometry = clip_fc.geometry()

  conus = clip_geometry.bounds()

  input_coll = (
      ee.ImageCollection(_INPUT.value)
      .select(['embedding_B.*'])
      .filterBounds(conus)
      .mosaic()
      .set('system:time_start', ee.Date('2022-01-01').millis())
  )

  ds = xr.open_dataset(
      ee.ImageCollection(input_coll),
      crs=_CRS.value,
      scale=_SCALE.value,
      engine=xee.EarthEngineBackendEntrypoint,
      geometry=conus,
      ee_init_if_necessary=True,
      ee_init_kwargs=init_params,
      io_chunks=target_chunks,
      getitem_kwargs=dict(max_retries=10,),
  )

  template = xbeam.make_template(ds)
  itemsize = max(variable.dtype.itemsize for variable in template.values())

  with beam.Pipeline(runner=_RUNNER.value, argv=argv) as root:
    _ = (
        root
        | xbeam.DatasetToChunks(ds, chunks=target_chunks)
        | xbeam.ChunksToZarr(_OUTPUT.value, template, target_chunks)
    )


if __name__ == '__main__':
  app.run(main, flags_parser=_parse_dataflow_flags)