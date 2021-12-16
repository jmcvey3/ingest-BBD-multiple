import os
import xarray as xr
from utils import expand, set_env
from ingest.seb import Pipeline

parent = os.path.dirname(__file__)


# TODO – Developer: Update paths to your input files here. Please add tests if needed.
def test_seb_pipeline():
    set_env()
    pipeline = Pipeline(
        expand("config/pipeline_config_seb.yml", parent),
        expand("config/storage_config_seb.yml", parent),
    )
    output = pipeline.run(expand("tests/data/input/data.csv", parent))
    expected = xr.open_dataset(expand("tests/data/expected/data.csv", parent))
    xr.testing.assert_allclose(output, expected)
