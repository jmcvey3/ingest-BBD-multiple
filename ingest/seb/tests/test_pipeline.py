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
    output = pipeline.run(
        expand("tests/data/input/selected_raw_merge_data_2019_2020.csv", parent)
    )
    expected = xr.open_dataset(
        expand(
            "tests/data/expected/SEB.BBD-parameters-1min.b1.20190801.003400.nc", parent
        )
    )
    xr.testing.assert_allclose(output, expected)


if __name__ == "__main__":
    test_seb_pipeline()
