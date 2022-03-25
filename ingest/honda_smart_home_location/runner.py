from ingest.honda_smart_home_location import Pipeline
from utils import expand, set_env
import os

if __name__ == "__main__":
    set_env()
    pipeline = Pipeline(
        expand("config/pipeline_config_honda_smart_home_location.yml", __file__),
        expand("config/storage_config_honda_smart_home_location.yml", __file__),
    )

    pipeline.run(expand("input_data/Honda_AllData0420_analyzed.csv", __file__))
    #pipeline.run(expand("input_data/test.csv", __file__))
