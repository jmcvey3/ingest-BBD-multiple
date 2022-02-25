import os
from glob import glob

from ingest.seb import Pipeline
from utils import expand, set_env


if __name__ == "__main__":
    set_env()
    pipeline = Pipeline(
        expand("config/pipeline_config_seb.yml", __file__),
        expand("config/storage_config_seb.yml", __file__),
    )

    # os.chdir("ingest/seb")
    # files = glob(os.path.join("data", "*.csv"))
    # for fname in files:
    #     pipeline.run(expand(fname, __file__))

    pipeline.run(expand("data/selected_raw_merge_data_2019_2020.csv", __file__))
