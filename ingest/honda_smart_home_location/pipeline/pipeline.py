import os
import cmocean
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from typing import Dict
from tsdat import DSUtil
from utils import IngestPipeline, format_time_xticks
import matplotlib as mpl


example_dir = os.path.abspath(os.path.dirname(__file__))

# TODO – Developer: Use hooks to add custom functionality to the pipeline including
# plots, as applicable. Remove any unused code.


class Pipeline(IngestPipeline):
    """--------------------------------------------------------------------------------
    HONDA SMART HOME INGESTION PIPELINE

    description

    --------------------------------------------------------------------------------"""

    def hook_customize_raw_datasets(
        self, raw_dataset_mapping: Dict[str, xr.Dataset]
    ) -> Dict[str, xr.Dataset]:
        return raw_dataset_mapping

    def hook_customize_dataset(
        self, dataset: xr.Dataset, raw_mapping: Dict[str, xr.Dataset]
    ) -> xr.Dataset:
        return dataset

    def hook_finalize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        return dataset

    def hook_generate_and_persist_plots(self, dataset: xr.Dataset) -> None:
        ds = dataset
        print(ds)

        for var in ds.data_vars:
            if "qc" not in var:
                nm = var.replace("\\", "/").rsplit("/")[-1]
                filename = DSUtil.get_plot_filename(ds, nm, "png")
                with self.storage._tmp.get_temp_filepath(filename) as tmp_path:

                    # Create the figure and axes objects
                    fig, ax = plt.subplots(
                        nrows=1, ncols=1, figsize=(14, 8), constrained_layout=True
                    )
                    fig.suptitle(nm)
                    ds[var].plot(label="Raw data")
                    ds[var].where(ds["qc_" + var]).plot(label="QC inserted")
                    plt.legend()
                    # Set the labels and ticks
                    # format_time_xticks(ax)
                    ax.set_title("")  # Remove title created by xarray
                    ax.set_xlabel("Time (UTC)")
                    # ax.set_ylabel(r"Wind Speed (ms$^{-1}$)")

                    # Save the figure
                    fig.savefig(tmp_path, dpi=100)
                    self.storage.save(tmp_path)
                    plt.close()
        return
