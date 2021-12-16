import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.decomposition import NMF
from typing import Optional

from tsdat import DSUtil, QualityChecker, QualityHandler


class DummyQCTest(QualityChecker):
    """-------------------------------------------------------------------
    Class containing placeholder code to perform a single QC test on a
    Dataset variable.

    See https://tsdat.readthedocs.io/ for more QC examples.
    -------------------------------------------------------------------"""

    def run(self, variable_name: str) -> Optional[np.ndarray]:
        """-------------------------------------------------------------------
        Test a dataset's variable to see if it passes a quality check.
        These tests can be performed on the entire variable at one time by
        using xarray vectorized numerical operators.

        Args:
            variable_name (str):  The name of the variable to check

        Returns:
            np.ndarray | None: If the test was performed, return a
            ndarray of the same shape as the variable. Each value in the
            data array will be either True or False, depending upon the
            results of the test.  True means the test failed.  False means
            it succeeded.

            Note that we are using an np.ndarray instead of an xr.DataArray
            because the DataArray contains coordinate indexes which can
            sometimes get out of sync when performing np arithmectic vector
            operations.  So it's easier to just use numpy arrays.

            If the test was skipped for some reason (i.e., it was not
            relevant given the current attributes defined for this dataset),
            then the run method should return None.
        -------------------------------------------------------------------"""

        # Just return an array of all False of same shape as the variable
        return np.full_like(self.ds[variable_name].data, False, dtype=bool)


class DummyErrorHandler(QualityHandler):
    """-------------------------------------------------------------------
    Class containing placeholder code for a custom error handler.

    See https://tsdat.readthedocs.io/ for more QC examples.
    -------------------------------------------------------------------"""

    def run(self, variable_name: str, results_array: np.ndarray):
        """-------------------------------------------------------------------
        Perform a follow-on action if a qc test fails.  This can be used to
        correct data if needed (such as replacing a bad value with missing value,
        emailing a contact persion, adding additional metadata, or raising an
        exception if the failure constitutes a critical error).

        Args:
            variable_name (str): Name of the variable that failed
            results_array (np.ndarray)  : An array of True/False values for
            each data value of the variable.  True means the test failed.
        -------------------------------------------------------------------"""
        print(f"QC test failed for variable {variable_name}")


class RemoveFailedValues(QualityHandler):
    """-------------------------------------------------------------------
    Replace all the failed values with _FillValue
    -------------------------------------------------------------------"""

    def run(self, variable_name: str, results_array: np.ndarray):
        fill_value = DSUtil.get_fill_value(self.ds, variable_name)
        self.ds[variable_name] = self.ds[variable_name].where(
            ~results_array, fill_value
        )


class ReplaceFailedValuesWithPrevious(QualityHandler):
    """-------------------------------------------------------------------
    Fill all the failed values with previous values
    -------------------------------------------------------------------"""

    def run(self, variable_name: str, results_array: np.ndarray):
        keep_array = ~results_array
        failed_indices = np.where(results_array)

        var_values = self.ds[variable_name].data
        num_indices_to_search = self.params.get("num_indices_to_search", 0)

        for index in failed_indices[0]:
            for i in range(1, num_indices_to_search + 1):
                if index - i >= 0 and keep_array[index - i]:
                    var_values[index] = var_values[index - i]
                    break


class ReplaceFailedValuesWithForwardFill(QualityHandler):
    """-------------------------------------------------------------------
    Forward Fill all the failed values
    -------------------------------------------------------------------"""

    def run(self, variable_name: str, results_array: np.ndarray):
        results = self.ds[variable_name].where(~results_array)
        da = self.ds[variable_name].where(results != 0)
        da = da.ffill("time", limit=None)
        self.ds[variable_name] = da


class ReplaceFailedValuesWithLinear(QualityHandler):
    """-------------------------------------------------------------------
    Linear Fill all the failed values
    -------------------------------------------------------------------"""

    def run(self, variable_name: str, results_array: np.ndarray):
        results = self.ds[variable_name].where(~results_array)
        da = self.ds[variable_name].where(results != 0)
        da = da.interpolate_na(
            dim="time",
            method="linear",
            fill_value=da.median(),
            limit=None,
            keep_attrs=True,
        )
        self.ds[variable_name] = da


class ReplaceFailedValuesWithPolynomial(QualityHandler):
    """-------------------------------------------------------------------
    Polynomial Fill all the failed values with Order Type 2
    -------------------------------------------------------------------"""

    def run(self, variable_name: str, results_array: np.ndarray):
        results = self.ds[variable_name].where(~results_array)
        da = self.ds[variable_name].where(results != 0)
        da = da.interpolate_na(
            dim="time", method="polynomial", order=2, limit=None, keep_attrs=True
        )
        self.ds[variable_name] = da


class ReplaceFailedValuesWithKNN(QualityHandler):
    """-------------------------------------------------------------------
    Sk-learn's K-Nearest Neighbors (KNN) Fill all dataset
    -------------------------------------------------------------------"""

    def run(self, variable_name: str, results_array: np.ndarray):

        if results_array.any():
            # If max value isn't = 0, replace 0s with nan
            for var in self.ds.data_vars:
                if self.ds[var].max() != 0:
                    self.ds[var] = self.ds[var].where(self.ds[var] != 0)

            # Run KNN using correlated "features" (column names) that meet a correlation threshold
            # Group correlated columns
            df = self.ds.to_dataframe()
            correlation_df = df.corr(method="spearman")

            correlation_threshold = 0.5  # Set this in config somehow?
            idp = np.array(np.where(correlation_df > correlation_threshold))
            # Remove self-correlated features
            idx = idp[:, ~(idp[0] == idp[1])]

            # Initiate longest possible dictionary that could be written
            length = idx.shape
            d = {}
            for i in range(length[1]):
                d[i] = []
            # Group all correlated columns together
            i_init = 0
            for j in range(0, length[1]):
                d[j].append(idx[0, j])
                for i in range(i_init, length[1]):
                    if idx[0, i] == idx[0, j]:
                        d[j].append(idx[1, i])
                    else:
                        i_init = i
                        break
                # if the inner "for" loop doesn't run
                if len(d[j]) == 1:
                    d[j] = []
            # Run grouped columns through KNN imputation
            already_run = []
            for i in range(len(d)):
                # Use dataframe b/c we already converted it
                var = df.columns[d[i]]
                # Check to see if already run or empty
                if any([nm for nm in var if nm in already_run]):
                    pass
                elif not any(var):
                    pass
                else:
                    out = KNNImputer(n_neighbors=3).fit_transform(df[var])
                    # add output directly into dataset
                    for i, nm in enumerate(var):
                        self.ds[nm].values = out[:, i]
                    already_run.extend(var)

            not_run = list(set(df.columns.values) - set(already_run))

            for col in not_run:
                data = df[col].fillna(value=df[col].median())
                self.ds[col].values = data.values


class ReplaceFailedValuesWithNMF(QualityHandler):
    """-------------------------------------------------------------------
    Sk-learn's Non-Negative Matrix Factorization (NMF) Fill all dataset
    -------------------------------------------------------------------"""

    def run(self, variable_name: str, results_array: np.ndarray):

        if results_array.any():
            # TODO Scikit learn version can't handle missing values
            # # If max value isn't = 0, replace 0s with nan
            # for var in self.ds.data_vars:
            #     if self.ds[var].max() != 0:
            #         self.ds[var] = self.ds[var].where(self.ds[var] != 0)

            var_names = [i.name for i in self.ds.data_vars]
            nmf_model = NMF(n_components=len(var_names), random_state=0, shuffle=False)

            # NMF if the gap is larger than one day
            W = nmf_model.fit_transform(self.ds.to_pandas())
            H = nmf_model.components_
            data = W.dot(H)

            for nm, i in enumerate(var_names):
                self.ds[nm].values = data[:, i]


class CheckGap(QualityChecker):
    def run(self, variable_name: str) -> Optional[np.ndarray]:
        """-------------------------------------------------------------------
        Check the rows with minimum time gap
        -------------------------------------------------------------------"""
        variables = self.params.get("variables", [variable_name])
        if variables == "All":
            variables = self.ds.keys()
        results_arrays = []

        for variable_name in variables:
            fill_value = DSUtil.get_fill_value(self.ds, variable_name)

            # If the variable has no _FillValue attribute, then
            # we select a default value to use
            if fill_value is None:
                fill_value = np.nan

            # Make sure fill value has same data type as the variable
            fill_value = np.array(
                fill_value, dtype=self.ds[variable_name].values.dtype.type
            )
            # First check if any values are assigned to _FillValue
            results_array = np.equal(self.ds[variable_name].values, fill_value)
            # Then, if the value is numeric, we should also check if any values are assigned to NaN
            if self.ds[variable_name].values.dtype.type in (
                type(0.0),
                np.float16,
                np.float32,
                np.float64,
            ):
                results_array |= np.isnan(self.ds[variable_name].values)

            # TODO: we also need to check if any values are outside valid range
            # TODO: in the config file, we need a replace with missing handler for this test

            keep_array = np.logical_not(results_array)
            timestamp = self.ds["time"].data

            min_time_gap = self.params.get("min_time_gap", 0)
            max_time_gap = self.params.get("max_time_gap", 0)

            df = pd.DataFrame({"time": timestamp, "status": keep_array})
            missing_data = df[df["status"] == 0]
            data_max_time = 0

            if not missing_data.empty:
                start_index = missing_data.head(1).index.values[0]
                end_index = missing_data.tail(1).index.values[0]
                start_time_list = []
                end_time_list = []

                for index, data in missing_data.iterrows():
                    if start_index == index:
                        pre_index = index
                        continue

                    if pre_index == index - 1:
                        pre_index = index

                    elif pre_index != index - 1:
                        time_gap = (
                            missing_data["time"][pre_index]
                            - missing_data["time"][start_index]
                        )

                        if (time_gap.seconds / 60) > data_max_time:
                            data_max_time = time_gap.seconds / 60

                        if min_time_gap < (time_gap.seconds / 60) < max_time_gap:
                            start_time_list.append(start_index)
                            end_time_list.append(pre_index)

                        pre_index = index
                        start_index = index

                    if index == end_index:
                        time_gap = (
                            missing_data["time"][pre_index]
                            - missing_data["time"][start_index]
                        )
                        if min_time_gap < (time_gap.seconds / 60) < max_time_gap:
                            start_time_list.append(start_index)
                            end_time_list.append(pre_index)

            else:
                start_time_list = []
                end_time_list = []
            print(
                f"Max time gap --> {data_max_time} minutes, [min: {min_time_gap}, max: {max_time_gap}], Number of missing gaps: {len(start_time_list)} --> {variable_name}"
            )

            keep_index = list(range(len(timestamp)))

            rev_start_time_list = start_time_list[::-1]
            rev_end_time_list = end_time_list[::-1]

            for count, i in enumerate(rev_start_time_list):
                del keep_index[
                    rev_start_time_list[count] : rev_end_time_list[count] + 1
                ]

            if keep_index:
                final_results_array = np.full(self.ds[variable_name].data.shape, True)
                final_results_array[np.array(keep_index)] = False
            else:
                final_results_array = np.full(self.ds[variable_name].data.shape, True)

            results_arrays.append(final_results_array)

        final_results_array = np.full(self.ds[variable_name].data.shape, False)

        for results_array in results_arrays:
            final_results_array |= results_array

        return final_results_array
