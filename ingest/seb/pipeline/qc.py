import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.decomposition import NMF
from typing import Optional

from tsdat import DSUtil, QualityChecker, QualityHandler


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
        da = self.ds[variable_name].where(~results_array)
        da = da.ffill("time", limit=None)
        self.ds[variable_name] = da


class ReplaceFailedValuesWithLinear(QualityHandler):
    """-------------------------------------------------------------------
    Linear Fill all the failed values
    -------------------------------------------------------------------"""

    def run(self, variable_name: str, results_array: np.ndarray):
        da = self.ds[variable_name].where(~results_array)
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
        da = self.ds[variable_name].where(~results_array)
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
            self.ds[variable_name] = self.ds[variable_name].where(~results_array)

            # Run KNN using correlated "features" (column names) that meet a correlation threshold
            # Group correlated columns
            df = self.ds.to_dataframe()
            correlation_df = df.corr(method="spearman")

            idp = np.array(np.where(correlation_df > self.params["correlation_thresh"]))
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

            var_to_knn = []
            for i in range(len(d)):
                # Use dataframe b/c we already converted it
                var = df.columns[d[i]]
                if variable_name in var:
                    var_to_knn.extend(var)

            # Run grouped columns through KNN imputation
            if len(var_to_knn) > 1:
                var_to_knn = np.unique(var_to_knn)
                out = KNNImputer(n_neighbors=3).fit_transform(df[var_to_knn])

                # add output directly into dataset
                idx = np.where(variable_name in var_to_knn)[0][0]
                self.ds[variable_name].values = out[:, idx]
            else:
                # Fills in all nans
                self.ds[variable_name] = self.ds[variable_name].fillna(
                    self.ds[variable_name].median()
                )
                # Only replace nan data in results array, ignores time before data started being saved
                # self.ds[variable_name] = self.ds[variable_name].where(
                #     ~results_array, other=df[variable_name].median()
                # )


class ReplaceFailedValuesWithNMF(QualityHandler):
    """-------------------------------------------------------------------
    Sk-learn's Non-Negative Matrix Factorization (NMF) Fill all dataset
    -------------------------------------------------------------------"""

    def run(self, variable_name: str, results_array: np.ndarray):
        if results_array.any():
            self.ds[variable_name] = self.ds[variable_name].where(~results_array)

            # TODO Scikit learn version can't handle missing values
            var_names = [i.name for i in self.ds.data_vars]
            nmf_model = NMF(n_components=len(var_names), random_state=0, shuffle=False)

            # NMF if the gap is larger than one day
            W = nmf_model.fit_transform(self.ds.to_pandas())
            H = nmf_model.components_
            out = W.dot(H)

            idx = np.where(variable_name in var_names)[0]
            self.ds[variable_name].values = out[:, idx]


class CheckGap(QualityChecker):
    def run(self, variable_name: str) -> Optional[np.ndarray]:
        """-------------------------------------------------------------------
        Check the rows with minimum time gap
        -------------------------------------------------------------------"""
        # Misses nan's at beginning of dataset
        variables = self.params.get("variables", [variable_name])
        results_array = []
        if variables == "All" or variable_name in variables:
            pass
        else:
            return results_array

        fill_value = DSUtil.get_fill_value(self.ds, variable_name)

        # If the variable has no _FillValue attribute, then
        # we select a default value to use
        if fill_value is None:
            fill_value = np.nan

        # Make sure fill value has same data type as the variable
        fill_value = np.array(
            fill_value, dtype=self.ds[variable_name].values.dtype.type
        )

        # Remove 0's from variables
        if self.ds[variable_name].max() != 0:
            self.ds[variable_name] = self.ds[variable_name].where(
                self.ds[variable_name] != 0
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
        if len(start_time_list):
            print(
                f"Max time gap --> {data_max_time} minutes, [min: {min_time_gap}, max: {max_time_gap}], Number of missing gaps: {len(start_time_list)} --> {variable_name}"
            )

        keep_index = list(range(len(timestamp)))

        rev_start_time_list = start_time_list[::-1]
        rev_end_time_list = end_time_list[::-1]

        for count, i in enumerate(rev_start_time_list):
            del keep_index[rev_start_time_list[count] : rev_end_time_list[count] + 1]

        if keep_index:
            final_results_array = np.full(self.ds[variable_name].data.shape, True)
            final_results_array[np.array(keep_index)] = False
        else:
            final_results_array = np.full(self.ds[variable_name].data.shape, True)

        return final_results_array
