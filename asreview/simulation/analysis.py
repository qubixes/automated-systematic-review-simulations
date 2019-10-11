'''
Analysis and reading of log files.

Merged versions of functions work on the results of all files at the same time.
'''

import itertools
import os
import numpy as np
from scipy import stats

from asreview.simulation.readers import get_num_reviewed
from asreview.simulation.readers import read_json_results
from asreview.simulation.readers import reorder_results
from asreview.simulation.statistics import _ROC_merged, _get_labeled_order,\
    _get_proba_order
from asreview.simulation.statistics import _limits_merged
from asreview.simulation.statistics import _find_inclusions


class Analysis():
    """ Analysis object to plot things from the logs. """

    def __init__(self, data_dir):
        self.labels = None
        self.final_labels = None
        self.empty = True

        self.data_dir = data_dir
        self.key = os.path.basename(os.path.normpath(data_dir))
        self.results = read_json_results(data_dir)
        self.num_runs = len(self.results)
        if self.num_runs == 0:
            return

        self._first_file = list(self.results.keys())[0]
        self._rores = reorder_results(self.results)
        self._n_reviewed = get_num_reviewed(self.results)
        self.labels = self.results[self._first_file].get('labels', None)
        self.labels = np.array(self.labels, dtype=np.int)
        self.final_labels = self.results[self._first_file].get('final_labels',
                                                               None)
        self.empty = False

    @classmethod
    def from_dir(cls, data_dir):
        analysis_inst = Analysis(data_dir)
        if analysis_inst.empty:
            return None
        return analysis_inst

    def get_inc_found(self, final_labels=False, WSS_measures=[],
                      RRF_measures=[]):

        if final_labels:
            labels = self.final_labels
        else:
            labels = self.labels

        inclusions_found = []

        for res in self.results.values():
            inclusions, inc_after_init, n_initial = _find_inclusions(
                res["results"], labels)
            inclusions_found.append(inclusions)

        inc_found_avg = []
        inc_found_err = []
        for i_instance in itertools.count():
            cur_vals = []
            for i_file in range(self.num_runs):
                try:
                    cur_vals.append(inclusions_found[i_file][i_instance])
                except IndexError:
                    pass
            if len(cur_vals) == 0:
                break
            if len(cur_vals) == 1:
                err = cur_vals[0]
            else:
                err = stats.sem(cur_vals)
            avg = np.mean(cur_vals)
            inc_found_avg.append(avg)
            inc_found_err.append(err)

        dy = 0
        dx = 0
        x_norm = (len(labels)-n_initial)
        y_norm = (inc_after_init)

        norm_xr = (np.arange(len(inc_found_avg))-dx)/x_norm
        norm_yr = (np.array(inc_found_avg)-dy)/y_norm
        norm_y_err = np.array(inc_found_err)/y_norm
        if self.num_runs == 1:
            norm_y_err = np.zeros(len(norm_y_err))

        WSS_RRF_measures = []
        for i, WSS in enumerate(WSS_measures):
            WSS = round(WSS)
            if WSS < 0:
                WSS = 0
            if WSS > 100:
                WSS = 100
            key = "WSS" + str(WSS)
            WSS_RRF_measures.append(["WSS", WSS, key, None])

        for i, RRF in enumerate(RRF_measures):
            RRF = round(RRF)
            if RRF < 0:
                RRF = 0
            if RRF > 100:
                RRF = 100
            key = "RRF" + str(RRF)
            WSS_RRF_measures.append(["RRF", RRF, key, None])

        for i in range(len(norm_yr)):
            for measure in WSS_RRF_measures:
                if measure[0] == "RRF":
                    if measure[3] is None and norm_xr[i] >= measure[1]/100-1e-7:
                        measure[3] = (norm_yr[i], norm_xr[i])
                elif measure[0] == "WSS":
                    if measure[3] is None and norm_yr[i] >= measure[1]/100-1e-7:
                        measure[3] = (norm_yr[i] - norm_xr[i], norm_xr[i])
                else:
                    raise ValueError("Measure type unknown.")

        result = {
            "data": [norm_xr, norm_yr, norm_y_err],
        }
        for measure in WSS_RRF_measures:
            result[measure[2]] = measure[3]
        return result

    def avg_time_to_discovery(self):
        labels = self.labels

        one_labels = np.where(labels == 1)[0]
        time_results = {label: [] for label in one_labels}
        n_initial = []

        for i_file, result in enumerate(self.results.values()):
            result = result["results"]
            label_order, n = _get_labeled_order(result)
            n_initial.append(n)
            try:
                query_i = -1
                while "pool_proba" not in result[query_i]:
                    query_i -= 1
                proba_order = _get_proba_order(result[query_i]["pool_proba"])
            except IndexError:
                proba_order = []

            for i_time, idx in enumerate(label_order):
                if labels[idx] == 1:
                    time_results[idx].append(i_time)

            for i_time, idx in enumerate(proba_order):
                if labels[idx] == 1 and len(time_results[idx]) <= i_file:
                    time_results[idx].append(i_time + len(label_order))

            for idx in time_results:
                if len(time_results[idx]) < i_file+1:
                    time_results[idx].append(len(label_order) + len(proba_order))

        results = {}
        for label in time_results:
            trained_time = []
            for i_file, time in enumerate(time_results[label]):
                if time >= n_initial[i_file]:
                    trained_time.append(time)
            if len(trained_time) == 0:
                results[label] = 0
            else:
                results[label] = np.average(trained_time)
        return results

    def stat_test_merged(self, logname, stat_fn, final_labels=False, **kwargs):
        """
        Do a statistical test on the results.

        Arguments
        ---------
        logname: str
            Logname as given in the log file (e.g. "pool_proba").
        stat_fn: func
            Function to gather statistics, use merged_* version.
        kwargs: dict
            Extra keywords for the stat_fn function.

        Returns
        -------
        list:
            Results of the statistical test, format depends on stat_fn.
        """
        stat_results = []
        results = self._rores[logname]
        if final_labels and self.final_labels is not None:
            labels = self.final_labels
        else:
            labels = self.labels

        x_range = []
        for i_query, query in enumerate(results):
            if len(query) == 0:
                continue
            new_res = stat_fn(query, labels, **kwargs)
            stat_results.append(new_res)
            x_range.append(self._n_reviewed[i_query])
        return x_range, stat_results

    def ROC(self):
        """
        Plot the ROC for all directories and both the pool and
        the train set.
        """
        _, pool_roc = self.stat_test_merged("pool_proba", _ROC_merged)
        x_range, label_roc = self.stat_test_merged("train_proba", _ROC_merged)
        pool_roc_avg = []
        pool_roc_err = []
        train_roc_avg = []
        train_roc_err = []
        for pool_data in pool_roc:
            pool_roc_avg.append(pool_data[0])
            pool_roc_err.append(pool_data[1])
        for label_data in label_roc:
            train_roc_avg.append(label_data[0])
            train_roc_err.append(label_data[1])

        result = {
            "x_range": x_range,
            "pool": (pool_roc_avg, pool_roc_err),
            "train": (train_roc_avg, train_roc_err),
        }
        return result

    def limits(self, prob_allow_miss=[0.1]):
        results = {"x_range": self._n_reviewed, "limits": []}
        for p_miss in prob_allow_miss:
            x_range, limits_res = self.stat_test_merged(
                "pool_proba", _limits_merged, p_allow_miss=p_miss)
            results["x_range"] = x_range
            results["limits"].append(limits_res)
        return results
