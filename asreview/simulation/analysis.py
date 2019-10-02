'''
Analysis and reading of log files.

Merged versions of functions work on the results of all files at the same time.
'''

import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from asreview.simulation.readers import get_num_reviewed
from asreview.simulation.readers import read_json_results
from asreview.simulation.readers import reorder_results
from asreview.simulation.statistics import _ROC_merged
from asreview.simulation.statistics import _avg_proba_merged
from asreview.simulation.statistics import _speedup_merged
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
        self.final_labels = self.results[self._first_file].get('final_labels',
                                                               None)
        self.empty = False

    @classmethod
    def from_dir(cls, data_dir):
        analysis_inst = Analysis(data_dir)
        if analysis_inst.empty:
            return None
        return analysis_inst

    def get_inc_found(self, final_labels=False):

        if final_labels:
            labels = self.final_labels
        else:
            labels = self.labels

        inclusions_found = []
#         inclusions_after_init = []
#         num_initial = []
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
        x_norm = (len(labels)-n_initial)/100
        y_norm = (inc_after_init)/100

        norm_xr = (np.arange(len(inc_found_avg))-dx)/x_norm
        norm_yr = (np.array(inc_found_avg)-dy)/y_norm
        norm_y_err = np.array(inc_found_err)/y_norm
        return [norm_xr, norm_yr, norm_y_err]

    def _avg_time_found(self, _dir):
        results = self._rores[_dir]['labelled']
        time_results = {}
        res = {}
        num_query = len(results)
        n_labels = len(self._labels[_dir])
        n_queries = len(self._n_queries[_dir])

        for i, query in enumerate(results):
            n_queried = self._n_queries[_dir][i]
            n_files = len(results[query])
            for query_list in results[query]:
                for label_inc in query_list:
                    label = label_inc[0]
                    include = label_inc[1]
                    if not include:
                        continue
                    if label not in time_results:
                        time_results[label] = [0, 0, 0]
                    if i == 0:
                        time_results[label][2] += 1
                    else:
                        time_results[label][0] += n_queried
                        time_results[label][1] += 1
        penalty_not_found = 2*self._n_queries[_dir][n_queries-1] - self._n_queries[_dir][n_queries-2]
        for label in time_results:
            tres = time_results[label]
            n_not_found = n_files - tres[1] - tres[2]
#             print(n_not_found, penalty_not_found, n_files, tres[2])
            if n_files-tres[2]:
                res[label] = (n_not_found*penalty_not_found + tres[0])/(n_files-tres[2])
            else:
                res[label] = 0

        return res

    def print_avg_time_found(self):
        time_hist = []
        for _dir in self._dirs:
            res_dict = self._avg_time_found(_dir)
            res = list(res_dict.values())
            time_hist.append(res)
            for label in res_dict:
                if res_dict[label] > 1400:
                    print(f"{_dir}: label={label}, value={res_dict[label]}")
#             if time_hist is None:
#                 time_hist = np.array([res])
#             else:
#                 print((time_hist, np.array([res])))
#                 print(time_hist.shape)
#                 print(np.array([res]).shape)
#                 time_hist = np.concatenate((time_hist, np.array([res])))
#             print(time_hist)
#             time_hist = np.append(time_hist, np.array([res]), axis=1)
#         print(time_hist)
        plt.hist(time_hist, density=False)
        plt.show()

    def stat_test_merged(self, logname, stat_fn, final_labels=False, **kwargs):
        """
        Do a statistical test on the results.

        Arguments
        ---------
        _dir: str
            Base directory key (path removed).
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
        print(self._rores)
        results = self._rores[logname]
        if final_labels and self._final_labels is not None:
            labels = self._final_labels
        else:
            labels = self._labels
        for query in results:
            new_res = stat_fn(results[query], labels, **kwargs)
            stat_results.append(new_res)
        return stat_results

    def plot_ROC(self):
        """
        Plot the ROC for all directories and both the pool and
        the train set.
        """
        legend_name = []
        legend_plt = []
        pool_name = "pool_proba"
        label_name = "train_proba"
        for i, _dir in enumerate(self._dirs):
            pool_roc = self.stat_test_merged(
                _dir, pool_name, _ROC_merged)
            label_roc = self.stat_test_merged(
                _dir, label_name, _ROC_merged)
            cur_pool_roc = []
            cur_pool_err = []
            cur_label_roc = []
            cur_label_err = []
            xr = self._n_queries[_dir]
            for pool_data in pool_roc:
                cur_pool_roc.append(pool_data[0])
                cur_pool_err.append(pool_data[1])
            for label_data in label_roc:
                cur_label_roc.append(label_data[0])
                cur_label_err.append(label_data[1])

            col = "C"+str(i % 10)
            myplot = plt.errorbar(xr, cur_pool_roc, cur_pool_err, color=col)
            plt.errorbar(xr, cur_label_roc, cur_label_err, color=col, ls="--")
            legend_name.append(f"{_dir}")
            legend_plt.append(myplot)

        plt.legend(legend_plt, legend_name, loc="upper right")
        plt.title("Area Under Curve of ROC")
        plt.show()

    def plot_proba(self):
        """
        Plot the average prediction probabilities of samples in the pool
        and training set.
        """
        pool_plt = []
        pool_leg_name = []
        label_plt = []
        label_leg_name = []
        legend_name = []
        legend_plt = []
        pool_name = "pool_proba"
        label_name = "train_proba"
        linestyles = ['-', '--', '-.', ':']

        for i, _dir in enumerate(self._dirs):
            pool_proba = self.stat_test_merged(
                _dir, pool_name, _avg_proba_merged)
            label_proba = self.stat_test_merged(
                _dir, label_name, _avg_proba_merged)
            col = "C"+str(i % 10)
            for true_cat in range(2):
                cur_pool_prob = []
                cur_pool_err = []
                cur_label_prob = []
                cur_label_err = []
                xr = self._n_queries[_dir]
                for pool_data in pool_proba:
                    cur_pool_prob.append(pool_data[true_cat][1])
                    cur_pool_err.append(pool_data[true_cat][2])
                for label_data in label_proba:
                    cur_label_prob.append(label_data[true_cat][1])
                    cur_label_err.append(label_data[true_cat][2])
                ls1 = linestyles[true_cat*2]
                ls2 = linestyles[true_cat*2+1]
                myplot = plt.errorbar(xr, cur_pool_prob, cur_pool_err,
                                      color=col, ls=ls1)
                myplot2 = plt.errorbar(xr, cur_label_prob, cur_label_err,
                                       color=col, ls=ls2)
                if i == 0:
                    pool_plt.append(myplot)
                    pool_leg_name.append(f"Pool: label = {true_cat}")
                    label_plt.append(myplot2)
                    label_leg_name.append(f"Train: label = {true_cat}")
                if true_cat == 1:
                    legend_plt.append(myplot)

            legend_name.append(f"{_dir}")
        legend_name += pool_leg_name
        legend_name += label_leg_name
        legend_plt += pool_plt
        legend_plt += label_plt
        plt.legend(legend_plt, legend_name, loc="upper right")
        plt.title("Probability of inclusion")
        plt.show()

    def plot_speedup(self, n_allow_miss=[0], normalize=False):
        """
        Plot the average number of papers that can be discarded safely.

        Arguments
        ---------
        n_allow_miss: list[int]
            A list of the number of allowed False Negatives.
        normalize: bool
            Normalize the output with the expected outcome (not correct).
        """
        legend_plt = []
        legend_name = []
        linestyles = ['-', '--', '-.', ':']

        for i, _dir in enumerate(self._dirs):
            for i_miss, n_miss in enumerate(n_allow_miss):
                speed_res = self.stat_test_merged(
                    _dir, "pool_proba", _speedup_merged,
                    n_allow_miss=n_miss, normalize=normalize)
                xr = self._n_queries[_dir]
                cur_avg = []
                cur_err = []
                for sp in speed_res:
                    cur_avg.append(sp[0])
                    cur_err.append(sp[1])
                col = "C"+str(i % 10)
                my_plot = plt.errorbar(xr, cur_avg, cur_err,
                                       color=col, capsize=4,
                                       ls=linestyles[i_miss % len(linestyles)])
                if n_miss == n_allow_miss[0]:
                    legend_plt.append(my_plot)
                    legend_name.append(f"{_dir}")

        plt.legend(legend_plt, legend_name, loc="upper left")
        plt.title("Articles that do not have to be read.")
        plt.show()

    def plot_limits(self, prob_allow_miss=[0.1]):
        legend_plt = []
        legend_name = []
        linestyles = ['-', '--', '-.', ':']

        for i, _dir in enumerate(self._dirs):
            for i_miss, p_miss in enumerate(prob_allow_miss):
                limits_res = self.stat_test_merged(
                    _dir, "pool_proba", _limits_merged,
                    p_allow_miss=p_miss)
                xr = self._n_queries[_dir]
                col = "C"+str(i % 10)
                my_plot, = plt.plot(xr, limits_res, color=col,
                                    ls=linestyles[i_miss % len(linestyles)])
                if i_miss == 0:
                    legend_plt.append(my_plot)
                    legend_name.append(f"{_dir}")

        plt.legend(legend_plt, legend_name, loc="upper left")
        plt.title("Articles that do not have to be read.")
        plt.show()

    