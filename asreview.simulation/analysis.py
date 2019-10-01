'''
Analysis and reading of log files.

Merged versions of functions work on the results of all files at the same time.
'''

import os
import re
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc


class Analysis(object):
    """ Analysis object to plot things from the logs. """

    def __init__(self, data_dir):
        self._dirs = []
        self.labels = None
        self.empty = True

        self.data_dir = data_dir
        self.dir_key = os.path.basename(os.path.normpath(data_dir))
        self.results = read_json_results(data_dir)
        if len(self.results) == 0:
            return

        print(self.results)
        self._first_file = self._results.keys()[0]
        self._rores = reorder_results(self._results)
        self._n_queries = get_num_queries(self._results)
        self.labels = self._results[self._first_file].get('labels', None)
        self.empty = False

    @classmethod
    def from_dir(cls, data_dir):
        analysis_inst = Analysis(data_dir)
        if analysis_inst.empty:
            return None
        return analysis_inst

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

    def stat_test_merged(self, _dir, logname, stat_fn, final_labels=False, **kwargs):
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
        results = self._rores[_dir][logname]
#         print(self._results[_dir])
        if final_labels and self._final_labels is not None:
            labels = self._final_labels[_dir]
        else:
            labels = self._labels[_dir]
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

    def get_inc_found(self, _dir, *args, final_labels=False, **kwargs):
        inc_found = self.stat_test_merged(_dir, *args, final_labels=final_labels, **kwargs)
        cur_inc_found = []
        cur_inc_found_err = []
        if final_labels:
            labels = self._final_labels[_dir]
        else:
            labels = self._labels[_dir]

        xr = self._n_queries[_dir]
        for inc_data in inc_found:
            cur_inc_found.append(inc_data[0])
            cur_inc_found_err.append(inc_data[1])

        dy = cur_inc_found[0]
        dx = self._n_queries[_dir][0]
        x_norm = (len(labels)-dx)/100
        y_norm = (np.sum(labels)-dy)/100

        norm_xr = (np.array(xr)-dx)/x_norm
        norm_yr = (np.array(cur_inc_found)-dy)/y_norm
        norm_y_err = cur_inc_found_err/y_norm
        return [norm_xr, norm_yr, norm_y_err]

    def plot_inc_found(self, out_fp=None):
        """
        Plot the number of queries that turned out to be included
        in the final review.
        """
        legend_name = []
        legend_plt = []
        pool_name = "pool_proba"
        res_dict = {}

        for i, _dir in enumerate(self._dirs):
            inc_found = self.get_inc_found(_dir, pool_name,
                                           _inc_queried_merged)
            final_avail = self._final_labels is not None and _dir in self._final_labels
            if final_avail:
                inc_found_final = self.get_inc_found(
                    _dir, pool_name, _inc_queried_merged, final_labels=True)
#             res_dict[_dir] = self.stat_test_merged(_dir, pool_name,
#                                                    _inc_queried_all_merged)
            col = "C"+str(i % 10)

            myplot = plt.errorbar(*inc_found, color=col)
            legend_name.append(f"{_dir}")
            if final_avail:
                plt.errorbar(*inc_found_final, color=col, ls="--")
            legend_plt.append(myplot)

#         print(res_dict)
#         if out_fp is not None:
#             with open(out_fp, "w") as f:
#                 json.dump(res_dict, f)

        plt.legend(legend_plt, legend_name, loc="upper left")
        symb = "%"

        plt.xlabel(f"{symb} Queries")
        plt.ylabel(f"< {symb} Inclusions queried >")
        plt.title("Average number of inclusions found")
        plt.grid()
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

    