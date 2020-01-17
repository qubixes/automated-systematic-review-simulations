import os
from os.path import join, isfile
import json
import pickle
from distutils.dir_util import copy_tree

from hyperopt import STATUS_OK, Trials, fmin, tpe
import numpy as np
from tqdm import tqdm

from asreview.simulation.serial_executor import serial_executor
from asreview.readers import ASReviewData
from asreview.models.utils import get_model_class
from asreview.feature_extraction.utils import get_feature_class
from asreview.balance_strategies.utils import get_balance_class


def empty_shared():
    return {
        "query_src": {},
        "current_queries": {}
    }


def quality(result_list, alpha=1):
    q = 0
    for _, rank in result_list:
        q += rank**alpha

    return (q/len(result_list))**(1/alpha)


def loss_from_files(data_fps, labels_fp):
    with open(labels_fp, "r") as fp:
        labels = np.array(json.load(fp), dtype=int)
    results = {}
    for data_fp in data_fps:
        with open(data_fp, "r") as fp:
            data = json.load(fp)
        train_idx = np.array(data["train_idx"])
        proba = np.array(data["proba"])
        test_idx = np.delete(np.arange(len(labels)), train_idx)
        proba_test = [
            (idx, -proba[idx]) for idx in test_idx]
        proba_test = sorted(proba_test, key=lambda x: x[1])
        for position, item in enumerate(proba_test):
            idx = item[0]
            if labels[idx] == 1:
                if idx not in results:
                    results[idx] = [0, 0]
                results[idx][0] += position
                results[idx][1] += 1

    result_list = []
    for key, item in results.items():
        new_value = item[0]/(item[1]*(len(labels)-len(train_idx)))
        result_list.append([int(key), new_value])

    result_list = sorted(result_list, key=lambda x: x[1])

    return quality(result_list, 1.0)


def get_trial_fp(datasets, model_name, balance_name, feature_name):
    trials_dir = join("output", "inactive", "_".join(
        [model_name, balance_name, feature_name]), "_".join(datasets))
    os.makedirs(trials_dir, exist_ok=True)
    trials_fp = os.path.join(trials_dir, f"trials.pkl")

    return trials_dir, trials_fp


def create_jobs(param, data_names, n_run):
    jobs = []
    for data_name in data_names:
        for i_run in range(n_run):
            jobs.append({
                "param": param,
                "data_name": data_name,
                "i_run": i_run,
                })
    return jobs


def compute_train_idx(y, seed):
    np.random.seed(seed)
    one_idx = np.where(y == 1)[0]
    zero_idx = np.where(y == 0)[0]

    n_zero_train = round(0.75*len(zero_idx))
    n_one_train = round(0.75*len(one_idx))
    train_one_idx = np.random.choice(one_idx, n_one_train, replace=False)
    train_zero_idx = np.random.choice(zero_idx, n_zero_train, replace=False)
    train_idx = np.append(train_one_idx, train_zero_idx)
    return train_idx


def _get_prefix_param(raw_param, prefix):
    return {key[4:]: value for key, value in raw_param.items()
            if key[:4] == prefix}


def get_split_param(raw_param):
    return {
        "model_param": _get_prefix_param(raw_param, "mdl_"),
        "query_param": _get_prefix_param(raw_param, "qry_"),
        "balance_param": _get_prefix_param(raw_param, "bal_"),
        "feature_param": _get_prefix_param(raw_param, "fex_"),
    }


def data_fp_from_name(data_dir, data_name):
    file_list = os.listdir(data_dir)
    file_list = [file_name for file_name in file_list
                 if file_name.endswith((".csv", ".xlsx", ".ris")) and
                 os.path.splitext(file_name)[0] == data_name]
    return join(data_dir, file_list[0])


def get_out_dir(trials_dir, data_name):
    out_dir = join(trials_dir, "current", data_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def get_out_fp(trials_dir, data_name, i_run):
    return join(get_out_dir(trials_dir, data_name),
                f"results_{i_run}.json")


def get_label_fp(trials_dir, data_name):
    return join(get_out_dir(trials_dir, data_name), "labels.json")


class InactiveJobRunner():
    def __init__(self, data_names, model_name, balance_name, feature_name,
                 executor=serial_executor, n_run=10):

        self.trials_dir, self.trials_fp = get_trial_fp(
            data_names, model_name, balance_name, feature_name)

        self.model_class = get_model_class(model_name)
        self.feature_class = get_feature_class(feature_name)
        self.balance_class = get_balance_class(balance_name)

        self.data_names = data_names
        self.executor = executor
        self.n_run = n_run
        self.data_dir = "data"
        self._cache = {data_name: {"train_idx": {}}
                       for data_name in data_names}

    def create_loss_function(self):
        def objective_func(param):
            jobs = create_jobs(param, self.data_names, self.n_run)

            self.executor(jobs, self)
            losses = []
            for data_name in self.data_names:
                label_fp = get_label_fp(self.trials_dir, data_name)
                res_files = [get_out_fp(self.trials_dir, data_name, i_run)
                             for i_run in range(self.n_run)]
                losses.append(loss_from_files(res_files, label_fp))
            return {"loss": np.average(losses), 'status': STATUS_OK}

        return objective_func

    def execute(self, param, data_name, i_run):
        split_param = get_split_param(param)
        model = self.model_class(**split_param["model_param"])
        balance_model = self.balance_class(**split_param["balance_param"])
        feature_model = self.feature_class(**split_param["feature_param"])

        as_data = self.get_cached_as_data(data_name)
        train_idx = self.get_cached_train_idx(data_name, i_run)
        out_fp = get_out_fp(self.trials_dir, data_name, i_run)

        np.random.seed(i_run)
        X = feature_model.fit_transform(
            as_data.texts, as_data.title, as_data.abstract)
        X_train, y_train = balance_model.sample(
                X, as_data.labels, train_idx, empty_shared())
        model.fit(X_train, y_train)
        proba = model.predict_proba(X)[:, 1]

        with open(out_fp, "w") as fp:
            json.dump(
                {"proba": proba.tolist(), "train_idx": train_idx.tolist()},
                fp)

        label_fp = get_label_fp(self.trials_dir, data_name)
        if i_run == 0 and not isfile(label_fp):
            with open(label_fp, "w") as fp:
                json.dump(as_data.labels.tolist(), fp)

    def get_cached_as_data(self, data_name):
        try:
            return self._cache[data_name]["as_data"]
        except KeyError:
            pass
        data_fp = data_fp_from_name(self.data_dir, data_name)
        as_data = ASReviewData.from_file(data_fp)
        self._cache[data_name]["as_data"] = as_data
        return as_data

    def get_cached_train_idx(self, data_name, i_run):
        try:
            return self._cache[data_name]["train_idx"][i_run]
        except KeyError:
            pass

        as_data = self.get_cached_as_data(data_name)
        train_idx = compute_train_idx(as_data.labels, i_run)
        self._cache[data_name]["train_idx"][i_run] = train_idx
        return train_idx

    def get_hyper_space(self):
        model_hs, model_hc = self.model_class().hyper_space()
        balance_hs, balance_hc = self.balance_class().hyper_space()
        feature_hs, feature_hc = self.feature_class().hyper_space()
        hyper_space = {**model_hs, **balance_hs, **feature_hs}
        hyper_choices = {**model_hc, **balance_hc, **feature_hc}
        return hyper_space, hyper_choices

    def hyper_optimize(self, n_iter):
        obj_function = self.create_loss_function()
        hyper_space, hyper_choices = self.get_hyper_space()

        try:
            with open(self.trials_fp, "rb") as fp:
                trials, _ = pickle.load(fp)
        except FileNotFoundError:
            trials = None
            print(f"Creating new hyper parameter optimization run: "
                  f"{self.trials_fp}")

        if trials is None:
            trials = Trials()
            n_start_evals = 0
        else:
            n_start_evals = len(trials.trials)

        for i in tqdm(range(n_iter)):
            fmin(fn=obj_function,
                 space=hyper_space,
                 algo=tpe.suggest,
                 max_evals=i+n_start_evals+1,
                 trials=trials,
                 show_progressbar=False)
            with open(self.trials_fp, "wb") as fp:
                pickle.dump((trials, hyper_choices), fp)
            if trials.best_trial['tid'] == len(trials.trials)-1:
                copy_tree(os.path.join(self.trials_dir, "current"),
                          os.path.join(self.trials_dir, "best"))
