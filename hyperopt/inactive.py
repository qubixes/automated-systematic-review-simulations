#!/usr/bin/env python

import os
import pickle
import sys
from os.path import splitext, basename, join
import argparse
from distutils.dir_util import copy_tree
import json

import numpy as np
from tqdm import tqdm

import asreview
from hyperopt import STATUS_OK, Trials, fmin, tpe
from asreview.models.utils import get_model_class
from asreview.feature_extraction.utils import get_feature_class
from asreview.balance_strategies.utils import get_balance_class
from asreview.simulation.mpi_executor import mpi_executor, mpi_hyper_optimize
from asreview.simulation.serial_executor import serial_executor,\
    serial_hyper_optimize
# from asreview.simulation import mpi_executor
from inactive_job import InactiveJobRunner


def _parse_arguments():
    parser = argparse.ArgumentParser(prog=sys.argv[0])
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="dense_nn",
        help="Prediction model for active learning."
    )
    parser.add_argument(
        "-b", "--balance_strategy",
        type=str,
        default="simple",
        help="Balance strategy for active learning."
    )
    parser.add_argument(
        "-e", "--feature_extraction",
        type=str,
        default="doc2vec",
        help="Feature extraction method.")
    parser.add_argument(
        "-n", "--n_iter",
        type=int,
        default=1,
        help="Number of iterations of Bayesian Optimization."
    )
    parser.add_argument(
        "-d", "--datasets",
        type=str,
        default="all",
        help="Datasets to use in the hyper parameter optimization "
        "Separate by commas to use multiple at the same time [default: all].",
    )
    parser.add_argument(
        "--mpi",
        dest='use_mpi',
        action='store_true',
        help="Use the mpi implementation.",
    )
    return parser


# def _get_prefix_param(raw_param, prefix):
#     return {key[4:]: value for key, value in raw_param.items()
#             if key[:4] == prefix}
# 
# 
# def get_split_param(raw_param):
#     return {
#         "model_param": _get_prefix_param(raw_param, "mdl_"),
#         "query_param": _get_prefix_param(raw_param, "qry_"),
#         "balance_param": _get_prefix_param(raw_param, "bal_"),
#         "feature_param": _get_prefix_param(raw_param, "fex_"),
#     }


# def get_trial_fp(datasets, model_name, balance_name, feature_name):
#     trials_dir = join("output", "inactive", "_".join(
#         [model_name, balance_name, feature_name]), "_".join(datasets))
#     os.makedirs(trials_dir, exist_ok=True)
#     trials_fp = os.path.join(trials_dir, f"trials.pkl")
# 
#     return trials_dir, trials_fp
# 
# 
# def get_data_fps(datasets):
#     data_dir = "data"
#     file_list = os.listdir(data_dir)
#     file_list = [file_name for file_name in file_list
#                  if file_name.endswith((".csv", ".xlsx", ".ris"))]
#     if "all" not in datasets:
#         file_list = [file_name for file_name in file_list
#                      if splitext(file_name)[0] in datasets]
#     return [join(data_dir, file_name) for file_name in file_list]


def get_data_names(datasets):
    data_dir = "data"
    file_list = os.listdir(data_dir)
    file_list = [file_name for file_name in file_list
                 if file_name.endswith((".csv", ".xlsx", ".ris"))]
    if "all" not in datasets:
        file_list = [file_name for file_name in file_list
                     if splitext(file_name)[0] in datasets]
    return [splitext(file_name)[0] for file_name in file_list]
 
# 
# def quality(result_list, alpha=1):
#     q = 0
#     for _, rank in result_list:
#         q += rank**alpha
# 
#     return (q/len(result_list))**(1/alpha)


# def empty_shared():
#     return {
#         "query_src": {},
#         "current_queries": {}
#     }


# def test_inactive2(data, model_class, balance_class, feature_class,
#                    split_param, train_idx, out_fp):
#     model = model_class(**split_param["model_param"])
#     balance_model = balance_class(**split_param["balance_param"])
#     feature_model = feature_class(**split_param["feature_param"])
#     X = feature_model.fit_transform(
#         data["texts"], data["titles"], data["abstracts"])
#     X_train, y_train = balance_model.sample(
#             X, data["labels"], train_idx, empty_shared())
#     model.fit(X_train, y_train)
#     proba = model.predict_proba(X)[:, 1]
#     with open(out_fp, "w") as fp:
#         json.dump(
#             {"proba": proba.tolist(), "train_idx": train_idx.tolist()},
#             fp)


# def loss_from_files(data_fps, labels_fp):
#     with open(labels_fp, "r") as fp:
#         labels = np.array(json.load(fp), dtype=int)
#     results = {}
#     for data_fp in data_fps:
#         with open(data_fp, "r") as fp:
#             data = json.load(fp)
#         train_idx = np.array(data["train_idx"])
#         proba = np.array(data["proba"])
#         test_idx = np.delete(np.arange(len(labels)), train_idx)
#         proba_test = [
#             (idx, -proba[idx]) for idx in test_idx]
#         proba_test = sorted(proba_test, key=lambda x: x[1])
#         for position, item in enumerate(proba_test):
#             idx = item[0]
#             if labels[idx] == 1:
#                 if idx not in results:
#                     results[idx] = [0, 0]
#                 results[idx][0] += position
#                 results[idx][1] += 1
# 
#     result_list = []
#     for key, item in results.items():
#         new_value = item[0]/(item[1]*(len(labels)-len(train_idx)))
#         result_list.append([int(key), new_value])
# 
#     result_list = sorted(result_list, key=lambda x: x[1])
# 
#     return quality(result_list, 1.0)


# def compute_train_idx(y):
#     one_idx = np.where(y == 1)[0]
#     zero_idx = np.where(y == 0)[0]
# 
#     n_zero_train = round(0.75*len(zero_idx))
#     n_one_train = round(0.75*len(one_idx))
#     train_one_idx = np.random.choice(one_idx, n_one_train, replace=False)
#     train_zero_idx = np.random.choice(zero_idx, n_zero_train, replace=False)
#     train_idx = np.append(train_one_idx, train_zero_idx)
#     return train_idx


# def test_inactive(model, balance_model, X, y, out_fp, n_run=100):
# 
#     one_idx = np.where(y == 1)[0]
#     zero_idx = np.where(y == 0)[0]
# 
#     n_zero_train = round(0.75*len(zero_idx))
#     n_one_train = round(0.75*len(one_idx))
#     n_train = n_zero_train+n_one_train
# 
#     results = {}
#     for _ in range(n_run):
#         train_one_idx = np.random.choice(
#             one_idx, n_one_train, replace=False)
#         train_zero_idx = np.random.choice(
#             zero_idx, n_zero_train, replace=False)
#         train_idx = np.append(train_one_idx, train_zero_idx)
#         test_idx = np.delete(np.arange(len(y)), train_idx)
#         X_train, y_train = balance_model.sample(
#             X, y, train_idx, empty_shared())
#         model.fit(X_train, y_train)
# 
#         proba_test = model.predict_proba(X[test_idx])[:, 1]
#         proba_test = [
#             (test_idx[idx], -proba) for idx, proba in enumerate(proba_test)]
#         proba_test = sorted(proba_test, key=lambda x: x[1])
# 
#         for position, item in enumerate(proba_test):
#             idx = item[0]
#             if y[idx] == 1:
#                 if idx not in results:
#                     results[idx] = [0, 0]
#                 results[idx][0] += position
#                 results[idx][1] += 1
# 
#     result_list = []
#     for key, item in results.items():
#         new_value = item[0]/(item[1]*(len(y)-n_train))
#         result_list.append([int(key), new_value])
# 
#     result_list = sorted(result_list, key=lambda x: x[1])
# 
#     with open(out_fp, "w") as fp:
#         json.dump(result_list, fp)
# 
#     return quality(result_list, 1.0)

# 
# def create_job(param, data_list, train_idx_list, data_name, i_run,
#                current_dir, **kwargs):
#     job = {
#         "split_param": get_split_param(param),
#         "data": data_list[data_name],
#         "train_idx": train_idx_list[data_name][i_run],
#         "out_fp": join(current_dir, data_name,
#                        f"result_{i_run}.json")
#     }
#     job.update(kwargs)
#     return job

# 
# def create_jobs(param, data_list, current_dir, train_idx_list, model_class,
#                 balance_class, feature_class, n_run, ):
#     all_jobs = []
#     res_files = {}
#     label_files = {}
#     for data_name in data_list:
#         labels_fp = join(current_dir, data_name, "labels.json")
#         label_files[data_name] = labels_fp
#         res_files[data_name] = []
#         with open(labels_fp, "w") as fp:
#             json.dump(data_list[data_name]["labels"].tolist(), fp)
#         for i_run in range(n_run):
#             job = create_job(param, data_list, train_idx_list, data_name,
#                              i_run, current_dir, model_class=model_class,
#                              balance_class=balance_class,
#                              feature_class=feature_class)
#             res_files[data_name].append(job["out_fp"])
#             all_jobs.append({"run_kwargs": job, "run_args": []})
#     return all_jobs, res_files, label_files


# def create_objective_func(data_fps, trials_dir, model_class, balance_class,
#                           feature_class, executor=serial_executor, n_run=10):
#     current_dir = join(trials_dir, "current")
#     data_list = {}
#     train_idx_list = {}
#     for data_fp in data_fps:
#         data_name = splitext(basename(data_fp))[0]
#         as_data = asreview.ASReviewData.from_file(data_fp)
#         _, texts, labels = as_data.get_data()
#         data_list[data_name] = {
#             "texts": texts,
#             "labels": labels,
#             "titles": as_data.title,
#             "abstracts": as_data.abstract,
#         }
# 
#         train_idx_list[data_name] = []
#         np.random.seed(123987912)
#         for _ in range(n_run):
#             train_idx_list[data_name].append(compute_train_idx(labels))
# 
#         os.makedirs(join(current_dir, data_name), exist_ok=True)
# 
#     assert len(data_list) > 0
# 
#     def objective_func(param):
#         jobs, res_files, label_files = create_jobs(
#             param, data_list, current_dir, train_idx_list, model_class,
#             balance_class, feature_class, n_run)
# 
#         executor(jobs, job_function=test_inactive2)
#         losses = []
#         for data_name in data_list:
#             losses.append(
#                 loss_from_files(res_files[data_name], label_files[data_name]))
# 
#         return {"loss": np.average(losses), 'status': STATUS_OK}
#     return objective_func


# def hyper_optimize(datasets, data_fps, model_name, balance_name, feature_name,
#                    n_iter, executor=serial_executor):
#     trials_dir, trials_fp = get_trial_fp(
#         datasets, model_name, balance_name, feature_name)
# 
#     model_class = get_model_class(model_name)
#     feature_class = get_feature_class(feature_name)
#     balance_class = get_balance_class(balance_name)
# 
#     obj_function = create_objective_func(
#         data_fps, trials_dir, model_class, balance_class, feature_class,
#         n_run=10, executor=executor)
# 
#     model_hyper_space, model_hyper_choices = model_class().hyper_space()
#     balance_hyper_space, balance_hyper_choices = balance_class().hyper_space()
#     feature_hyper_space, feature_hyper_choices = feature_class().hyper_space()
#     hyper_space = {**model_hyper_space, **balance_hyper_space,
#                    **feature_hyper_space}
#     hyper_choices = {**model_hyper_choices, **balance_hyper_choices,
#                      **feature_hyper_choices}
# 
#     try:
#         with open(trials_fp, "rb") as fp:
#             trials, _ = pickle.load(fp)
#     except FileNotFoundError:
#         trials = None
#         print(f"Creating new hyper parameter optimization run: {trials_fp}")
# 
#     if trials is None:
#         trials = Trials()
#         n_start_evals = 0
#     else:
#         n_start_evals = len(trials.trials)
# 
#     for i in tqdm(range(n_iter)):
#         fmin(fn=obj_function,
#              space=hyper_space,
#              algo=tpe.suggest,
#              max_evals=i+n_start_evals+1,
#              trials=trials,
#              show_progressbar=False)
#         with open(trials_fp, "wb") as fp:
#             pickle.dump((trials, hyper_choices), fp)
#         if trials.best_trial['tid'] == len(trials.trials)-1:
#             copy_tree(os.path.join(trials_dir, "current"),
#                       os.path.join(trials_dir, "best"))


if __name__ == "__main__":
    parser = _parse_arguments()
    args = vars(parser.parse_args(sys.argv[1:]))
    datasets = args["datasets"].split(",")
#     data_fps = get_data_fps(datasets)
    model_name = args["model"]
    feature_name = args["feature_extraction"]
    balance_name = args["balance_strategy"]
    n_iter = args["n_iter"]
    use_mpi = args["use_mpi"]

#     hyper_args = [datasets, data_fps, model_name, balance_name, feature_name,
#                   n_iter]

    data_names = get_data_names(datasets)
    if use_mpi:
        executor = mpi_executor
    else:
        executor = serial_executor

    job_runner = InactiveJobRunner(data_names, model_name, balance_name,
                                   feature_name, executor=executor)

    if use_mpi:
        mpi_hyper_optimize(job_runner, n_iter)
    else:
        serial_hyper_optimize(job_runner, n_iter)
# 
#     trials_dir, trials_fp = get_trial_fp(
#         datasets, model_name, balance_name, feature_name)
# 
#     model_class = get_model_class(model_name)
#     feature_class = get_feature_class(feature_name)
#     balance_class = get_balance_class(balance_name)
# 
#     obj_function = create_objective_func(
#         data_fps, trials_dir, model_class, balance_class, feature_class,
#         n_run=10)
# 
#     model_hyper_space, model_hyper_choices = model_class().hyper_space()
#     balance_hyper_space, balance_hyper_choices = balance_class().hyper_space()
#     feature_hyper_space, feature_hyper_choices = feature_class().hyper_space()
#     hyper_space = {**model_hyper_space, **balance_hyper_space,
#                    **feature_hyper_space}
#     hyper_choices = {**model_hyper_choices, **balance_hyper_choices,
#                      **feature_hyper_choices}
# 
#     try:
#         with open(trials_fp, "rb") as fp:
#             trials, _ = pickle.load(fp)
#     except FileNotFoundError:
#         trials = None
#         print(f"Creating new hyper parameter optimization run: {trials_fp}")
# 
#     if trials is None:
#         trials = Trials()
#         n_start_evals = 0
#     else:
#         n_start_evals = len(trials.trials)
# 
#     for i in tqdm(range(n_iter)):
#         fmin(fn=obj_function,
#              space=hyper_space,
#              algo=tpe.suggest,
#              max_evals=i+n_start_evals+1,
#              trials=trials,
#              show_progressbar=False)
#         with open(trials_fp, "wb") as fp:
#             pickle.dump((trials, hyper_choices), fp)
#         if trials.best_trial['tid'] == len(trials.trials)-1:
#             copy_tree(os.path.join(trials_dir, "current"),
#                       os.path.join(trials_dir, "best"))
