import os
import tempfile
from multiprocessing import Process
import random
import logging
import copy

from modAL.models import ActiveLearner
import numpy as np

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from asreview.review.factory import get_reviewer
from asreview.simulation.analysis import Analysis
from asreview.models.sklearn_models import create_svc_model
import pickle
from asreview.readers import ASReviewData
from numpy import average


SVM_KERNELS = ['poly', 'rbf', 'sigmoid', 'linear']
BALANCE_STRATS = ['simple', 'undersample', 'triple_balance']
SVM_GAMMA = ['scale', 'auto']


def loss_integrated(inc_results):
    inc_data = inc_results["data"]
    loss = 0
    x_reviewed = inc_data[0]
    y_reviewed = inc_data[1]
    dx = (inc_data[0][1] - inc_data[0][0])/100
    for y_val in y_reviewed:
        loss -= dx*y_val/100

    dx = (1-x_reviewed[-1]/100)
    dy = (1-y_reviewed[-1]/100)
    dy_avg = 1-dy/2
    loss -= dx * dy_avg
    return loss


def loss_WSS(inc_results):
    keys = []
    for key in inc_results:
        if key.startswith("WSS"):
            keys.append(key)

    loss = 0
    for key in keys:
        loss += loss_single_WSS(inc_results, key)/len(keys)
    return loss


def loss_single_WSS(inc_results, WSS_measure):
    inc_data = inc_results["data"]
    WSS_y = int(WSS_measure[3:])/100
    WSS_x = inc_results[WSS_measure]
    if WSS_x is not None:
        return WSS_x[1]
    last_x = inc_data[0][-1]
    last_y = inc_data[1][-1]
    b = (1-last_y)/(1-last_x)
    a = 1 - b
    if WSS_y > 0.99999:
        WSS_y = 0.99999
    WSS_x = (WSS_y - a)/b
#     print(a, b, WSS_x, last_x, last_y)
    return WSS_x


def run_model(params, *args, pid=0, **kwargs):
    reviewer = get_reviewer(*args, **kwargs)
    rand_seed = pid

    class_weight = params.pop("class_weight", None)
    if class_weight is not None:
        params["class_weight"] = {
            0: 1,
            1: class_weight,
        }

    np.random.seed(rand_seed)
    random.seed(rand_seed)
    reviewer.model = create_svc_model(**params, random_state=rand_seed)
    reviewer.learner = ActiveLearner(
            estimator=reviewer.model,
            query_strategy=reviewer.query_strategy
    )
    reviewer.review()


def loss_from_dataset(dataname, dataset, params, query_strategy, n_instances, n_papers,
                      n_runs, included_sets, excluded_sets, WSS_measures,
                      **kwargs):
    log_dir = os.path.join("temp", dataname)
    os.makedirs(log_dir, exist_ok=True)

    params = copy.deepcopy(params)
    run_args = [params, dataset, "simulate", "svm"]
    run_kwargs = dict(
        query_strategy=query_strategy,
        n_instances=n_instances,
        n_papers=n_papers,
        **kwargs
    )

    if 'balance_strategy' in params:
        run_kwargs['balance_strategy'] = params.pop('balance_strategy')

    procs = []
    for i_run in range(n_runs):
        run_kwargs["log_file"] = os.path.join(
            log_dir, f"results_{i_run}.json")
        run_kwargs["prior_included"] = included_sets[i_run]
        run_kwargs["prior_excluded"] = excluded_sets[i_run]
        run_kwargs["pid"] = i_run
        p = Process(
            target=run_model,
            args=copy.deepcopy(run_args),
            kwargs=copy.deepcopy(run_kwargs),
            daemon=True,
        )
        procs.append(p)

    for p in procs:
        p.start()

    for p in procs:
        p.join()
#         print(results)
#         os.remove(log_file)
#         os.rmdir(log_dir)
    analysis = Analysis.from_dir(log_dir)
    results = analysis.get_inc_found(WSS_measures=WSS_measures)
    if len(WSS_measures) > 0:
        loss = loss_WSS(results)
    else:
        loss = loss_integrated(results)
    return loss


def create_objective_func(datasets, WSS_measures=[95, 98, 100],
                          query_strategy="rand_max",
                          n_runs=8, n_included=10, n_excluded=10, n_papers=520,
                          n_instances=50, **kwargs):

    files = {}
    excluded_sets = {}
    included_sets = {}
    for dataset in datasets:
        files[dataset] = os.path.join("..", "..", "data", "test", dataset+".csv")
        asdata = ASReviewData.from_file(files[dataset])
        ones = np.where(asdata.labels == 1)[0]
        zeros = np.where(asdata.labels == 0)[0]

        np.random.seed(81276149)

        included_sets[dataset] = []
        excluded_sets[dataset] = []
        for _ in range(n_runs):
            included_sets[dataset].append(np.random.choice(ones, n_included, replace=False))
            excluded_sets[dataset].append(np.random.choice(zeros, n_excluded, replace=False))

    def objective_func(params):
        loss = []
        for dataset in datasets:
            loss.append(
                loss_from_dataset(
                    dataset, files[dataset], params, query_strategy, n_instances,
                    n_papers, n_runs, included_sets[dataset], excluded_sets[dataset],
                    WSS_measures, **kwargs)
            )

        return {"loss": average(loss), 'status': STATUS_OK}
    return objective_func


def optimize_svm(datasets=["ptsd", "ace", "hall"],
                 max_evals=20,
                 trials_fp=None):
    obj_fun = create_objective_func(datasets)
    param_space = {
        'C': hp.lognormal('svm_C', 0, 2),
        'class_weight': hp.lognormal('class_weight', 0, 1),
        'kernel': hp.choice('kernel', SVM_KERNELS),
        'gamma': hp.choice('svm_gamma', SVM_GAMMA),
        'balance_strategy': hp.choice('balance_strategy', BALANCE_STRATS),
    }
    trials = None
    if trials_fp is not None:
        try:
            with open(trials_fp, "rb") as fp:
                trials = pickle.load(fp)
        except FileNotFoundError:
            print(f"Cannot find {trials_fp}")

    if trials is None:
        trials = Trials()

    best = fmin(fn=obj_fun,
                space=param_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)
    best['kernel'] = SVM_KERNELS[best['kernel']]
    if 'gamma' in best:
        best['gamma'] = SVM_GAMMA[best['gamma']]
    if 'balance_strategy' in best:
        best['balance_strategy'] = BALANCE_STRATS[best['balance_strategy']]
#     except KeyboardInterrupt:
#         pass
    with open(trials_fp, "wb") as fp:
        pickle.dump(trials, fp)

    return best
