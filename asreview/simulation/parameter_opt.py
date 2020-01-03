import copy
import logging
import os
import pickle
import random
from distutils.dir_util import copy_tree
from multiprocessing import Process

from hyperopt import fmin, tpe, STATUS_OK, Trials
from modAL.models import ActiveLearner
import numpy as np
from numpy import average
from tqdm import tqdm

from asreview.analysis import Analysis
from asreview.balance_strategies.utils import get_balance_class
from asreview.models.utils import get_model_class
from asreview.query_strategies.utils import get_query_class
from asreview.review.factory import get_reviewer
from asreview.readers import ASReviewData


def loss_spread(time_results, n_papers, moment=1.0):
    loss = 0
    for label in time_results:
        loss += (time_results[label]/n_papers)**moment
    return (loss**(1/moment))/len(time_results)


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
    return WSS_x


def run_model(*args, model_name, balance_strategy, pid=0, **kwargs):
    reviewer = get_reviewer(*args, model=model_name,
                            balance_strategy=balance_strategy, **kwargs)
    rand_seed = pid

    np.random.seed(rand_seed)
    random.seed(rand_seed)

    reviewer.learner = ActiveLearner(
            estimator=reviewer.model,
            query_strategy=reviewer.query_strategy
    )
    reviewer.review()


def compute_args(dataname, dataset, included_sets,
                 excluded_sets, trials_dir, model, balance_strategy,
                 query_strategy, params, n_instances, n_papers,  i_run):
    log_dir = os.path.join(trials_dir, "current", dataname)
    os.makedirs(log_dir, exist_ok=True)

    logging.debug(f"params 2: {params}")
    params = copy.deepcopy(params)
    run_args = [dataset, "simulate"]
    run_kwargs = dict(
        query_strategy=query_strategy,
        n_instances=n_instances,
        n_papers=n_papers,
    )

#     run_kwargs["embedding_fp"] = os.path.splitext(dataset)[0]+".vec"
    run_kwargs["embedding_fp"] = os.path.splitext(dataset)[0]+".json"

    logging.debug(f"params 3: {params}")
    model_param = {}
    balance_param = {}
    query_param = {}
    for par in params:
        if par.startswith("mdl_"):
            model_param[par[4:]] = params[par]
        elif par.startswith("bal_"):
            balance_param[par[4:]] = params[par]
        elif par.startswith("qry_"):
            query_param[par[4:]] = params[par]
        else:
            run_kwargs[par] = params

    logging.debug(f"Balance 2: {balance_param}")
    run_kwargs["model_name"] = model
    run_kwargs["balance_strategy"] = balance_strategy
    run_kwargs["model_param"] = model_param
    run_kwargs["balance_param"] = balance_param
    run_kwargs["query_param"] = query_param
    run_kwargs["log_file"] = os.path.join(
        log_dir, f"results_{i_run}.h5")
    try:
        os.remove(run_kwargs["log_file"])
    except FileNotFoundError:
        pass

    run_kwargs["prior_included"] = included_sets[dataname][i_run]
    run_kwargs["prior_excluded"] = excluded_sets[dataname][i_run]
    run_kwargs["pid"] = i_run

    return run_args, run_kwargs, log_dir


def loss_from_dataset(dataname, dataset, included_sets, excluded_sets,
                      trials_dir, model, balance_strategy, params,
                      query_strategy, n_instances, n_papers, n_runs):
    procs = []
    for i_run in range(n_runs):
        run_args, run_kwargs, log_dir = compute_args(
            dataname, dataset, trials_dir, model, balance_strategy,
            query_strategy, params, n_instances, n_papers, included_sets,
            excluded_sets, i_run)

        run_kwargs["pid"] = i_run
        p = Process(
            target=run_model,
            args=copy.deepcopy(run_args),
            kwargs=copy.deepcopy(run_kwargs),
            daemon=False,
        )
        procs.append(p)

    for p in procs:
        p.start()

    for p in procs:
        p.join()

    analysis = Analysis.from_dir(log_dir)
    results = analysis.avg_time_to_discovery()
    loss = loss_spread(results, len(analysis.labels), 1.0)

    return loss


def simple_loss_aggregation(files, included_sets, excluded_sets, *args, **kwargs):
    loss = []
    for dataset in list(files.keys()):
        loss.append(
            loss_from_dataset(
                dataset, files[dataset], included_sets[dataset],
                excluded_sets[dataset],
                *args, **kwargs)
        )
    logging.debug(f"losses: {loss}")
    return {"loss": average(loss), 'status': STATUS_OK}


def create_objective_func(data_dir,
                          model,
                          balance_strategy,
                          query_strategy,
                          trials_dir,
                          loss_function=loss_from_dataset,
                          n_runs=8, n_included=1, n_excluded=1, n_papers=1502,
                          n_instances=50, **kwargs):

    files = {}
    excluded_sets = {}
    included_sets = {}

    trials_data_dir = os.path.join(trials_dir, "data")
    os.makedirs(trials_data_dir, exist_ok=True)

    base_files = os.listdir(trials_data_dir)
    if len(os.listdir(data_dir)) > len(base_files):
        copy_tree(data_dir, trials_data_dir)
        base_files = os.listdir(trials_data_dir)

    for file in base_files:
        if not os.path.isfile(os.path.join(trials_data_dir, file)):
            continue
        if not (file.endswith(".csv") or file.endswith(".ris")
                or file.endswith(".xlsx")):
            continue
        dataset = os.path.splitext(file)[0]
        files[dataset] = os.path.join(trials_data_dir, file)

    for dataset, data_fp in files.items():
        asdata = ASReviewData.from_file(data_fp)
        ones = np.where(asdata.labels == 1)[0]
        zeros = np.where(asdata.labels == 0)[0]

        np.random.seed(81276149)

        included_sets[dataset] = []
        excluded_sets[dataset] = []
        for _ in range(n_runs):
            included_sets[dataset].append(
                np.random.choice(ones, n_included, replace=False))
            excluded_sets[dataset].append(
                np.random.choice(zeros, n_excluded, replace=False))

    def objective_func(params):
        return loss_function(
            files, included_sets,
            excluded_sets, trials_dir, model,
            balance_strategy, query_strategy, params,
            n_instances,
            n_papers, n_runs=n_runs,
            **kwargs)

    return objective_func


def hyper_optimize(datadir="data",
                   model="svm",
                   balance_strategy="simple",
                   query_strategy="cluster",
                   n_iter=20,
                   trials_file="trials.pkl",
                   trials_dir="hyper_optimize",
                   loss_function=loss_from_dataset,
                   ):
    obj_fun = create_objective_func(datadir, model, balance_strategy,
                                    query_strategy,
                                    trials_dir, loss_function=loss_function)

    model = get_model_class(model)
    balance = get_balance_class(balance_strategy)
    query = get_query_class(query_strategy)
    hyper_space, hyper_names = model().hyper_space()
    hyper_space.update(balance().hyperopt_space())
    hyper_space.update(query().hyperopt_space())

    if len(hyper_space) == 0:
        print("Hyperparameter space is empty.")
        exit()

    trials = None
    trials_fp = os.path.join(trials_dir, trials_file)
    if trials_fp is not None:
        try:
            with open(trials_fp, "rb") as fp:
                trials = pickle.load(fp)
        except FileNotFoundError:
            print(f"Cannot find {trials_fp}")

    if trials is None:
        trials = Trials()
        n_start_evals = 0
    else:
        n_start_evals = len(trials.trials)

    for i in tqdm(range(n_iter)):
        fmin(fn=obj_fun,
             space=hyper_space,
             algo=tpe.suggest,
             max_evals=i+n_start_evals+1,
             trials=trials,
             show_progressbar=False)
        with open(trials_fp, "wb") as fp:
            pickle.dump(trials, fp)
        if trials.best_trial['tid'] == len(trials.trials)-1:
            copy_tree(os.path.join(trials_dir, "current"),
                      os.path.join(trials_dir, "best"))

    return trials, hyper_names
