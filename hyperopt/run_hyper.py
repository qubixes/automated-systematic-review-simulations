#!/usr/bin/env python

import argparse
import os
import sys


import logging
logging.getLogger().setLevel(logging.ERROR)

ABBREVIATIONS = {
    "triple_balance": "tb",
    "undersampling": "us",
    "naive_bayes": "nb",
    "cluster": "cl",
}


def _parse_arguments():
    parser = argparse.ArgumentParser(prog=sys.argv[0])
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="svm",
        help="Prediction model for active learning."
    )
    parser.add_argument(
        "-b", "--balance_strategy",
        type=str,
        default="simple",
        help="Balance strategy for active learning."
    )
    parser.add_argument(
        "-q", "--query_strategy",
        type=str,
        default="rand_max",
        help="Query strategy for active learning."
    )
    parser.add_argument(
        "-n", "--n_iter",
        type=int,
        default=1,
        help="Number of iterations of Bayesian Optimization."
    )
    parser.add_argument(
        "--mpi",
        dest='use_mpi',
        action='store_true',
        help="Use the mpi implementation.",
    )
    return parser


def main(model, balance_strategy, query_strategy, kwargs):
    import pandas
    from asreview.simulation.parameter_opt import hyper_optimize
    from pandas import DataFrame

    short_model = ABBREVIATIONS.get(model, model)
    short_bal = ABBREVIATIONS.get(balance_strategy, balance_strategy)
    short_query = ABBREVIATIONS.get(query_strategy, query_strategy)
    trials_dir = os.path.join("hyper_trials",
                              f"{model}_{balance_strategy}_{query_strategy}")

    use_mpi = kwargs.pop('use_mpi', False)
    if use_mpi:
        from mpi4py import MPI
        from asreview.simulation.mpiq import mpi_hyper_optimize

        rank = MPI.COMM_WORLD.Get_rank()
        trials, hyper_names = mpi_hyper_optimize(trials_dir=trials_dir, **kwargs)
        if rank != 0:
            return
    else:
        trials, hyper_names = hyper_optimize(trials_dir=trials_dir, **kwargs)

    results = trials.vals

    final_results = {}

    for full_name in results:
        replace_list = hyper_names.get(full_name, None)
        if replace_list is not None:
            for i in range(len(results[full_name])):
                results[full_name][i] = replace_list[results[full_name][i]]
        if full_name.startswith("mdl_"):
            new_name = short_model + "_" + full_name[4:]
        elif full_name.startswith("bal_"):
            new_name = short_bal + "_" + full_name[4:]
        elif full_name.startswith("qry_"):
            new_name = short_query + "_" + full_name[4:]
        elif full_name == "loss":
            new_name = "loss"
        else:
            new_name = "unknownsrc_" + full_name[4:]
        final_results[new_name] = results[full_name]

    final_results["loss"] = trials.losses()

    pandas.options.display.max_rows = 999
#     pandas.options.display.max_cols = 120
    pandas.options.display.width = 0
    print(DataFrame(final_results).sort_values("loss"))


if __name__ == "__main__":

    parser = _parse_arguments()
    args = vars(parser.parse_args(sys.argv[1:]))

    model = args["model"]
    balance_strategy = args["balance_strategy"]
    query_strategy = args["query_strategy"]

    main(model, balance_strategy, query_strategy, args)
