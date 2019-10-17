#!/usr/bin/env python

import argparse
import os
import sys


ABBREVIATIONS = {
    "triple_balance": "tb",
    "undersampling": "us",
    "naive_bayes": "nb",
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
        "-n", "--n_iter",
        type=int,
        default=1,
        help="Number of iterations of Bayesian Optimization."
    )
    return parser


def main(model, balance_strategy, kwargs):
    import pandas
    from asreview.simulation.parameter_opt import hyper_optimize
    from pandas import DataFrame

    short_model = ABBREVIATIONS.get(model, model)
    short_bal = ABBREVIATIONS.get(balance_strategy, balance_strategy)
    trials_dir = os.path.join("hyper_trials", f"{model}_{balance_strategy}")

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
        elif full_name == "loss":
            new_name = "loss"
        else:
            new_name = "unknownsrc_" + full_name[4:]
        final_results[new_name] = results[full_name]

    final_results["loss"] = trials.losses()

    pandas.options.display.max_rows = 999
    print(DataFrame(final_results).sort_values("loss"))


if __name__ == "__main__":

    parser = _parse_arguments()
    args = vars(parser.parse_args(sys.argv[1:]))

    model = args["model"]
    balance_strategy = args["balance_strategy"]

    main(model, balance_strategy, args)
