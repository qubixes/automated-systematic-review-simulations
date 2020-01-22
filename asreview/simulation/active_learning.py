#!/usr/bin/env python

import sys
import argparse


from asreview.simulation.mpi_executor import mpi_executor, mpi_hyper_optimize
from asreview.simulation.serial_executor import serial_executor
from asreview.simulation.serial_executor import serial_hyper_optimize
from asreview.simulation.job_utils import get_data_names
from asreview.simulation.al_job import ActiveLearnJobRunner
from asreview.entry_points.base import BaseEntryPoint


class HyperActiveEntryPoint(BaseEntryPoint):
    description = "Hyper parameter optimization for active learning."

    def execute(self, argv):
        main(argv)


def _parse_arguments():
    parser = argparse.ArgumentParser(prog=sys.argv[0])
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="nb",
        help="Prediction model for active learning."
    )
    parser.add_argument(
        "-q", "--query_strategy",
        type=str,
        default="max_random",
        help="Query strategy for active learning."
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
        default="tfidf",
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


def main(argv=sys.argv[1:]):
    parser = _parse_arguments()
    args = vars(parser.parse_args(argv))
    datasets = args["datasets"].split(",")
    model_name = args["model"]
    feature_name = args["feature_extraction"]
    balance_name = args["balance_strategy"]
    query_name = args["query_strategy"]
    n_iter = args["n_iter"]
    use_mpi = args["use_mpi"]

    data_names = get_data_names(datasets)
    if use_mpi:
        executor = mpi_executor
    else:
        executor = serial_executor

    job_runner = ActiveLearnJobRunner(
        data_names, model_name=model_name, query_name=query_name,
        balance_name=balance_name, feature_name=feature_name,
        executor=executor)

    if use_mpi:
        mpi_hyper_optimize(job_runner, n_iter)
    else:
        serial_hyper_optimize(job_runner, n_iter)


if __name__ == "__main__":
    main()
