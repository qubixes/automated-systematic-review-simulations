#!/usr/bin/env python

import sys
import argparse


from asreview.simulation.mpi_executor import mpi_executor, mpi_hyper_optimize
from asreview.simulation.serial_executor import serial_executor
from asreview.simulation.serial_executor import serial_hyper_optimize
from asreview.simulation.job_utils import get_data_names
from asreview.simulation.cluster_job import ClusterJobRunner


def _parse_arguments():
    parser = argparse.ArgumentParser(prog=sys.argv[0])
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


if __name__ == "__main__":
    parser = _parse_arguments()
    args = vars(parser.parse_args(sys.argv[1:]))
    datasets = args["datasets"].split(",")
    feature_name = args["feature_extraction"]
    n_iter = args["n_iter"]
    use_mpi = args["use_mpi"]

    data_names = get_data_names(datasets)
    if use_mpi:
        executor = mpi_executor
    else:
        executor = serial_executor

    job_runner = ClusterJobRunner(data_names, feature_name, executor=executor)

    if use_mpi:
        mpi_hyper_optimize(job_runner, n_iter)
    else:
        serial_hyper_optimize(job_runner, n_iter)
