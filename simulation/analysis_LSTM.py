#!/usr/bin/env python

import sys
from asreview.simulation import Plot

args = sys.argv[1:]
if len(args) > 0:
    json_dirs = args
else:
    json_dirs = ["output"]


my_plotter = Plot.from_dirs(json_dirs)
my_plotter.plot_inc_found()
my_plotter.plot_time_to_discovery()
my_plotter.plot_ROC()
my_plotter.plot_limits()
