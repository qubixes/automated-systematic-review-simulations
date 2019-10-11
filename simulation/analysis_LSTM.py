#!/usr/bin/env python

import sys
from asreview.simulation import Analysis
from asreview.simulation import Plot

args = sys.argv[1:]
if len(args) > 0:
    json_dirs = args
else:
    json_dirs = ["output"]


my_plotter = Plot.from_dirs(json_dirs)
my_plotter.plot_inc_found()
my_plotter.plot_time_to_discovery()
# my_plotter.plot_ROC()
# my_analysis.plot_proba()
# my_analysis.plot_speedup([0, 1, 2, 3], normalize=False)
# my_analysis.plot_ROC()
# my_analysis.plot_limits([0.1, 0.5, 1, 2])
# my_analysis.print_avg_time_found()
