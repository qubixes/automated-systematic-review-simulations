#!/usr/bin/env python

import sys
from asreview.simulation import Plot

args = sys.argv[1:]
if len(args) > 0:
    json_dirs = args
else:
    json_dirs = ["output"]

with Plot.from_dirs(json_dirs) as plot:
#     plot.plot_inc_found(result_format="percentage")
#     plot.plot_time_to_discovery()
# my_plotter.plot_ROC()
    plot.plot_limits()
