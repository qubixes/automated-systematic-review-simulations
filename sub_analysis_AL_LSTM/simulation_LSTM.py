#!/opt/local/bin/python

from pargrid.batch_generator import batch_from_params


var_param = {'simulation': range(50),
             'query_strategy': ['random', 'lc']}

fix_param = {}
fix_param["prior_included"] = \
    '1136 1940 466 4636 2708 4301 1256 1552 4048 3560'
fix_param["prior_excluded"] = \
    '1989 2276 1006 681 4822 3908 4896 3751 2346 2166'
fix_param["n_queries"] = 12
fix_param["n_instances"] = 40

# sub_analysis_LSTM/
param_file = "params.csv"
data_file = "pickle/ptsd_vandeschoot_words_20000.pkl"
config_file = "slurm_lisa.ini"

batch_from_params(var_param, fix_param, data_file, param_file, config_file)
