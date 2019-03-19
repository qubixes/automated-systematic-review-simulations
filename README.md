# Automated Systematic Review - Simulation study

This project contains the code of the simulation study for the [Automated
Systematic Review](https://github.com/msdslab/automated-systematic-review)
project. This repository is used to simulate, aggregate and visualise the
this study.

We make use of the SURFSara HPC infrastructure. But with some modifications, the code can be run on other HPC facilities as well.

## Installation 

The Automated Systematic Review project requires Python 3.6+. 

The simulation project can be directly installed with: 

```bash
pip install --user git+https://github.com/msdslab/automated-systematic-review-simulations
```
Dependencies are automatically installed as well.

## Simulations 

The simulations in this study are listed on this page:

- [Simulation AL-LSTM ](/sub_analysis_AL_LSTM) - This is a simulation in
  which we explore an Active Learning solution, using Long Short-Term Memory
  (LSTM) model. 


## Data preparation

To prevent waste due to the loading of the embedding file, a Python pickle file is created with only the words that are in the abstracts. 

They can be created from the CLI:

``` bash
pickle_asr data_file embedding_file [--words=20000]
```
From within Python:

``` python
write_pickle(data_file, embedding_file, num_words=20000)
```

Or automatically while creating the batch:

``` python
batch_from_params(var_param, fix_param, data_file, embedding_file, param_file, config_file)
```

Pickle files are created in the current working directory/pickle.

To run a simulation, one needs code such as:

``` python 
#!/opt/local/bin/python

from pargrid import batch_from_params

# Generate a grid with these variables, different combinations for each run.
var_param = {'simulation': range(50),
             'query_strategy': ['random', 'lc']}

# Fixed parameters that are the same for each run.
fix_param = {}
fix_param["prior_included"] = \
    '1136 1940 466 4636 2708 4301 1256 1552 4048 3560'
fix_param["prior_excluded"] = \
    '1989 2276 1006 681 4822 3908 4896 3751 2346 2166'
fix_param["n_queries"] = 12
fix_param["n_instances"] = 40

# sub_analysis_LSTM/
# Define file names and data sources
param_file = "params.csv"
config_file = "slurm_lisa.ini"
embedding_file = "../../cc.en.300.vec"
data_file = "../../schoot.csv"

# Create a batch from the above file names and data sources.
batch_from_params(var_param, fix_param,
                  data_file, embedding_file, param_file, config_file)

```

This creates all the batch files needed in current directory/batch.slurm\_lisa/${job\_name}.

