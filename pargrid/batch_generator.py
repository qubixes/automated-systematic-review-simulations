import pandas as pd
import numpy as np
import random
from batchgen import batch_from_strings
from sklearn.model_selection import ParameterGrid


def _create_df_parameter_grid(var_param, sample=None):
    " Create a parameter grid with or without sampling. "
    grid = ParameterGrid(var_param)
    grid = list(grid)
    if sample is not None:
        random.seed(9238752938)
        grid = random.sample(grid, sample)
    df = pd.DataFrame(grid)
    return df


def _create_df_fix_param(fix_param, n_sim):
    '''go throw the params and make a data frame (df)'''
    df_args = pd.DataFrame()
    for key in fix_param:
        df_args[key] = np.tile(fix_param[key], n_sim)
    return df_args


def params_to_file(var_param, fix_param, param_file, sample=None):

    df_pg = _create_df_parameter_grid(var_param, sample=sample)
    df_args = _create_df_fix_param(fix_param, df_pg.shape[0])

    df_all = pd.concat([df_pg, df_args], axis=1)
    df_all.index.name = 'T'
    df_all.to_csv(param_file)

    return df_all


def _args_from_row(row, param_names, data_file):
    param_val = map(lambda p: ' --' + p + ' ' + str(getattr(row, p)),
                    param_names)
    param_val_all = " ".join(list(param_val))
    job_str = data_file + param_val_all
    job_str += " --log_file output/results" + str(getattr(row, "T")) + ".log"
    return job_str


def commands_from_csv(data_file, param_file):
    """Create commands from a parameter (CSV) file.
       Arguments
       ---------
       data_file_path: str
           File with data to simulate.
       params_file_path: str
           File with parameter grid (CSV).
       config_file: str
       """
    params = pd.read_csv(param_file)
    base_job = "${python} -m asr simulate "
    param_names_all = list(params.columns.values)
    param_names = [p for p in param_names_all if p not in ['T', 'simulation']]

    script = []
    for row in params.itertuples():
        job_str = base_job + _args_from_row(row, param_names, data_file)
        script.append(job_str)

    return script


def pre_compute_defaults():
    """Define default the pre-compute commands

    Returns
    -------
    str:
        List of lines to execute before computation.
    """
    # TODO: install asr package
    # check if results.log is a file or folder
    mydef = """\
module load eb
module load Python/3.6.1-intel-2016b

cd $HOME/asr
mkdir -p "$TMPDIR"/output
rm -rf "$TMPDIR"/results.log
cp -r $HOME/asr/pickle "$TMPDIR"
cd "$TMPDIR"
    """
    return mydef


def post_compute_defaults():
    """Definition of post-compute commands
    Returns
    -------
    str:
        List of lines to execute after computation.
    """
    mydef = 'cp -r "$TMPDIR"/output  $HOME/asr'
    return mydef


def generate_shell_script(data_file, param_file, config_file):
    """ Create job script including job setup information for the batch system
        as well as job commands.
    Arguments
    ---------
    data_file_path: str
        File with systematic review data.
    params_file_path: str
        File with parameter grid (CSV).
    config_file: str
        configuration information for the batch system
    """
    script = commands_from_csv(data_file, param_file)
    script_commands = "\n".join(script)
    pre_com_string = pre_compute_defaults()
    post_com_string = post_compute_defaults()

    batch_from_strings(command_string=script_commands, config_file=config_file,
                       pre_com_string=pre_com_string,
                       post_com_string=post_com_string,
                       force_clear=True)


def batch_from_params(var_param, fix_param, data_file, param_file, config_file,
                      sample=None):
    params_to_file(var_param, fix_param, param_file, sample)
    generate_shell_script(data_file, param_file, config_file)
