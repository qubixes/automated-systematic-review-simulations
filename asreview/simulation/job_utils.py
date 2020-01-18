import os
from os.path import join, splitext


def empty_shared():
    return {
        "query_src": {},
        "current_queries": {}
    }


def quality(result_list, alpha=1):
    q = 0
    for _, rank in result_list:
        q += rank**alpha

    return (q/len(result_list))**(1/alpha)


def get_trial_fp(datasets, model_name=None, query_name=None, balance_name=None,
                 feature_name=None, hyper_type="inactive"):

    name_list = [
        name for name in [model_name, query_name, balance_name, feature_name]
        if name is not None
    ]

    trials_dir = join("output", hyper_type, "_".join(name_list),
                      "_".join(datasets))
    os.makedirs(trials_dir, exist_ok=True)
    trials_fp = os.path.join(trials_dir, f"trials.pkl")

    return trials_dir, trials_fp


def get_data_names(datasets, data_dir="data"):
    data_dir = "data"
    file_list = os.listdir(data_dir)
    file_list = [file_name for file_name in file_list
                 if file_name.endswith((".csv", ".xlsx", ".ris"))]
    if "all" not in datasets:
        file_list = [file_name for file_name in file_list
                     if splitext(file_name)[0] in datasets]
    return [splitext(file_name)[0] for file_name in file_list]


def _get_prefix_param(raw_param, prefix):
    return {key[4:]: value for key, value in raw_param.items()
            if key[:4] == prefix}


def get_split_param(raw_param):
    return {
        "model_param": _get_prefix_param(raw_param, "mdl_"),
        "query_param": _get_prefix_param(raw_param, "qry_"),
        "balance_param": _get_prefix_param(raw_param, "bal_"),
        "feature_param": _get_prefix_param(raw_param, "fex_"),
    }


def data_fp_from_name(data_dir, data_name):
    file_list = os.listdir(data_dir)
    file_list = [file_name for file_name in file_list
                 if file_name.endswith((".csv", ".xlsx", ".ris")) and
                 os.path.splitext(file_name)[0] == data_name]
    return join(data_dir, file_list[0])


def get_out_dir(trials_dir, data_name):
    out_dir = join(trials_dir, "current", data_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def get_out_fp(trials_dir, data_name, i_run):
    return join(get_out_dir(trials_dir, data_name),
                f"results_{i_run}.json")


def get_label_fp(trials_dir, data_name):
    return join(get_out_dir(trials_dir, data_name), "labels.json")
