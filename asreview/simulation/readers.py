import json
import os
import re


def read_json_results(data_dir):
    """
    Find all results in a directory and read them in memory.
    Assume that all files in this directory have the same model parameters.

    Arguments
    ---------
    data_dir: str
        Directory in which to find any log files.

    Returns
    -------
    dict:
        Dictionary containing the results.
    """
    json_data = {}
    files = os.listdir(data_dir)
    if not files:
        print(f"Error: {data_dir} is empty")
        return None

    min_queries = int(10**9)
    print(files)

    res_files = []
    for json_file in files:
        print(json_file)
        if not re.match(r'^results', json_file):
            continue

        res_files.append(json_file)
        with open(os.path.join(data_dir, json_file), "r") as fp:
            json_data[json_file] = json.load(fp)

        i = 0
        while i < len(json_data[json_file]):
            if str(i) not in json_data[json_file]:
                min_queries = min(min_queries, i)
                break
            i += 1

    # Make sure they all have the same number of queries.
#     print(f"min_queries: {min_queries}")
    for json_file in res_files:
        i = min_queries
        max_i = len(json_data[json_file])
        while i < max_i:
            if str(i) not in json_data[json_file]:
                break
            del json_data[json_file][str(i)]
#             print(f"Warning: not using query {i} from file {json_file}")
            i += 1
#         print(f"{json_data[json_file].keys()}")

    return json_data


def reorder_results(old_results):
    """
    From a dictionary of results, create a better ordered result.
    The hierarchy of the new dictionary is:
    logname -> query_id -> filename -> data.

    Arguments
    ---------
    old_results: dict
        Results to reorder.

    Returns
    dict:
        Reordered results.
    """
    results = {}
    first_file = list(old_results.keys())[0]
    lognames = old_results[first_file]["0"].keys()
    queries = old_results[first_file].keys()
    files = old_results.keys()

    for log in lognames:
        results[log] = {}
        for query in queries:
            try:
                int(query)
            except ValueError:
                continue
            results[log][query] = []
            for fp in files:
                results[log][query].append(old_results[fp][query][log])
    return results


def get_num_queries(results):
    """ Get the number of queries from the non-reordered results. """
    num_queries = []
    for filename in results:
        cur_num = []
        for query in results[filename]:
            # All results are an integer number.
            try:
                int(query)
            except ValueError:
                continue
            # Count the number of labeled samples each query.
            d_num = len(results[filename][query]["labelled"])
            if len(cur_num) == 0:
                cur_num.append(d_num)
            else:
                cur_num.append(d_num + cur_num[-1])
        # Assert that the number of queries is the same for all files.
        if len(num_queries) == 0:
            num_queries = cur_num
        else:
            assert num_queries == cur_num
    return num_queries
