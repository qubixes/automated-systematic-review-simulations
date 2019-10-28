
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc
from sklearn.utils.estimator_checks import check_decision_proba_consistency


# def _find_inclusions(results, labels, remove_initial=True):
#     inclusions = []
#     n_initial_inc = 0
#     cur_inclusions = 0
#     n_initial = 0
#     for query in results:
#         cursor = 0
#         for new_method in query["label_methods"]:
#             for i in range(cursor, cursor+new_method[1]):
#                 idx = query["labelled"][i][0]
#                 if new_method[0] == "initial" and remove_initial:
#                     n_initial_inc += labels[idx]
#                     n_initial += 1
#                 else:
#                     cur_inclusions += labels[idx]
#                     inclusions.append(cur_inclusions)
#             cursor += new_method[1]
# 
#     inclusions_after_init = sum(labels)
#     if remove_initial:
#         inclusions_after_init -= n_initial_inc
# #     exit()
#     return inclusions, inclusions_after_init, n_initial


def _find_inclusions(logger, labels, remove_initial=True):
    inclusions = []
    n_initial_inc = 0
    cur_inclusions = 0
    n_initial = 0
    n_queries = logger.n_queries()
    for query_i in range(n_queries):
        label_methods = logger.get("label_methods", query_i)
        label_idx = logger.get("label_idx", query_i)
        for i in range(len(label_idx)):
            if label_methods[i] == "initial" and remove_initial:
                n_initial_inc += labels[label_idx[i]]
                n_initial += 1
            else:
                cur_inclusions += labels[label_idx[i]]
                inclusions.append(cur_inclusions)

    inclusions_after_init = sum(labels)
    if remove_initial:
        inclusions_after_init -= n_initial_inc
#     exit()
    return inclusions, inclusions_after_init, n_initial


def _get_labeled_order(logger):
    label_order = []
    n_initial = 0
    n_queries = logger.n_queries()
    for query_i in range(n_queries):
        label_methods = logger.get("label_methods", query_i)
        label_idx = logger.get("label_idx", query_i)
        for i in range(len(label_idx)):
            if label_methods[i] == "initial":
                n_initial += 1
        label_order.extend(label_idx)
    return label_order, n_initial


def _get_last_proba_order(logger):

    n_queries = logger.n_queries()
    pool_idx = None
    for query_i in reversed(range(n_queries)):
        pool_idx = logger.get("pool_idx", query_i)
        if pool_idx is not None:
            proba = logger.get("proba", query_i)
            break

    if pool_idx is None:
        return []
    return pool_idx[np.argsort(-proba[pool_idx])]


def _get_proba_order(logger, query_i):
    try:
        pool_idx = logger.get("pool_idx", query_i)
    except KeyError:
        pool_idx = None

    if pool_idx is None:
        return None
    proba = logger.get("proba", query_i)[pool_idx]
    return pool_idx[np.argsort(proba)]


def _n_false_neg(logger, query_i, labels):
    proba_order = _get_proba_order(logger, query_i)
    if proba_order is None:
        return None
    res = np.zeros(len(proba_order))

    n_one = 0
    for i in range(len(res)):
        if labels[proba_order[i]] == 1:
            n_one += 1
        res[i] = n_one
    return np.array(list(reversed(res)))


def _get_limits(loggers, query_i, labels, proba_allow_miss=[]):
    num_left = None

    for logger in loggers.values():
        new_num_left = _n_false_neg(logger, query_i, labels)
        if new_num_left is None:
            return None

        if num_left is None:
            num_left = new_num_left
        else:
            num_left += new_num_left
    num_left /= len(loggers)
    limits = [len(num_left)]*len(proba_allow_miss)
    allow_miss = {i: proba for i, proba in enumerate(proba_allow_miss)}
#     allow_miss = list(zip(proba_allow_miss, range(len(proba_allow_miss)))
    for i in range(len(num_left)):
        for i_prob, prob in list(allow_miss.items()):
            if num_left[i] < prob:
                limits[i_prob] = i
                del allow_miss[i_prob]
        if len(allow_miss) == 0:
            break
    return limits


def _split_probabilities(proba, labels):
    """ Split probabilities into the two classes for further processing. """
    class_proba = [[], []]
    for res in proba:
        sample_id = res[0]
        prob = res[1]
        true_cat = labels[sample_id]
        class_proba[true_cat].append(prob)
    for i in range(2):
        class_proba[i] = np.array(class_proba[i])
    return class_proba


def _avg_false_neg(proba, labels):
    res = np.zeros(len(proba))
    proba.sort(key=lambda x: x[1])

    n_one = 0
    for i, item in enumerate(proba):
        sample_id = item[0]
        true_cat = labels[sample_id]
        if true_cat == 1:
            n_one += 1
        res[i] = n_one
    return res


def _limits_merged(proba, labels, p_allow_miss):
    if len(proba) == 0:
        return len(labels)
    n_samples = len(proba[0])
    n_proba = len(proba)
    false_neg = np.zeros(n_samples)
    for sub_proba in proba:
        false_neg += _avg_false_neg(sub_proba, labels)/n_proba

    for i, p_miss in enumerate(false_neg):
        if p_miss > p_allow_miss:
            return n_samples-i
    return 0


def _ROC(proba, labels):
    """ Compute the ROC of the prediction probabilities. """
    class_proba = _split_probabilities(proba, labels)
    num_c0 = class_proba[0].shape[0]
    num_c1 = class_proba[1].shape[0]

    y_score = np.concatenate((class_proba[0], class_proba[1]))
    np.nan_to_num(y_score, copy=False)
    y_true = np.concatenate((np.zeros(num_c0), np.ones(num_c1)))
    fpr, tpr, _ = roc_curve(y_true, y_score)

    roc_auc = auc(fpr, tpr)
    return roc_auc


def _ROC_merged(proba, labels):
    """ Merged version of _ROC(). """
    rocs = []
    for sub in proba:
        rocs.append(_ROC(sub, labels))
    roc_avg = np.mean(rocs)
    if len(rocs) > 1:
        roc_err = stats.sem(rocs)
    else:
        roc_err = 0
    return [roc_avg, roc_err]
