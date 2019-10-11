
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc


def _find_inclusions(results, labels, remove_initial=True):
    inclusions = []
    n_initial_inc = 0
    cur_inclusions = 0
    n_initial = 0
    for query in results:
        cursor = 0
        for new_method in query["label_methods"]:
            for i in range(cursor, cursor+new_method[1]):
                idx = query["labelled"][i][0]
                if new_method[0] == "initial" and remove_initial:
                    n_initial_inc += labels[idx]
                    n_initial += 1
                else:
                    cur_inclusions += labels[idx]
                    inclusions.append(cur_inclusions)
            cursor += new_method[1]

    inclusions_after_init = sum(labels)
    if remove_initial:
        inclusions_after_init -= n_initial_inc
#     exit()
    return inclusions, inclusions_after_init, n_initial


def _get_labeled_order(results):
    label_order = []
    n_initial = 0
    for query in results:
        cursor = 0
        for new_method in query["label_methods"]:
            for i in range(cursor, cursor+new_method[1]):
                idx = query["labelled"][i][0]
                if new_method[0] == "initial":
                    n_initial += 1
                label_order.append(idx)
            cursor += new_method[1]
    return label_order, n_initial


def _get_proba_order(proba):
    indices = np.array([x[0] for x in proba])
    probabilities = np.array([x[1] for x in proba])
    return indices[np.argsort(-probabilities)]


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
