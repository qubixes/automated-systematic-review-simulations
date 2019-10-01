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


def _speedup(proba, labels, n_allow_miss=0, normalize=True):
    """
    Compute the number of non included papers below the worst (proba)
    one that is included in the final review.

    Arguments
    ---------
    proba: float
        List of probabilities with indices.
    labels: int
        True categories of each index.
    n_allow_miss: int
        Number of allowed misses.
    normalize: bool
        Whether to normalize with the expected number.

    Returns
    -------
    float:
        Number of papers that don't need inclusion.
    """
    class_proba = _split_probabilities(proba, labels)
    for i in range(2):
        class_proba[i].sort()

    # Number of 0's / 1's / total in proba set.
    num_c0 = class_proba[0].shape[0]
    num_c1 = class_proba[1].shape[0]
    num_tot = num_c0 + num_c1

    # Determine the normalization factor, probably not quite correct.
    if normalize and num_c1 > 0:
        norm_fact = (1+2*n_allow_miss)*num_tot/(2*num_c1)
    else:
        norm_fact = 1

    # If the number of 1 labels is smaller than the number allowed -> all.
    if num_c1 <= n_allow_miss:
        return num_tot/norm_fact
    lowest_prob = class_proba[1][n_allow_miss]
    for i, val in enumerate(class_proba[0]):
        if val >= lowest_prob:
            return (i+n_allow_miss)/norm_fact
    return num_c0/norm_fact


def _avg_false_neg(proba, labels):
    res = np.zeros(len(proba))
    proba.sort(key=lambda x: x[1])
#     print(proba[:100])
    n_one = 0
    for i, item in enumerate(proba):
        sample_id = item[0]
        true_cat = labels[sample_id]
        if true_cat == 1:
            n_one += 1
        res[i] = n_one
    return res


def _limits_merged(proba, labels, p_allow_miss):
    n_samples = len(proba[0])
    n_proba = len(proba)
    false_neg = np.zeros(n_samples)
    for sub_proba in proba:
        false_neg += _avg_false_neg(sub_proba, labels)/n_proba

    for i, p_miss in enumerate(false_neg):
        if p_miss > p_allow_miss:
            return i-1
    return n_samples


def _speedup_merged(proba, labels, n_allow_miss=0, normalize=True):
    """ Merged version of _speedup(), compute average and mean. """
    speedup = []
    for sub_proba in proba:
        speedup.append(_speedup(sub_proba, labels, n_allow_miss, normalize))
    speed_avg = np.mean(speedup)
    if len(speedup) > 1:
        speed_err = stats.sem(speedup)
    else:
        speed_err = 0
    return [speed_avg, speed_err]


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


def _inc_queried(proba, labels):
    """ Compute the number of queried labels that were one. """
    class_proba = _split_probabilities(proba, labels)
    num_pool1 = class_proba[1].shape[0]
    num_tot1 = sum(labels)
    return num_tot1-num_pool1


def _inc_queried_merged(proba, labels):
    """ Merged version of _inc_queried. """
    found = []
    for sub in proba:
        found.append(_inc_queried(sub, labels))
    found_avg = np.mean(found)
    if len(found) > 1:
        found_err = stats.sem(found)
    else:
        found_err = 0
    return [found_avg, found_err]


def _inc_queried_all_merged(proba, labels):
    """ Merged version of _inc_queried. """
    found = []
    for sub in proba:
        found.append(_inc_queried(sub, labels))
    return found


def _avg_proba(proba, labels):
    """ Average of the prediction probabilities. """
    n = len(proba)
    class_proba = _split_probabilities(proba, labels)
    results = []
    for i in range(2):
        new_mean = np.mean(class_proba[i])
        new_sem = stats.sem(class_proba[i])
        results.append([n, new_mean, new_sem])
    return results


def _avg_proba_merged(proba, labels):
    """ Merged version of prediction probabilities. """

    # Flatten list
    flat_proba = []
    for sub in proba:
        for item in sub:
            flat_proba.append(item)
    return _avg_proba(flat_proba, labels)


