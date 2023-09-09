# True Positives (TP) - These are the correctly predicted positive values which means
# that the value of actual class is yes and the value of predicted class is also yes.
# True Negatives (TN) - These are the correctly predicted negative values which means
# that the value of actual class is no and value of predicted class is also no.
# False Positives (FP) – When actual class is no and predicted class is yes.
# False Negatives (FN) – When actual class is yes but predicted class in no.

# Precision = TP/TP+FP
# Recall = TP/TP+FN

# F1 Score = 2*(Recall * Precision) / (Recall + Precision)
# F1 is usually more useful than accuracy, especially if for an uneven class distribution.


def true_positives(y_real: list, y_predicted: list) -> int:
    # these are the correctly predicted
    # positive values which means that
    # the value of actual class is yes
    # and the value of predicted class is also yes.
    tp = 0
    for i in range(len(y_real)):
        if y_real[i] == 1 and y_predicted[i] == 1:
            tp += 1
    return tp


def true_negatives(y_real: list, y_predicted: list) -> int:
    # These are the correctly predicted
    # negative values which means that the
    # value of actual class is no and value
    # of predicted class is also no.
    tn = 0
    for i in range(len(y_real)):
        if y_real[i] == 0 and y_predicted[i] == 0:
            tn += 1
    return tn


def false_positives(y_real: list, y_predicted: list) -> int:
    # When actual class is no and predicted class is yes.
    fp = 0
    for i in range(len(y_real)):
        if y_real[i] == 0 and y_predicted[i] == 1:
            fp += 1
    return fp


def false_negatives(y_real: list, y_predicted: list) -> int:
    # When actual class is yes but predicted class in no.
    fn = 0
    for i in range(len(y_real)):
        if y_real[i] == 1 and y_predicted[i] == 0:
            fn += 1
    return fn


def precision(y_real: list, y_predicted: list) -> float:
    tp = true_positives(y_real, y_predicted)
    fp = false_positives(y_real, y_predicted)
    return tp/float(tp+fp)


def recall(y_real: list, y_predicted: list) -> float:
    tp = true_positives(y_real, y_predicted)
    fn = false_negatives(y_real, y_predicted)
    return tp/float(tp+fn)


def f1_score(y_real: list, y_predicted: list) -> float:
    # F1 is usually more useful than accuracy,
    # especially if for an uneven class distribution.
    recall_ = recall(y_real, y_predicted)
    precision_ = precision(y_real, y_predicted)
    return 2.0*(recall_ * precision_) / float(recall_ + precision_)


def accuracy(y_real: list, y_predicted: list) -> float:
    count_errors = 0
    for i in range(len(y_real)):
        if y_real[i] != y_predicted[i]:
            count_errors += 1
    return count_errors/float(len(y_real))
