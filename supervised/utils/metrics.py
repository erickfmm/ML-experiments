import math


# TODO: make this good, now its some random function that does anything else mutual information
def mutual_information(y_real, y_predicted):
    classes = max([len(set(y_real)), len(set(y_predicted))])
    p_real = [0 for _ in range(classes)]
    p_predicted = [0 for _ in range(classes)]
    p_ij =  [0 for _ in range(classes)]
    for y_r in y_real:
        p_real[y_r] += 1
    p_real = [p_real[i]/float(len(y_real)) for i in range(len(p_real))]
    for y_p in y_predicted:
        p_predicted[y_p] += 1
    p_predicted = [p_predicted[i] / float(len(y_predicted)) for i in range(len(p_predicted))]
    for i_real in range(len(y_real)):
        if y_real[i_real] == y_predicted[i_real]:
            p_ij[y_real[i_real]] += 1
    p_ij = [p_ij[i]/float(len(y_real)) for i in range(len(p_ij))]
    prob_join = [(p_real[i]+p_predicted[i])/2.0 for i in range(classes)]
    mi = 0.0
    total_len = len(y_real) + len(y_predicted)
    for iclass_real in range(classes):
        for iclass_pred in range(classes):
            mi += p_ij[iclass_real] * math.log((len(y_real) * (p_ij[iclass_real] * len(y_real)))
                                               / float(p_real[iclass_real] * float(len(y_real)) *
                                                       p_predicted[iclass_pred] * float(len(y_predicted))))
        # mi += prob_join[iclass_real] * math.log( (p_ij[iclass_real]) / float(p_real[iclass_real]) )
        # mi += p_ij[iclass_real] * math.log(p_ij[iclass_real] / float(p_real[iclass_real] * p_predicted[iclass_real]))
    return mi, p_ij, p_predicted, p_real, classes
