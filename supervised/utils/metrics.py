import math

#this code: import with_nolib.supervised.metrics as ms


#TODO: make this good, now its some random function that does anything else mutual information
def mutual_information(y_real, y_pred):
    classes = max([len(set(y_real)), len(set(y_pred))])
    p_real = [0 for _ in range(classes)]
    p_pred = [0 for _ in range(classes)]
    p_ij =  [0 for _ in range(classes)]
    for y_r in y_real:
        p_real[y_r] += 1
    p_real = [p_real[i]/float(len(y_real)) for i in range(len(p_real))]
    for y_p in y_pred:
        p_pred[y_p] += 1
    p_pred = [p_pred[i]/float(len(y_pred)) for i in range(len(p_pred))]
    for i_real in range(len(y_real)):
        if y_real[i_real] == y_pred[i_real]:
            p_ij[y_real[i_real]] += 1
    p_ij = [p_ij[i]/float(len(y_real)) for i in range(len(p_ij))]
    prob_join = [(p_real[i]+p_pred[i])/2.0 for i in range(classes)]
    mi = 0.0
    total_len = len(y_real) + len(y_pred)
    for iclass_real in range(classes):
        for iclass_pred in range(classes):
            mi += p_ij[iclass_real] * math.log(( len(y_real) * (p_ij[iclass_real] * len(y_real)) )/ float(p_real[iclass_real]*float(len(y_real)) *  p_pred[iclass_pred]*float(len(y_pred)) ) )
        #mi += prob_join[iclass_real] * math.log( (p_ij[iclass_real]) / float(p_real[iclass_real]) )
        #mi += p_ij[iclass_real] * math.log(p_ij[iclass_real] / float(p_real[iclass_real] * p_pred[iclass_real]))
    return mi, p_ij, p_pred, p_real, classes
