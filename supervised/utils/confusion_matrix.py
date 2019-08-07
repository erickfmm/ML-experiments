

def confusion_matrix(yreal, ypred):
    n_classes = len(set(yreal))
    new_matrix = []
    print("len yreal: ", len(yreal))
    print("len ypred: ", len(ypred))
    print("n classes: ", n_classes)
    for i_class in range(n_classes):
        new_matrix.append([])
        for _ in range(n_classes): #i_ypred
            new_matrix[i_class].append(0.0)
    for i_yreal in range(len(yreal)):
        new_matrix[yreal[i_yreal]][ypred[i_yreal]] += 1
    return new_matrix

def normalize(confusion_matrix):
    new_matrix = []
    for irow in range(len(confusion_matrix)):
        new_matrix.append([])
        sum_row = sum(confusion_matrix[irow])
        for idata in range(len(confusion_matrix[irow])):
            if sum_row == 0:
                new_matrix[irow].append(0.0)
            else:
                new_matrix[irow].append(confusion_matrix[irow][idata]/float(sum_row))
    return new_matrix