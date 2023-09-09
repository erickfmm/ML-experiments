

def confusion_matrix(y_real, y_predicted):
    n_classes = len(set(y_real))
    new_matrix = []
    print("len y_real: ", len(y_real))
    print("len y_predicted: ", len(y_predicted))
    print("n classes: ", n_classes)
    for i_class in range(n_classes):
        new_matrix.append([])
        for _ in range(n_classes):  # i_y_predicted
            new_matrix[i_class].append(0.0)
    for i_yreal in range(len(y_real)):
        new_matrix[y_real[i_yreal]][y_predicted[i_yreal]] += 1
    return new_matrix


def normalize(confusion_matrix_):
    new_matrix = []
    for i_row in range(len(confusion_matrix_)):
        new_matrix.append([])
        sum_row = sum(confusion_matrix_[i_row])
        for idata in range(len(confusion_matrix_[i_row])):
            if sum_row == 0:
                new_matrix[i_row].append(0.0)
            else:
                new_matrix[i_row].append(confusion_matrix_[i_row][idata] / float(sum_row))
    return new_matrix
