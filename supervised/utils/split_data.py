import random

def split_percentage(data, labels, percentage, to_permute=True):
    if len(data) != len(labels):
        raise ValueError("length of data and labels are not equal")
    data_len = len(data)
    indexes = [i for i in range(data_len)]
    if to_permute:
        random.shuffle(indexes)
    max_idx = round(data_len*percentage)
    data_train = [data[idx] for idx in indexes[0:max_idx]]
    labels_train = [labels[idx] for idx in indexes[0:max_idx]]
    data_test = [data[idx] for idx in indexes[max_idx:data_len]]
    labels_test = [labels[idx] for idx in indexes[max_idx:data_len]]
    return data_train, labels_train, data_test, labels_test

def permute(data, labels):
    if len(data) != len(labels):
        raise ValueError("length of data and labels are not equal")
    data_len = len(data)
    indexes = [i for i in range(data_len)]
    random.shuffle(indexes)
    perm_data = [data[i] for i in indexes]
    perm_labels = [labels[i] for i in indexes]
    return perm_data, perm_labels

def k_split(data, labels, k, to_permute=True):
    if len(data) != len(labels):
        raise ValueError("length of data and labels are not equal")
    if to_permute:
        data, labels = permute(data, labels)
    data_len_k = len(data)/float(k)
    datas = []
    klabels = []
    for i_k in range(k):
        datas.append([data[i] for i in range(int(data_len_k*i_k), int(data_len_k*(i_k+1)))])
        klabels.append([labels[i] for i in range(int(data_len_k*i_k), int(data_len_k*(i_k+1)))])
    return datas, klabels

def k_split_equally(data, labels, k, to_permute=True):
    if len(data) != len(labels):
        raise ValueError("length of data and labels are not equal")
    if to_permute:
        data, labels = permute(data, labels)
    data_len_k = int(len(data)/float(k))
    datas = []
    klabels = []
    for i_k in range(k):
        datas.append([data[i] for i in range(data_len_k*i_k, data_len_k*(i_k+1))])
        klabels.append([labels[i] for i in range(data_len_k*i_k, data_len_k*(i_k+1))])
    rest = [data[i] for i in range(data_len_k*k, len(data))]
    rest_labels = [labels[i] for i in range(data_len_k*k, len(data))]
    return datas, klabels, rest, rest_labels