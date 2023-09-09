import random


def split_percentage(data, percentage, to_permute=True):
    data_len = len(data)
    indexes = [i for i in range(data_len)]
    if to_permute:
        random.shuffle(indexes)
    max_idx = round(data_len*percentage)
    data_train = [data[idx] for idx in indexes[0:max_idx]]
    data_test = [data[idx] for idx in indexes[max_idx:data_len]]
    return data_train, data_test


def permute(data):
    data_len = len(data)
    indexes = [i for i in range(data_len)]
    random.shuffle(indexes)
    return [data[i] for i in indexes]


def k_split(data, k, to_permute=True):
    if to_permute:
        data = permute(data)
    data_len_k = len(data)/float(k)
    datas = []
    for i_k in range(k):
        datas.append([data[i] for i in range(int(data_len_k*i_k), int(data_len_k*(i_k+1)))])
    return datas


def k_split_equally(data, k, to_permute=True):
    if to_permute:
        data = permute(data)
    data_len_k = int(len(data)/float(k))
    datas = []
    for i_k in range(k):
        datas.append([data[i] for i in range(data_len_k*i_k, data_len_k*(i_k+1))])
    rest = [data[i] for i in range(data_len_k*k, len(data))]
    return datas, rest
