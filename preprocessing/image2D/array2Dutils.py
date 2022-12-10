__all__ = ["transpose"]


def transpose(data2d):
    new_data = []
    for i_col in range(len(data2d[0])):
        new_data.append([])
        for i_row in range(len(data2d)):
            new_data[i_col].append(data2d[i_row][i_col])
    return new_data
