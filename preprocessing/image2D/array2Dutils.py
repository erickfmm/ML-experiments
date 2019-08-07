

def transpose(data2D):
    new_data = []
    for icol in range(len(data2D[0])):
        new_data.append([])
        for irow in range(len(data2D)):
            new_data[icol].append(data2D[irow][icol])
    return new_data