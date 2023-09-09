import numpy as np

__all__ = ["get_bigmatrix", "get_limits", "normalize3D", "get_data_thresholded",]

# codigo del profesor
def get_bigmatrix(D, nrows=7):
    (xdim, ydim, zdim) = D.shape
    ncols = int(np.ceil(float(zdim)/nrows))
    msk = np.ones( (ydim, xdim) )
    Manat = np.tile(np.array([]), (xdim*ncols, 1)).T
    Mmsk = np.tile(np.array([]), (xdim*ncols, 1)).T
    for r in range(nrows):
        Mrow = np.tile(np.array([]), (ydim, 1))
        Mmskrow = np.tile(np.array([]), (ydim, 1))
        for c in range(ncols):
            sl = c + ncols*(r)
            if sl <= zdim-1:
                Mrow = np.hstack([Mrow, D[:, ::-1, sl].T])
                Mmskrow = np.hstack([Mmskrow, msk*sl])
            else:
                Mrow = np.hstack([Mrow, np.zeros((ydim, xdim))])
                Mmskrow = np.hstack([Mmskrow, (-10)*np.ones((ydim, xdim))])
        Manat = np.vstack( [Mrow, Manat] )
        Mmsk = np.vstack( [Mmskrow, Mmsk] )
    return Manat, Mmsk


def get_limits(data):
    min_number = None
    max_number = None
    for i in range(len(data)):
        current_slice = data[i]
        current_slice = np.array(current_slice)
        current_slice[np.where(np.isnan(current_slice))] = 0
        tmp_min_number = np.min(current_slice)
        tmp_max_number = 1 if np.max(current_slice) < 1 else np.max(current_slice)
        if (min_number is None) or (max_number is None):
            min_number = tmp_min_number
            max_number = tmp_max_number
        else:
            min_number = tmp_min_number if tmp_min_number < min_number else min_number
            max_number = tmp_max_number if tmp_max_number > max_number else max_number
    return min_number, max_number

def normalize3D(data, use_local=False):
    data_to_return = []
    mins = []
    maxs = []
    for i in range(len(data)):
        current_slice = data[i]
        current_slice = np.array(current_slice)
        current_slice[np.where(np.isnan(current_slice))] = 0
        min_number = None
        max_number = None
        if use_local:
            print(i, ", min: ",np.min(current_slice))
            print(i, ", max: ",np.max(current_slice))
            min_number = np.min(current_slice)
            max_number = 1 if np.max(current_slice) < 1 else np.max(current_slice)
        else:
            (min_number, max_number) = get_limits(data)
        mins.append(min_number)
        maxs.append(max_number)
        for j in range(len(current_slice)):
            for k in range(len(current_slice[j])):
                if np.isnan(current_slice[j][k]):
                    current_slice[j][k] = 0
                else:
                    current_slice[j][k] = float(current_slice[j][k]+np.abs(min_number)) / float(max_number+np.abs(min_number))
        data_to_return.append(current_slice)
    return data_to_return, mins, maxs


def get_data_thresholded(anat, tmap, threshold):
    data_to_return = []
    wanat_normalized, mins_wanat, maxs_wanat = normalize3D(anat)
    tmap_normalized, mins_tmap, maxs_tmap = normalize3D(tmap)
    for i in range(len(anat)):
        anat_slice = anat[i]
        tmap_slice = tmap[i]
        data = []
        for j in range(len(anat_slice)):
            data.append([])
            for k in range(len(anat_slice[j])):
                if tmap_slice[j][k] > threshold:
                    data[j].extend([tmap_normalized[i][j][k], 0, tmap_normalized[i][j][k]*0.2])
                else:
                    data[j].extend([wanat_normalized[i][j][k],wanat_normalized[i][j][k],wanat_normalized[i][j][k]])
        data_to_return.append(data)
    return data_to_return
