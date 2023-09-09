import png
from os.path import join

__all__ = ["save_as_png", "save_all_images",]


def save_as_png(folder: str, filename: str, data: list, is_gray: bool):
    f = open(join(folder, filename), 'wb')      # binary mode is important
    lx = len(data[0])
    if not is_gray:
        lx = int(len(data[0])/3.0)
    w = png.Writer(lx, len(data), greyscale=is_gray)
    w.write(f, data)
    f.close()


def save_all_images(array3d: list, folder: str, name: str, is_gray: bool):
    # copy array
    array_of_data = []
    for i in range(len(array3d)):
        array_of_data.append([])
        for j in range(len(array3d[i])):
            array_of_data[i].append([])
            for k in range(len(array3d[i][j])):
                array_of_data[i][j].append(array3d[i][j][k])

    for i in range(len(array_of_data)):
        slice_data = array_of_data[i]
        for j in range(len(slice_data)):
            for k in range(len(slice_data[j])):
                slice_data[j][k] = int(slice_data[j][k]*255)
        save_as_png(folder, name + str(i) + '.png', slice_data, is_gray)
