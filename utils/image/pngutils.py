import png
from os.path import join

__all__ = ["save_as_png", "save_all_images",]

def save_as_png(folder, filename, data, isgray):
    f = open(join(folder, filename), 'wb')      # binary mode is important
    lx = len(data[0])
    if not isgray:
        lx = int(len(data[0])/3.0)
    w = png.Writer(lx, len(data), greyscale=isgray)
    w.write(f, data)
    f.close()

def save_all_images(array3D, folder, name, isgray):
    #copy array
    array_of_data = []
    for i in range(len(array3D)):
        array_of_data.append([])
        for j in range(len(array3D[i])):
            array_of_data[i].append([])
            for k in range(len(array3D[i][j])):
                array_of_data[i][j].append(array3D[i][j][k])

    for i in range(len(array_of_data)):
        slice_data = array_of_data[i]
        for j in range(len(slice_data)):
            for k in range(len(slice_data[j])):
                slice_data[j][k] = int(slice_data[j][k]*255)
        save_as_png(folder, name+str(i)+'.png', slice_data, isgray)
