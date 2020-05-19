#luminancia, cromancia
#https://en.wikipedia.org/wiki/YCbCr
#code:
#https://stackoverflow.com/questions/7041172/pils-colour-space-conversion-ycbcr-rgb

from numpy import dot, ndarray, array

def yuv2rgb(im, version='SDTV'):
    """
    Convert array-like YUV image to RGB colourspace

    version:
      - 'SDTV':  ITU-R BT.601 version  (default)
      - 'HDTV':  ITU-R BT.709 version
    """
    if not im.dtype == 'uint8':
        raise TypeError('yuv2rgb only implemented for uint8 arrays')

    # clip input to the valid range
    yuv = ndarray(im.shape)  # float64
    yuv[:,:, 0] = im[:,:, 0].clip(16, 235).astype(yuv.dtype) - 16
    yuv[:,:,1:] = im[:,:,1:].clip(16, 240).astype(yuv.dtype) - 128

    if version.upper() == 'SDTV':
        A = array([[1.,                 0.,  0.701            ],
                   [1., -0.886*0.114/0.587, -0.701*0.299/0.587],
                   [1.,  0.886,                             0.]])
        A[:,0]  *= 255./219.
        A[:,1:] *= 255./112.
    elif version.upper() == 'HDTV':
        A = array([[1.164,     0.,  1.793],
                   [1.164, -0.213, -0.533],
                   [1.164,  2.112,     0.]])
    else:
        raise Exception("Unrecognised version (choose 'SDTV' or 'HDTV')")

    rgb = dot(yuv, A.T)
    result = rgb.clip(0, 255).astype('uint8')

    return result


def get_ypbpr_functions(kr: float, kg: float, kb: float):
    y = lambda r, g, b : (kr * r) + (kg * g) + (kb*b)
    pb = lambda r, g, b : (1.0/2.0) * ( (b-y)/float(1-kb) )
    pr = lambda r, g, b : (1.0/2.0) * ( (r-y)/float(1-kr) )
    return y, pb, pr