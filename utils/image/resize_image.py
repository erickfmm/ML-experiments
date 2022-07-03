#code from:
#https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/

from PIL import Image, ImageOps

def with_image_module(im: Image, desired_size:int) -> Image:
    """Use an Image object and returns an object with the image
    scalated to a square size given in params, padding the borders
    It uses only the Image module from PIL

    Parameters
    ----------
    im:Image: 
        image object
    desired_size:int: 
        size in pixels of the desired length or width for the image

    Returns
    -------
    Image
        object with the image in the desired size x deired size

    """
    #desired_size = 368
    #im_pth = "/home/jdhao/test.jpg"

    #im = Image.open(im_path)
    old_size = im.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # use thumbnail() or resize() method to resize the input image

    # thumbnail is a in-place operation

    # im.thumbnail(new_size, Image.ANTIALIAS)

    im = im.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    #new_im.show()
    return new_im

def with_imageops_module(im:Image, desired_size:int) -> Image:
    """Use an Image object and returns an object with the image
    scalated to a square size given in params, padding the borders
    It uses only the Image module from PIL

    Parameters
    ----------
    im:Image: 
        image object
    desired_size:int: 
        size in pixels of the desired length or width for the image

    Returns
    -------
    Image
        object with the image in the desired size x deired size

    """
    #im = Image.open(im_path)
    old_size = im.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    im = im.resize(new_size, Image.ANTIALIAS)

    #here the new code
    delta_w = desired_size - new_size[0]
    delta_h = desired_size - new_size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_im = ImageOps.expand(im, padding)

    #new_im.show()
    return new_im