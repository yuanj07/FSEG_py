import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

from scipy import ndimage


def io_from_prompt(img_path: str, shape_size: tuple[int, int, int] or tuple[int, int], dtype: str) -> np.ndarray:
    """

    :param img_path:
    :param shape_size:
    :param dtype:
    :return:
    """

    if (img_path.split(".")[-1] == "raw"):
        if (dtype):
            img = np.fromfile(img_path, dtype=dtype)
        else:
            img = np.fromfile(img_path)
        if (len(shape_size) == 3):
            img = np.resize(img, shape_size)
        else:
            img = np.resize(img, (shape_size[0], shape_size[1]))

    else:
        if (dtype):
            img = cv2.imread(img_path, 0).astype(dtype)

        else:
            img = cv2.imread(img_path, 0)

    return img


def log_filter(sgm: float, fsize: list[int, int]) -> np.ndarray:
    """
    LoG filter
    :param sgm: sigma in Gaussian
    :param fsize: filter size, [h, w]
    :return: LoG filter
    """
    wins_x = int(fsize[1] / 2)
    wins_y = int(fsize[0] / 2)

    out = np.zeros(fsize, dtype=np.float32)

    for x in range(-wins_x, wins_x + 1):
        for y in range(-wins_y, wins_y + 1):
            out[wins_y + y, wins_x + x] = - 1. / (np.pi * sgm ** 4.) * (
                    1. - (x * x + y * y) / (2. * sgm * sgm)) * np.exp(-(x * x + y * y) / (2. * sgm * sgm))

    return out - np.mean(out)


def gabor_filter(sgm, theta):
    """
    Gabor filter
    :param sgm: sigma in Gaussian
    :param theta: direction
    :return: gabor filter
    """
    phs = 0
    gamma = 1
    wins = int(np.floor(sgm * 2))
    f = 1 / (sgm * 2.)
    out = np.zeros((2 * wins + 1, 2 * wins + 1))

    for x in range(-wins, wins + 1):
        for y in range(-wins, wins + 1):
            xPrime = x * np.cos(theta) + y * np.sin(theta)
            yPrime = y * np.cos(theta) - x * np.sin(theta)
            out[wins + y, wins + x] = 1 / (2 * np.pi * sgm * sgm) * np.exp(
                -.5 * ((xPrime) ** 2 + (yPrime * gamma) ** 2) / sgm ** 2) * np.cos(2 * np.pi * f * xPrime + phs)
    return out - np.mean(out)


def image_filtering(img: np.ndarray, filter_list: list[tuple[str, float, float or tuple[int, int]]]) -> np.ndarray:
    sub_img = []
    for filter in filter_list:
        assert (filter[0] == 'log') | (filter[0] == 'gabor'), 'Undefined filter name. '
        if filter[0] == 'log':
            f = log_filter(filter[1], filter[2])
            tmp = ndimage.correlate(np.float32(img), f, mode='reflect')
            sub_img.append(tmp)

        elif filter[0] == 'gabor':
            f = gabor_filter(filter[1], filter[2])
            tmp = ndimage.correlate(np.float32(img), f, mode='reflect')
            sub_img.append(tmp)

    return np.float32(np.stack(sub_img, axis=2))


def overlay(img1: np.ndarray, img2: np.ndarray, alpha: float, **kwargs: any) -> None:
    """
        Plots an overlay of an image and its segmentation using pyplot.imshow with useful parameters.
    Args:
        img1: original image, defalut cmap is fixed as gray an alpha is fixed as 1.0
        img2: label image, cmap can be set in kwargs
        alpha: alpha value of the label image
    Custom kwargs:
        size: if used, defines figsize in plt.figure(figsize(size, size))
    Default kwargs for imshow (can be overwritten):
        cmap: default is gray (and cannot be overwitten for img1, only for img2)
        interpolation:'none'
        aspect:'equal'
    """
    # figure and custom params
    if 'size' in kwargs.keys():
        squaresize = kwargs.pop('size')
        fig = plt.figure(figsize=(squaresize, squaresize))
    else:
        fig = plt.figure()

    if "save_fig" in kwargs.keys():
        save_fig_name = kwargs.pop("save_fig")
        save_dir = kwargs.pop("save_dir")
        title = kwargs.pop("plot_title")

    else:
        save_fig_name = ""
        save_dir = ""
        title = ""

    # default imshow params
    params = {'interpolation': 'none', 'aspect': 'equal'}
    # inserting user kwargs
    params.update(kwargs)
    # original image
    params.update({'cmap': 'gray', 'alpha': 1})
    plt.imshow(img1, **params)
    # segmentation layer (colored)
    params.update({'alpha': alpha * (img2 > 0)})  # masking alpha (filtering out the background (0) values)
    params.update(kwargs)  # inserting user kwargs (and overwriting cmap for the colored image)
    plt.title(title, fontsize=12)
    plt.suptitle("this method found {} labels".format(img2.max() + 1))
    plt.imshow(img2, **params)
    plt.axis('off')
    plt.tight_layout()

    if save_fig_name:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_fig_name)

    plt.show()
