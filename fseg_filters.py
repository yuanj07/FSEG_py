import numpy as np
import math
from scipy import ndimage


def log_filter(sgm, fsize):
    """
    LoG filter
    :param sgm: sigma in Gaussian
    :param fsize: filter size, [h, w]
    :return: LoG filter
    """
    wins_x = fsize[1] / 2
    wins_y = fsize[0] / 2

    out = np.zeros(fsize, dtype=np.float32)

    for x in range(-wins_x, wins_x+1):
        for y in range(-wins_y, wins_y+1):
            out[wins_y+y, wins_x+x] = - 1. / (math.pi * sgm**4.) * (1. - (x*x+y*y)/(2.*sgm*sgm)) * math.exp(-(x*x+y*y)/(2.*sgm*sgm))

    return out-np.mean(out)


def gabor_filter(sgm, theta):
    """
    Gabor filter
    :param sgm: sigma in Gaussian
    :param theta: direction
    :return: gabor filter
    """
    phs=0
    gamma=1
    wins=int(math.floor(sgm*2))
    f=1/(sgm*2.)
    out=np.zeros((2*wins+1, 2*wins+1))

    for x in range(-wins, wins+1):
        for y in range(-wins, wins+1):
            xPrime = x * math.cos(theta) + y * math.sin(theta)
            yPrime = y * math.cos(theta) - x * math.sin(theta)
            out[wins+y, wins+x] = 1/(2*math.pi*sgm*sgm)*math.exp(-.5*((xPrime)**2+(yPrime*gamma)**2)/sgm**2)*math.cos(2*math.pi*f*xPrime+phs)
    return out-np.mean(out)


def image_filtering(img, filter_list):
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
