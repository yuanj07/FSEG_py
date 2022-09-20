"""

Factorization based segmentation with manually selected seeds

"""
import time
import numpy as np
import argparse
import cv2

from scipy import linalg as LAsci
from utils.fseg_filters import image_filtering, overlay


def SHcomp(Ig, ws, BinN=11):
    """
    Compute local spectral histogram using integral histograms
    :param Ig: a n-band image
    :param ws: half window size
    :param BinN: number of bins of histograms
    :return: local spectral histogram at each pixel
    """
    h, w, bn = Ig.shape
    ws = int(ws)

    # quantize values at each pixel into bin ID
    for i in range(bn):
        b_max = np.max(Ig[:, :, i])
        b_min = np.min(Ig[:, :, i])
        assert b_max != b_min, "Band %d has only one value!" % i

        b_interval = (b_max - b_min) * 1. / BinN
        Ig[:, :, i] = np.floor((Ig[:, :, i] - b_min) / b_interval)

    Ig[Ig >= BinN] = BinN - 1
    Ig = np.int32(Ig)

    # convert to one hot encoding
    one_hot_pix = []
    for i in range(bn):
        one_hot_pix_b = np.zeros((h * w, BinN), dtype=np.int32)
        one_hot_pix_b[np.arange(h * w), Ig[:, :, i].flatten()] = 1
        one_hot_pix.append(one_hot_pix_b.reshape((h, w, BinN)))

    # compute integral histogram
    integral_hist = np.concatenate(one_hot_pix, axis=2)

    np.cumsum(integral_hist, axis=1, out=integral_hist)
    np.cumsum(integral_hist, axis=0, out=integral_hist)

    # compute spectral histogram based on integral histogram
    padding_l = np.zeros((h, ws + 1, BinN * bn), dtype=np.int32)
    padding_r = np.tile(integral_hist[:, -1:, :], (1, ws, 1))

    integral_hist_pad_tmp = np.concatenate([padding_l, integral_hist, padding_r], axis=1)

    padding_t = np.zeros((ws + 1, integral_hist_pad_tmp.shape[1], BinN * bn), dtype=np.int32)
    padding_b = np.tile(integral_hist_pad_tmp[-1:, :, :], (ws, 1, 1))

    integral_hist_pad = np.concatenate([padding_t, integral_hist_pad_tmp, padding_b], axis=0)

    integral_hist_1 = integral_hist_pad[ws + 1 + ws:, ws + 1 + ws:, :]
    integral_hist_2 = integral_hist_pad[:-ws - ws - 1, :-ws - ws - 1, :]
    integral_hist_3 = integral_hist_pad[ws + 1 + ws:, :-ws - ws - 1, :]
    integral_hist_4 = integral_hist_pad[:-ws - ws - 1, ws + 1 + ws:, :]

    sh_mtx = integral_hist_1 + integral_hist_2 - integral_hist_3 - integral_hist_4

    histsum = np.sum(sh_mtx, axis=-1, keepdims=True) * 1. / bn

    sh_mtx = np.float32(sh_mtx) / np.float32(histsum)

    return sh_mtx


def Fseg(Ig, ws, seeds):
    """
    Factorization based segmentation
    :param Ig: a n-band image
    :param ws: window size for local special histogram
    :param seeds: list of coordinates [row, column] for seeds. each seed represent one type of texture
    :param omega: error threshod for estimating segment number. need to adjust for different filter bank.
    :param nonneg_constraint: whether apply negative matrix factorization
    :return: segmentation label map
    """

    N1, N2, bn = Ig.shape

    ws = ws / 2
    sh_mtx = SHcomp(Ig, ws)

    Z = []
    for seed in seeds:
        Z.append(sh_mtx[seed[0], seed[1], :].reshape((-1, 1)))

    Z = np.hstack(Z)

    Y = sh_mtx.reshape((N1 * N2, -1))

    ZZTinv = LAsci.inv(np.dot(Z.T, Z))
    Beta = np.dot(np.dot(ZZTinv, Z.T), Y.T)

    seg_label = np.argmax(Beta, axis=0)

    return seg_label.reshape((N1, N2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", help="file path")
    parser.add_argument("-ws", "--window_size", help="window size for local special histogram")
    parser.add_argument("-s", "--seeds",
                        help="list of coordinates [row, column] for seeds. each seed represent one type of texture")
    parser.add_argument("-save_dir", "--save_dir", help="path with the folder to save the file")
    parser.add_argument("-save_file_name", "--save_file_name",
                        help="file name with the extension to save the final result")
    args = parser.parse_args()

    file_path = args.file
    ws = int(args.window_size)
    #seeds = args.seeds
    save_dir = args.save_dir
    save_file_name = args.save_file_name

    time0 = time.time()
    # an example of using Fseg
    img = cv2.imread(file_path, 0)

    # define filter bank and apply to image. for color images, convert rgb to grey scale and then apply filter bank
    filter_list = [('log', .5, [3, 3]), ('log', 1.2, [7, 7])]

    filter_out = image_filtering(img, filter_list=filter_list)

    # include original image as one band
    Ig = np.concatenate((np.float32(img.reshape((img.shape[0], img.shape[1], 1))), filter_out), axis=2)

    seeds = [[60, 238], [160, 160], [238, 60]]  # provide seeds

    # run segmentation. try different window size
    seg_out = Fseg(Ig, ws=ws, seeds=seeds)

    print('FSEG runs in %0.2f seconds. ' % (time.time() - time0))
    title = "Plot using ws={}".format(ws)

    # show results
    overlay(img, seg_out, 0.6, cmap="viridis", save_fig=save_dir + save_file_name, save_dir=save_dir, plot_title=title)
