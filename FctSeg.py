"""

Factorization based segmentation

"""
import time
import numpy as np
import argparse

from scipy import linalg as LAsci
from numpy import linalg as LA
from utils.fseg_filters import *


def _SHcomp(Ig: np.ndarray, ws: int, BinN: int = 11) -> np.ndarray:
    """
    Function that compute the local spectral histogram using integral histograms
    Args:
        Ig (np.ndarray): a n-band image
        ws (int): half window size
        BinN (np.ndarray): number of bins of histograms

    Returns:
        (np.ndarray): local spectral histogram at each pixel

    """
    h, w, bn = Ig.shape

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

    np.cumsum(integral_hist, axis=1, out=integral_hist, dtype=np.float32)
    np.cumsum(integral_hist, axis=0, out=integral_hist, dtype=np.float32)

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


def _SHedgeness(sh_mtx: np.ndarray, ws: int) -> np.ndarray:
    """
    Function that create the edge map

    Args:
        sh_mtx (np.ndarray): vector with the project features onto the subspace
        ws (int): window size for the feature maps

    Returns:
        (np.ndarray): returns a numpy array that contains all the edges on the image

    """
    h, w, _ = sh_mtx.shape
    edge_map = np.ones((h, w)) * -1
    for i in range(ws, h - ws - 1):
        for j in range(ws, w - ws - 1):
            edge_map[i, j] = np.sqrt(np.sum((sh_mtx[i - ws, j, :] - sh_mtx[i + ws, j, :]) ** 2)
                                     + np.sum((sh_mtx[i, j - ws, :] - sh_mtx[i, j + ws, :]) ** 2))
    return edge_map


def _Fseg(Ig: np.ndarray, ws: int, segn: int, omega: float, nonneg_constraint: bool = True) -> np.ndarray:
    """
    Factorization based segmentation function

    Args:
        Ig (np.ndarray): a n-band image
        ws (int): window size for local special histogram
        segn (int): number of segment. if set to 0, the number will be automatically estimated
        omega (float): error threshod for estimating segment number. need to adjust for different filter bank.
        nonneg_constraint (bool): whether apply negative matrix factorization

    Returns:
        (np.ndarray): retunrs the label mask as a numpy array

    """
    N1, N2, bn = Ig.shape

    ws //= 2
    sh_mtx = _SHcomp(Ig, ws)
    sh_dim = sh_mtx.shape[2]

    Y = (sh_mtx.reshape((N1 * N2, sh_dim)))
    S = np.dot(Y.T, Y)
    d, v = LA.eig(S)

    d_sorted = np.sort(d)
    idx = np.argsort(d)
    k = np.abs(d_sorted)

    if segn == 0:  # estimate the segment number
        lse_ratio = np.cumsum(k) * 1. / (N1 * N2)
        print(lse_ratio)
        print(np.sum(k) / (N1 * N2))
        segn = np.sum(lse_ratio > omega)
        print('Estimated segment number: %d' % segn)

        if segn <= 1:
            # TODO : I can change were for a better warning
            segn = 2
            print('Warning: Segment number is set to 2. May need to reduce omega for better segment number estimation.')

    dimn = segn

    U1 = v[:, idx[-1:-dimn - 1:-1]]

    Y1 = np.dot(Y, U1)  # project features onto the subspace

    edge_map = _SHedgeness(Y1.reshape((N1, N2, dimn)), ws)

    edge_map_flatten = edge_map.flatten()

    Y_woedge = Y1[(edge_map_flatten >= 0) & (edge_map_flatten <= np.max(edge_map) * 0.4), :]

    # find representative features using clustering
    cls_cen = np.zeros((segn, dimn), dtype=np.float32)
    L = np.sum(Y_woedge ** 2, axis=1)
    cls_cen[0, :] = Y_woedge[np.argmax(L), :]  # find the first initial center

    D = np.sum((cls_cen[0, :] - Y_woedge) ** 2, axis=1)
    cls_cen[1, :] = Y_woedge[np.argmax(D), :]

    cen_id = 1
    while cen_id < segn - 1:
        cen_id += 1
        D_tmp = np.zeros((cen_id, Y_woedge.shape[0]), dtype=np.float32)
        for i in range(cen_id):
            D_tmp[i, :] = np.sum((cls_cen[i, :] - Y_woedge) ** 2, axis=1)
        D = np.min(D_tmp, axis=0)
        cls_cen[cen_id, :] = Y_woedge[np.argmax(D), :]

    D_cen2all = np.zeros((segn, Y_woedge.shape[0]), dtype=np.float32)
    cls_cen_new = np.zeros((segn, dimn), dtype=np.float32)
    is_converging = 1
    while is_converging:
        for i in range(segn):
            D_cen2all[i, :] = np.sum((cls_cen[i, :] - Y_woedge) ** 2, axis=1)

        cls_id = np.argmin(D_cen2all, axis=0)

        for i in range(segn):
            cls_cen_new[i, :] = np.mean(Y_woedge[cls_id == i, :], axis=0)

        if np.max((cls_cen_new - cls_cen) ** 2) < .00001:
            is_converging = 0
        else:
            cls_cen = cls_cen_new * 1.
    cls_cen_new = cls_cen_new.T

    ZZTinv = LAsci.inv(np.dot(cls_cen_new.T, cls_cen_new))
    Beta = np.dot(np.dot(ZZTinv, cls_cen_new.T), Y1.T)

    seg_label = np.argmax(Beta, axis=0)

    if nonneg_constraint:
        w0 = np.dot(U1, cls_cen_new)
        dnorm0 = 1

        h = Beta * 1.
        for i in range(100):
            tmp, _, _, _ = LA.lstsq(np.dot(w0.T, w0) + np.eye(segn) * .01, np.dot(w0.T, Y.T))
            h = np.maximum(0, tmp)
            tmp, _, _, _ = LA.lstsq(np.dot(h, h.T) + np.eye(segn) * .01, np.dot(h, Y))
            w = np.maximum(0, tmp)
            w = w.T * 1.

            d = Y.T - np.dot(w, h)
            dnorm = np.sqrt(np.mean(d * d))
            print(i, np.abs(dnorm - dnorm0), dnorm)
            if np.abs(dnorm - dnorm0) < .1:
                break

            w0 = w * 1.
            dnorm0 = dnorm * 1.

        seg_label = np.argmax(h, axis=0)

    return seg_label.reshape((N1, N2))


def run_fct_seg(img: np.ndarray, ws: int, n_segments: int, omega: float, nonneg_constraint: bool, save_dir: str,
                save_file_name: str) -> np.ndarray:
    """
    Function to run the fct and segment an image

    Args:
        img (np.ndarray): image array
        ws (int): window size to use as local special histogram
        n_segments (int): number of segment. if set to 0, the number will be automatically estimated
        omega (float): error threshod for estimating segment number. need to adjust for different filter bank.
        nonneg_constraint (bool): flag that if True, will apply the negative matrix factorization
        save_dir (str): string to save into that directory
        save_file_name (str): string to save the file name

    Returns:

    """
    time0 = time.time()
    # TODO : Need to implement a efficient way to show a loading bar
    # an example of using Fseg
    # z = np.ones(shape=(3, 2))
    # x = 1
    # y = 0
    # actual_inter = (x * z.shape[1]) + (y + 1)
    # end_inter = (z.shape[0] * z.shape[1]) - actual_inter + 1
    # actual_inter = 0
    # progress_bar(actual_inter, end_inter)
    # for i in range(x, z.shape[0]):
    #     for j in range(y, z.shape[1]):
    #         progress_bar(actual_inter + 1, end_inter)
    #         actual_inter += 1
    # This is just a test for the loading bar
    # numbers = [x * 5 for x in range(2000, 3000)]
    # result = []
    # progress_bar(0, len(numbers))
    # for i, x in enumerate(numbers):
    #     result.append(math.factorial(x))
    #     progress_bar(i + 1, len(numbers))

    # define filter bank and apply to image. for color images, convert rgb to grey scale and then apply filter bank
    filter_list = [('log', .5, [3, 3]), ('log', 1, [5, 5]),
                   ('gabor', 1.5, 0), ('gabor', 1.5, np.pi / 2), ('gabor', 1.5, np.pi / 4),
                   ('gabor', 1.5, -np.pi / 4),
                   ('gabor', 2.5, 0), ('gabor', 2.5, np.pi / 2), ('gabor', 2.5, np.pi / 4),
                   ('gabor', 2.5, -np.pi / 4)
                   ]

    filter_out = image_filtering(img, filter_list=filter_list)

    # include original image as one band
    Ig = np.concatenate((np.float32(img.reshape((img.shape[0], img.shape[1], 1))), filter_out), axis=2)

    # run segmentation. try different window size, with and without nonneg constraints
    # seg_out = Fseg(Ig, ws=25, segn=0, omega=.045, nonneg_constraint=True) -> This's a good parameter to run
    seg_out = _Fseg(Ig, ws=ws, segn=n_segments, omega=omega, nonneg_constraint=nonneg_constraint)

    print('FSEG runs in %0.2f seconds. ' % (time.time() - time0))
    title = "Plot using ws={}, segn={}, omega={} and nonneg_flag={}".format(ws, n_segments, omega, nonneg_constraint)

    # show results
    overlay(img, seg_out, 0.6, cmap="viridis", save_fig=save_dir + save_file_name, save_dir=save_dir, plot_title=title)
    return seg_out


if __name__ == '__main__':
    """
    Main function method to run via terminal
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="file path")
    parser.add_argument("-shape_size", "--shape_size", nargs="+", type=int, help="shape size of the image")
    parser.add_argument("-dtype", "--dtype", nargs="?", type=str, help="image dtype")
    parser.add_argument("-ws", "--window_size", type=int, help="window size for local special histogram")
    parser.add_argument("-segn", "--number_segments", type=int,
                        help="number of segment. if set to 0, the number will be automatically estimated")
    parser.add_argument("-omega", "--error_treshold", type=float,
                        help="error threshod for estimating segment number. need to adjust for different filter bank.")
    parser.add_argument("-nonneg_constraint", "--nonneg_constraint", type=bool,
                        help="whether apply negative matrix factorization")
    parser.add_argument("-save_dir", "--save_dir", type=str, help="path with the folder to save the file")
    parser.add_argument("-save_file_name", "--save_file_name", type=str,
                        help="file name with the extension to save the final result")
    args = parser.parse_args()

    file_path = args.file
    img_shape = tuple(args.shape_size)
    dtype = args.dtype
    ws = args.window_size
    n_segments = args.number_segments
    omega = args.error_treshold
    nonneg_constraint = args.nonneg_constraint
    save_dir = args.save_dir
    save_file_name = args.save_file_name

    img = io_from_prompt(file_path, img_shape, dtype)
    _ = run_fct_seg(img, ws, n_segments, omega, nonneg_constraint, save_dir, save_file_name)
