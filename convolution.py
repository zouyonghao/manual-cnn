"""
calc convolution of an image
Date: Dec 26th, 2019
"""

__author__ = "Yong-Hao Zou"

import numpy as np


# image 28*28
# filter 8*3*3
# bias 8*1
# stride 1
def convolution(image, filter_, bias, stride=1):
    (filters, filter_size, _) = filter_.shape
    in_dim, _ = image.shape

    out_dim = int((in_dim - filter_size) / stride) + 1

    out = np.zeros((filters, out_dim, out_dim))

    for curr_filter in range(filters):
        curr_y = out_y = 0
        while curr_y + filter_size <= in_dim:
            curr_x = out_x = 0
            while curr_x + filter_size <= in_dim:
                # print(curr_filter)
                # print(curr_x)
                # print(curr_y)
                out[curr_filter, out_y, out_x] = \
                    np.sum(filter_[curr_filter] * image[curr_y:curr_y + filter_size, curr_x:curr_x + filter_size]) \
                    + bias[curr_filter]
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1

    return out


def convolution_backward(dconv_prev, conv_in, filter_, stride=1):
    """
    Backpropagation through a convolutional layer.
    """
    # (filters, filter_size, _) = filter_.shape
    # in_dim, _ = image.shape
    (n_f, f, _) = filter_.shape
    (orig_dim, _) = conv_in.shape
    ## initialize derivatives
    dout = np.zeros(conv_in.shape)
    dfilt = np.zeros(filter_.shape)
    dbias = np.zeros((n_f, 1))
    for curr_f in range(n_f):
        # loop through all filters
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # loss gradient of filter (used to update the filter)
                dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[curr_y:curr_y + f, curr_x:curr_x + f]
                # loss gradient of the input to the convolution operation (conv1 in the case of this network)
                dout[curr_y:curr_y + f, curr_x:curr_x + f] += dconv_prev[curr_f, out_y, out_x] * filter_[curr_f]
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1
        # loss gradient of the bias
        dbias[curr_f] = np.sum(dconv_prev[curr_f])

    return dout, dfilt, dbias


def maxpool(image, f=2, s=2):
    """
    Downsample `image` using kernel size `f` and stride `s`
    """
    n_c, h_prev, w_prev = image.shape

    h = int((h_prev - f) / s) + 1
    w = int((w_prev - f) / s) + 1

    downscaled = np.zeros((n_c, h, w))
    for i in range(n_c):
        # slide maxpool window over each part of the image and assign the max value at each step to the output
        curr_y = out_y = 0
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            while curr_x + f <= w_prev:
                downscaled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y + f, curr_x:curr_x + f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return downscaled


def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs


def maxpool_backward(dpool, orig, f, s):
    """
    Backpropagation through a maxpooling layer. The gradients are passed through the indices of greatest value in the
    original maxpooling during the forward step.
    """
    (n_c, orig_dim, _) = orig.shape

    dout = np.zeros(orig.shape)

    for curr_c in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # obtain index of largest value in input for current window
                (a, b) = nanargmax(orig[curr_c, curr_y:curr_y + f, curr_x:curr_x + f])
                dout[curr_c, curr_y + a, curr_x + b] = dpool[curr_c, out_y, out_x]

                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

    return dout


def softmax(x):
    out = np.exp(x)
    return out / np.sum(out)


def categorical_cross_entropy(probs, label):
    return -np.sum(label * np.log(probs))
