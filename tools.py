# -*- coding: utf-8 -*-
"""manual-cnn"""

__author__ = 'Yong-Hao Zou'

import struct

import numpy as np


def _read_idx3(name):
    train_file = open(name, "rb")
    buf = train_file.read()
    index = struct.calcsize('>IIII')
    data = []

    while index < len(buf):
        tmp = struct.unpack_from('>784B', buf, index)
        # im.append(np.reshape(tmp, (28, 28)))
        data.append(tmp)
        index += struct.calcsize('>784B')

    return data


def _read_idx1(name):
    train_label_file = open(name, "rb")
    buf = train_label_file.read()
    index = struct.calcsize('>II')
    label = []

    while index < len(buf):
        tmp = struct.unpack_from('>B', buf, index)
        label.append(tmp)
        # print(tmp)
        index += struct.calcsize('>B')

    return label


def get_train_data():
    return _read_idx3("train-images.idx3-ubyte")


def get_train_label():
    return _read_idx1("train-labels.idx1-ubyte")


def get_test_data():
    return _read_idx3("t10k-images.idx3-ubyte")


def get_test_label():
    return _read_idx1("t10k-labels.idx1-ubyte")

#
# def initialize_filter(size, scale=1.0):
#     stddev = scale / np.sqrt(np.prod(size))
#     return np.random.normal(loc=0, scale=stddev, size=size)


def initialize_weight(size):
    return np.random.standard_normal(size=size) * 0.01


if __name__ == "__main__":
    test_data = get_test_data()
    print(len(test_data))

    test_label = get_test_label()
    print(len(test_label))
