"""
train CNN network
Date: Dec 26th, 2019
"""

__author__ = "Yong-Hao Zou"

import pickle

from convolution import *
from tools import *

# if __name__ == '__main__':
# load data

images_data = get_train_data()
images_data = np.array(images_data, dtype=float)
label_data = np.array(get_train_label())
images_data -= int(np.mean(images_data))
images_data /= int(np.std(images_data))
print(images_data.shape)
print(label_data.shape)
train_data = np.hstack((images_data, label_data))
print(train_data.shape)

# filter 1
f1 = (8, 3, 3)
f1 = initialize_weight(f1)
b1 = np.zeros((f1.shape[0], 1))

# fc
fc_w = (10, 512)
fc_w = initialize_weight(fc_w)
fc_b = np.zeros((fc_w.shape[0], 1))

# train
lr = 0.01
batch_size = 32
batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

for batch in batches:
    images = batch[:, 0:-1]
    images = images.reshape(len(batch), 28, 28)
    labels = batch[:, -1]

    # initialize gradients
    df1 = np.zeros(f1.shape)
    db1 = np.zeros(b1.shape)
    dfc_w = np.zeros(fc_w.shape)
    dfc_b = np.zeros(fc_b.shape)

    for i in range(batch_size):
        image = images[i]
        # one-hot
        label = np.eye(10)[int(labels[i])].reshape(10, 1)

        # convolution
        convolution_result = convolution(image=image, filter_=f1, bias=b1)

        # ReLU
        convolution_result[convolution_result <= 0] = 0

        # 3x3 maxpool
        pooled = maxpool(convolution_result, 3, 3)

        (nf2, dim2, _) = pooled.shape
        fc = pooled.reshape((nf2 * dim2 * dim2, 1))

        # fully connection
        out = fc_w.dot(fc) + fc_b

        result = softmax(out)

        loss = categorical_cross_entropy(result, label)
        print("loss=" + str(loss))

        # back propagation
        dout = result - label

        # dout[dout < 0] = 0
        dfc_w_ = dout.dot(fc.T)
        dfc_b_ = np.sum(dout, axis=1).reshape(fc_b.shape)

        dfc = fc_w.T.dot(dout)

        dpool = dfc.reshape(pooled.shape)

        dconv = maxpool_backward(dpool, convolution_result, 3, 3)

        # backpropagate through ReLU
        dconv[dconv <= 0] = 0

        _, df1_, db1_ = convolution_backward(dconv, image, f1)

        df1 += df1_
        db1 += db1_
        # print(dfc_w.shape)
        # print(dfc_w_.shape)
        dfc_w += dfc_w_
        dfc_b += dfc_b_

    f1 -= lr * df1 / batch_size
    b1 -= lr * db1 / batch_size
    fc_w -= lr * dfc_w / batch_size
    fc_b -= lr * dfc_b / batch_size

params = [f1, b1, fc_w, fc_b]

pickle.dump(params, open("save", "wb"))
