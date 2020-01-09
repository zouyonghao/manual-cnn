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
fc_w1 = (256, 4608)
fc_w1 = initialize_weight(fc_w1)
fc_b1 = np.zeros((fc_w1.shape[0], 1))

fc_w2 = (10, 256)
fc_w2 = initialize_weight(fc_w2)
fc_b2 = np.zeros((fc_w2.shape[0], 1))

# train
# loss跳动
# lr = 0.2
lr = 0.05

batch_size = 32

for epoch in range(3):
    print("epoch: " + str(epoch))
    np.random.shuffle(train_data)
    batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]
    for batch in batches:
        images = batch[:, 0:-1]
        images = images.reshape(len(batch), 28, 28)
        labels = batch[:, -1]

        df1 = np.zeros(f1.shape)
        db1 = np.zeros(b1.shape)
        dfc_w1 = np.zeros(fc_w1.shape)
        dfc_b1 = np.zeros(fc_b1.shape)
        dfc_w2 = np.zeros(fc_w2.shape)
        dfc_b2 = np.zeros(fc_b2.shape)

        loss = 0
        for i in range(batch_size):
            image = images[i]
            # 标签转为onehot
            label = np.eye(10)[int(labels[i])].reshape(10, 1)
            # print(label)

            # 卷积
            convolution_result = convolution(image=image, filter_=f1, bias=b1)

            # ReLU
            convolution_result[convolution_result <= 0] = 0

            # 3x3 maxpool
            pooled = maxpool(convolution_result, 3, 1)

            (depth, size, _) = pooled.shape
            fc = pooled.reshape((depth * size * size, 1))

            # 两个全连接层
            fc1_result = fc_w1.dot(fc) + fc_b1
            fc1_result[fc1_result <= 0] = 0
            fc2_result = fc_w2.dot(fc1_result) + fc_b2

            result = softmax(fc2_result)
            # print(result)

            loss_ = categorical_cross_entropy(result, label)
            # print("loss=" + str(loss))
            loss += loss_

            # 反向传播
            dout = result - label

            # dout[dout < 0] = 0
            dfc_w2_ = dout.dot(fc1_result.T)
            dfc_b2_ = np.sum(dout, axis=1).reshape(fc_b2.shape)

            dfc_1_ = fc_w2.T.dot(dout)
            # 这里注意ReLU的反向传播需要使用之前的结果
            dfc_1_[fc1_result <= 0] = 0

            dfc_w1_ = dfc_1_.dot(fc.T)
            dfc_b1_ = np.sum(dfc_1_, axis=1).reshape(fc_b1.shape)

            dfc = fc_w1.T.dot(dfc_1_)

            dpool = dfc.reshape(pooled.shape)

            dconv = maxpool_backward(dpool, convolution_result, 3, 1)

            dconv[convolution_result <= 0] = 0

            _, df1_, db1_ = convolution_backward(dconv, image, f1)

            df1 += df1_
            db1 += db1_
            # print(dfc_w.shape)
            # print(dfc_w_.shape)
            dfc_w1 += dfc_w1_
            dfc_b1 += dfc_b1_
            dfc_w2 += dfc_w2_
            dfc_b2 += dfc_b2_

        print("loss = " + str(loss / batch_size))
        # print(df1)
        # print(dfc_w1)
        # print(dfc_w2)
        # f1 -= lr * df1
        # b1 -= lr * db1
        # fc_w1 -= lr * dfc_w1
        # fc_b1 -= lr * dfc_b1
        # fc_w2 -= lr * dfc_w2
        # fc_b2 -= lr * dfc_b2

        # 更新梯度
        f1 -= lr * df1 / batch_size
        b1 -= lr * db1 / batch_size
        fc_w1 -= lr * dfc_w1 / batch_size
        fc_b1 -= lr * dfc_b1 / batch_size
        fc_w2 -= lr * dfc_w2 / batch_size
        fc_b2 -= lr * dfc_b2 / batch_size

params = [f1, b1, fc_w1, fc_b1, fc_w2, fc_b2]

pickle.dump(params, open("save", "wb"))
