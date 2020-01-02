"""
predict with trained CNN network
Date: Jan 2th, 2020
"""

__author__ = "Yong-Hao Zou"

import pickle

from convolution import *
from tools import *

params = pickle.load(open("save", 'rb'))

[f1, b1, fc_w, fc_b] = params
print(f1)
print(b1)

# predict
images_data = get_test_data()
images_data = np.array(images_data, dtype=float)
label_data = np.array(get_test_label())
images_data -= int(np.mean(images_data))
images_data /= int(np.std(images_data))
test_data = np.hstack((images_data, label_data))

X = test_data[:, 0:-1]
X = X.reshape(len(test_data), 28, 28)
y = test_data[:, -1]

correct = 0
for i in range(len(X)):
    x = X[i]
    # convolution
    convolution_result = convolution(image=x, filter_=f1, bias=b1)
    convolution_result[convolution_result <= 0] = 0
    pooled = maxpool(convolution_result, 3, 3)
    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1))
    out = fc_w.dot(fc) + fc_b
    result = softmax(out)
    predict = np.argmax(result)
    if predict == y[i]:
        correct += 1
    print(predict)
    print(y[i])
    print(correct / (i + 1))