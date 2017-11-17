# coding=utf-8
import cPickle
import gzip
import numpy as np

def load_data():
    """
    从mnist.pkl.gz中读数据：
    原始数据格式为training_data[0].shape = (50000,784), training_data[1].shape = (50000,1),
    validation_data[0].shape = (10000,784), validation_data[0].shape = (10000,1),
    test_data[0].shape  = (10000,784), validation_data[0].shape = (10000,1)
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    """
    整理数据格式：
    training_inputs每个图片的784个像素信息，(784,1)*50000
    training_result每个图片对应的数字结果转化成10位0，对应的那个数字位为1
    training_data把上面两个结合起来，training_data[0][0]第0个训练数据的像素信息，training_data[0][1]第0个训练数据的数字
    validation_data 和 test_data 的数字结果不用转换成10位，其余操作和training_data 一致
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return training_data, validation_data, test_data


def vectorized_result(j):
    """生成10位list，j所在的那一位为1，其余为0"""
    e = np.zeros((10, 1))
    e[j] = 1
    return e
