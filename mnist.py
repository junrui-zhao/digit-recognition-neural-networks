# coding=utf-8

import numpy as np
import random



def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """sigmoid函数的导数"""
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    def __init__(self, sizes):
        """
        sizes参数代表神经网络的结构，[784,30,10]表示有三层，第一层784个神经元，第二层30个神经元，第三层10个神经元
        biases[l][j]表示l+1层第k个神经元的偏置
        weights[l][j][k]代表第l层第k个神经元到第l+1层第j个神经元的权重
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """参数a为输入向量，返回整个网络的输出"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)  # 如果test_data不为空，n_test = test_data的长度
        n = len(training_data)
        for j in xrange(epochs):  # xrange(epochs)生成一个1到epochs的数组, 循环遍历这个数组
            random.shuffle(training_data)  # random.shuffle随机排序
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]  # 把training_data根据size分割，mini_batches每个元素保存分割后的training_data
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)  # 对每一个mini_batch更新weight和biases
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        nable_b = [np.zeros(b.shape) for b in self.biases]
        nable_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nable_b, delta_nable_w = self.backprop(x, y)  # 计算梯度
            # 整个算法的step3
            nable_b = [nb + dnb for nb, dnb in zip(nable_b, delta_nable_b)]
            nable_w = [nw + dnw for nw, dnw in zip(nable_w, delta_nable_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nable_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nable_b)]

    def backprop(self, x, y):
        nable_b = [np.zeros(b.shape) for b in self.biases]
        nable_w = [np.zeros(w.shape) for w in self.weights]
        # step1: 输入a
        activation = x
        activations = [x]  # 存储所有a值的列表
        zs = []  # 存储所有z值的列表
        # step2向前计算
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # step3: 计算误差
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        # step4: 误差反传
        nable_b[-1] = delta
        nable_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            # step5: 计算梯度(偏导)
            nable_b[-l] = delta
            nable_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nable_b, nable_w

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

# 测试运行
import load_data

training_data, validation_data, test_data = load_data.load_data_wrapper()
net = Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)