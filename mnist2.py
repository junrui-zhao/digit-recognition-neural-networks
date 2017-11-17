# coding=utf-8

import random

from PyQt5.QtGui import QImage


class CrossEntropyCost(object):
    """利用cross-entropy函数计算cost，使得学习速率加快"""
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return (a - y)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """sigmoid函数的导数"""
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        sizes参数代表神经网络的结构，[784,30,10]表示有三层，第一层784个神经元，第二层30个神经元，第三层10个神经元
        biases[l][j]表示l+1层第k个神经元的偏置
        weights[l][j][k]代表第l层第k个神经元到第l+1层第j个神经元的权重
        cost表示指定计算cost的方法
        """
        print sizes
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        """
        高斯分布生成随机weights，均值为0方差为1，随机生成biases
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """
        随机生成biases和weights
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """参数a为输入向量，返回整个网络的输出"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """
        用小批量启发式梯度下降法训练神经网络
        :param training_data: 训练数据
        :param epochs: 迭代次数
        :param mini_batch_size: 分批空间
        :param eta: 训练速度，用来乘以梯度
        :param lmbda: 归一化参数
        :param evaluation_data: 评估数据
        :param monitor_evaluation_cost: 是否监测评估数据的cost
        :param monitor_evaluation_accuracy: 是否监测评估数据的准确数
        :param monitor_training_cost: 是否监测训练数据的cost
        :param monitor_training_accuracy: 是否监测训练数据的准确数
        """
        if evaluation_data: n_data = len(evaluation_data)  # 如果test_data不为空，n_test = test_data的长度
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in xrange(epochs):  # xrange(epochs)生成一个1到epochs的数组, 循环遍历这个数组
            random.shuffle(training_data)  # random.shuffle随机排序
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]  # 把training_data根据size分割，mini_batches每个元素保存分割后的training_data
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))  # 对每一个mini_batch更新weight和biases
            print ("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                training_cost_result = "Cost on training data: {}".format(cost)
                print training_cost_result
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                training_accuracy_result = "Accuracy on training data: {} / {}".format(accuracy, n)
                print training_accuracy_result
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                evaluation_cost_result = "Cost on evaluation data: {}".format(cost)
                print evaluation_cost_result
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                evaluation_accuracy_result = "Accuracy on evalution data: {} / {}".format(self.accuracy(evaluation_data), n_data)
                print evaluation_accuracy_result
            print
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        通过（一小批数据）梯度下降的反向传播算法，更新网络的权值和偏置
        :param mini_batch: （x，y）的list
        :param eta: 学习速率
        :param lmbda: 归一化参数
        :param n: 训练数据的大小
        """
        nable_b = [np.zeros(b.shape) for b in self.biases]
        nable_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nable_b, delta_nable_w = self.backprop(x, y)  # 计算梯度
            # 整个算法的step3
            nable_b = [nb + dnb for nb, dnb in zip(nable_b, delta_nable_b)]
            nable_w = [nw + dnw for nw, dnw in zip(nable_w, delta_nable_w)]
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nable_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nable_b)]

    def backprop(self, x, y):
        """
        反向传播算法
        :return: 梯度和
        """
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
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        # step4: 误差反传
        nable_b[-1] = delta
        nable_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            # step5: 计算梯度
            nable_b[-l] = delta
            nable_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nable_b, nable_w

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def total_cost(self, data, lmbda, convert=False):
        """
        计算某数据集的总代价
        :param data: 数据集
        :param lmbda: 归一化参数
        :param convert: 验证或测试为False，训练为True
        :return: 返回该data set的总代价
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(
            np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def accuracy(self, data, convert=False):
        """
        计算结果正确的数量
        :param data:
        :param convert: 如果计算的是validation_data 或 test_data, convert=False;
                        如果计算的是training_data, convert=True
        :return: 返回结果正确的个数
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


# ui
import load_data
import pyqtgraph as pg
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
import time

EVALUATION_COST, EVALUATION_ACCURACY = [], []
TRAINING_COST, TRAINING_ACCURACY = [], []
EPOCHS = 3
NN = 0
BATCH = 0
LMBDA = 0
ETA = 0

class Mythread(QThread):
    """
    子线程，用来同步刷新计算结果，显示在界面中
    """
    # 定义信号,定义参数为str类型
    _signal = pyqtSignal(str)

    def __init__(self):
        super(Mythread, self).__init__()

    def run(self):

        time1 = time.time()
        self._signal.emit('开始')
        training_data, validation_data, test_data = load_data.load_data_wrapper()
        print NN
        net = Network([784, int(NN), 10])
        epochs = int(EPOCHS)
        mini_batch_size = int(BATCH)
        eta = float(ETA)
        lmbda = float(LMBDA)
        if validation_data: n_data = len(validation_data)  # 如果test_data不为空，n_test = test_data的长度
        n = len(training_data)
        global EVALUATION_COST
        global EVALUATION_ACCURACY
        global TRAINING_COST
        global TRAINING_ACCURACY
        for j in xrange(epochs):  # xrange(epochs)生成一个1到epochs的数组, 循环遍历这个数组
            random.shuffle(training_data)  # random.shuffle随机排序
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]  # 把training_data根据size分割，mini_batches每个元素保存分割后的training_data
            for mini_batch in mini_batches:
                net.update_mini_batch(mini_batch, eta, lmbda, len(training_data))  # 对每一个mini_batch更新weight和biases
            n_epoch = "Epoch %s training complete" % j
            self._signal.emit(n_epoch)

            # if monitor_training_cost:
            cost = net.total_cost(training_data, lmbda)
            TRAINING_COST.append(cost)
            training_cost_result = "Cost on training data: {}".format(cost)
            self._signal.emit(training_cost_result)

            # if monitor_training_accuracy:
            accuracy = net.accuracy(training_data, convert=True)
            training_accuracy_result = "Accuracy on training data: {} / {}".format(accuracy, n)
            TRAINING_ACCURACY.append(accuracy/50000.0)
            self._signal.emit(training_accuracy_result)

            # if monitor_evaluation_cost:
            cost = net.total_cost(validation_data, lmbda, convert=True)
            EVALUATION_COST.append(cost)
            evaluation_cost_result = "Cost on evaluation data: {}".format(cost)
            self._signal.emit(evaluation_cost_result)

            # if monitor_evaluation_accuracy:
            accuracy = net.accuracy(validation_data)
            evaluation_accuracy_result = "Accuracy on evalution data: {} / {}".format(net.accuracy(validation_data), n_data)
            EVALUATION_ACCURACY.append(accuracy/10000.0)
            self._signal.emit(evaluation_accuracy_result)

            self._signal.emit('--------------------')

        time2 = time.time()
        training_time = "Training time: {}".format(time2 - time1)
        time3 = time.time()
        test_result = "Test result: {} / {}".format(net.accuracy(test_data), len(test_data))
        time4 = time.time()
        test_time = "Test time: {}".format(time4 - time3)
        self._signal.emit(training_time)
        self._signal.emit(test_time)
        self._signal.emit(test_result)
        self._signal.emit(str(TRAINING_ACCURACY))
        self._signal.emit(str(EVALUATION_ACCURACY))


class MyWindow(QWidget):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)

        self.setWindowTitle("BP神经网络-MNIST数据集")
        # 窗口元素
        self.btn = QtGui.QPushButton('开始学习')
        # self.btn.clicked.connect(self.on_btn_clicked)
        self.label_nn = QLabel("隐层神经元个数")
        self.edit_nn = QLineEdit()
        self.label_epochs = QLabel("迭代次数")
        self.edit_epochs = QLineEdit()
        self.label_batch = QLabel("批尺寸")
        self.edit_batch = QLineEdit()
        self.label_eta = QLabel("学习速度")
        self.edit_eta = QLineEdit()
        self.label_lmbda = QLabel("归一化参数")
        self.edit_lmbda = QLineEdit()
        self.btn2 = QtGui.QPushButton('设置参数')
        # self.text = QtGui.QTextEdit()
        self.listw = QtGui.QTextEdit()
        self.myplot = pg.PlotWidget()
        self.myplot.plot()  # pen=None, symbol='o'
        self.label_test = QLabel('测试结果')
        self.test_result = QTextEdit()


        # QGridLayout
        layout = QtGui.QGridLayout()
        self.setLayout(layout)

        # 添加元素到窗口相应位置

        layout.addWidget(self.label_nn, 0, 0)  # text edit goes in middle-left
        layout.addWidget(self.edit_nn, 0, 1)
        layout.addWidget(self.label_epochs, 1, 0)  # text edit goes in middle-left
        layout.addWidget(self.edit_epochs, 1, 1)
        layout.addWidget(self.label_batch, 2, 0)  # text edit goes in middle-left
        layout.addWidget(self.edit_batch, 2, 1)
        layout.addWidget(self.label_eta, 3, 0)  # text edit goes in middle-left
        layout.addWidget(self.edit_eta, 3, 1)
        layout.addWidget(self.label_lmbda, 4, 0)  # text edit goes in middle-left
        layout.addWidget(self.edit_lmbda, 4, 1)
        layout.addWidget(self.btn2, 5, 0, 1, 2)  # text edit goes in middle-left
        layout.addWidget(self.btn, 6, 0, 1, 2)  # button goes in upper-left
        layout.addWidget(self.listw, 7, 0, 1, 2)  # list widget goes in bottom-left
        layout.addWidget(self.myplot, 0, 2, 10, 2)  # plot goes on right side, spanning 3 rows
        layout.addWidget(self.label_test, 8, 0, 1, 2)
        layout.addWidget(self.test_result, 9, 0, 1, 2)

        self.epochs = np.arange(EPOCHS)
        self.training_accuracy = np.zeros(EPOCHS)


        # from PIL import Image
        # training_data, validation_data, test_data = load_data.load_data_wrapper()
        # I = training_data[0][0]
        # I.resize((28, 28))
        # im = Image.fromarray((I * 256).astype('uint8'))
        # im.show()
        # self.pic.setPicture(im)
        self.btn2.clicked.connect(self.on_btn2_clicked)
        # self.test_btn.clicked.connect(on_test_btn_clicked)

    def on_btn2_clicked(self):
        global NN
        global EPOCHS
        global BATCH
        global ETA
        global LMBDA
        NN = self.edit_nn.text()
        EPOCHS = self.edit_epochs.text()
        BATCH = self.edit_batch.text()
        ETA = self.edit_eta.text()
        LMBDA = self.edit_lmbda.text()
        self.epochs = np.arange(int(EPOCHS))
        self.training_accuracy = np.zeros(int(EPOCHS))
        print NN, EPOCHS, BATCH, ETA, LMBDA

    # def on_test_btn_clicked(self):


if __name__ == '__main__':
    import sys
    """主线程"""
    app = QApplication(sys.argv)
    app.setApplicationName('MyWindow')
    main = MyWindow()

    def chuli(s):
        if s[0] == '[' :
            main.myplot.addLegend()
            main.myplot.plot(main.epochs, TRAINING_ACCURACY, pen=(19, 234, 201), symbolBrush=(19,234,201), symbolPen='w', symbol='o', symbolSize=10, name="training_accuracy") # , symbol='o'
            main.myplot.plot(main.epochs, EVALUATION_ACCURACY, pen=(237, 177, 32), symbolBrush=(237,177,32), symbolPen='w', symbol='star', symbolSize=10, name="evaluation_accuracy")
        elif 'Training time' in s:
            main.listw.append(s)
        elif 'Test result' in s:
            main.test_result.append(s)
        elif 'Test time' in s:
            main.test_result.append(s)
        else:
            main.listw.append(s)

    @pyqtSlot()
    def on_btn_clicked(self):
        print "Button Clicked!"
        thread._signal.connect(chuli)
        thread.start()

    thread = Mythread()
    main.btn.clicked.connect(on_btn_clicked)
    main.show()
    sys.exit(app.exec_())

