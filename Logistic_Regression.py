# -*- coding: UTF-8 -*-
"""
@Project ：PycharmProjects
@File    ：Logistic_Regression.py
@Author  ：guo
@Date    ：2022/9/20 下午9:42
"""
import numpy as np
from sklearn import metrics


class Logistic_Regression:
    """
    逻辑回归（二分类算法）
    """

    def __init__(self, input_data, result, theta=None):
        """
        构造函数
        @param input_data: 训练数据集
        @param result: 训练数据的结果
        """
        self.train_data = input_data
        self.train_result = result
        self.theta = theta
        if self.theta is None:
            size = self.train_data.shape[1]
            self.theta = np.random.randn(size)

    def Sigmoid(self, x):
        """
        对数几率函数，也叫Sigmoid函数，值域0~1
        @param x:自变量
        @return:函数值
        """
        return 1 / (1 + np.exp(-x))

    def Shuffle_Sequence(self):
        """
        打乱数据集
        @return: None
        """
        length = self.train_data.shape[0]
        index = np.random.permutation(np.arange(length))
        self.train_data = self.train_data[index]
        self.train_result = self.train_result[index]

    def BGD(self, alpha):
        """
        BGD(梯度下降算法)
        @param alpha:学习率
        @return: None
        """
        pradient_increasement = []
        predict = self.Sigmoid(self.train_data.dot(self.theta))
        for i in range(self.train_data.shape[0]):
            pradient_increasement.append((self.train_result[i] - predict[i]) * self.train_data[i])
        avg_g = np.average(pradient_increasement, 0)
        self.theta = self.theta + alpha * avg_g

    def SGD(self, alpha):
        """
        SGD(随机梯度下降算法)
        @param alpha: 学习率
        @return: None
        """
        self.Shuffle_Sequence()
        length = self.train_data.shape[0]
        for i in range(length):
            predict = self.Sigmoid(self.train_data.dot(self.theta))
            grad = (self.train_result[i] - predict[i]) * self.train_data[i]
            self.theta = self.theta + alpha * grad

    def MBGD(self, alpha, batch_size):
        """
        MBGD(小批量梯度下降算法)
        @param alpha: 学习率
        @param batch_size: 小批量样本范围
        @return: None
        """
        self.Shuffle_Sequence()
        for start in np.arange(0,len(self.train_data),batch_size):
            pradient_increasement = []
            # 判断start+batch_size是否大于数组长度，
            # 防止最后一组小样本规模可能小于batch_size的情况
            end = np.min([start + batch_size, len(self.train_data)])
            mini_train_data = self.train_data[start:end]
            mini_train_result = self.train_result[start:end]
            predict = self.Sigmoid(mini_train_data.dot(self.theta))
            for i in range(predict.shape[0]):
                pradient_increasement.append((mini_train_result[i] - predict[i]) * mini_train_data[i])
            avg_g = np.average(pradient_increasement,0)
            self.theta = self.theta + alpha * avg_g

    def cost(self):
        """
        损失函数
        :return:损失量
        """
        predict = self.Sigmoid(self.train_data.dot(self.theta))
        cost = []
        for y, h in zip(self.train_result, predict):
            # 防止出现真数为0的情况
            cost.append(-(y * np.log(h + 1e-6) + (1 - y) * np.log(1 - h + 1e-6)))
        return np.average(cost)

    def train_BGD(self, times, alpha, test_data, test_result):
        """
        利用BGD算法迭代
        :param times:迭代次数
        :param alpha: 学习率
        :param test_data: 测试数据
        :param test_result: 测试数据的实际结果
        :return: 每次迭代后的损失量，每次迭代后的正确率
        """
        Cost = []
        AC = []
        for i in range(times):
            self.BGD(alpha)
            Cost.append(self.cost())
            AC.append(self.ac(test_data, test_result))
        return Cost, AC

    def train_SGD(self,times,alpha,test_data,test_result):
        """
        利用SGD算法进行迭代
        @param times: 迭代次数
        @param alpha: 学习率
        @param test_data: 测试数据集
        @param test_result: 测试数据集对应的实际结果
        @return: 每次迭代后的损失量，每次迭代后的正确率
        """
        Cost = []
        AC = []
        for i in range(times):
            self.SGD(alpha)
            Cost.append(self.cost())
            AC.append(self.ac(test_data, test_result))
        return Cost, AC

    def train_MBGD(self,times,alpha,test_data,test_result,batch_size):
        """
        利用MBGD算法进行迭代
        @param times: 迭代次数
        @param alpha: 学习率
        @param test_data: 测试数据集
        @param test_result: 测试数据集对应的实际结果
        @param batch_size: 小批量样本范围
        @return: 每次迭代后的损失量，每次迭代后的正确率
        """
        Cost = []
        AC = []
        for i in range(times):
            self.MBGD(alpha,batch_size)
            Cost.append(self.cost())
            AC.append(self.ac(test_data, test_result))
        return Cost, AC



    def test(self, test_data):
        pred = test_data.dot(self.theta)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return pred

    def ac(self, test_data, test_result):
        pred = self.test(test_data)
        ac = metrics.accuracy_score(test_result, pred)
        return ac
