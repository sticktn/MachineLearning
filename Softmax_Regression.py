# -*- coding: UTF-8 -*-
"""
@Project ：PycharmProjects \n
@File    ：Softmax_Regression.py \n
@Author  ：guo \n
@Date    ：2022/9/24 下午8:30 \n
"""
import numpy as np


def change_onehot(result: np.ndarray, size: np.ndarray):
    """
    将每一个结果转化成onehot标签
    :param size: 分类的个数
    :param result: 分类类型
    :return: 分类矩阵
    """
    length = len(result)
    arr = np.zeros((length, size))
    arr[result == 1] = np.array([1, 0, 0, 0])
    arr[result == 2] = np.array([0, 1, 0, 0])
    arr[result == 3] = np.array([0, 0, 1, 0])
    arr[result == 4] = np.array([0, 0, 0, 1])
    return arr


class Softmax_Regression:
    """
     softmax回归（多分类）
    """

    def __init__(self, train_data: np.ndarray, train_result: np.ndarray, theta: np.ndarray = None):
        """
        softmax回归的构造函数
        :param train_data: 训练数据集
        :param train_result: 训练数据结果
        :param theta: 回归系数
        """
        self.train_data = train_data
        self.train_result = train_result
        self.theta = theta

    def softmax(self, x: np.ndarray):
        """
        softmax函数
        :param x: 自变量
        :return: 函数值
        """
        sum = np.sum(np.exp(x))
        return np.exp(x) / sum

    def Shuffle_Sequence(self):
        """
        打乱数据集
        :return: None
        """
        length = self.train_data.shape[0]
        index = np.random.permutation(np.arange(length))
        self.train_data = self.train_data[index]
        self.train_result = self.train_result[index]

    def cost(self):
        """
        计算模型平均训练损失
        :return: 平均损失量
        """
        Cost = []
        for (data, result) in zip(self.train_data, self.train_result):
            predict = self.softmax(data.dot(self.theta))
            cost = - result * np.log(predict + 1e-6)
            Cost.append(np.sum(cost))
        return np.average(Cost)

    def BGD(self, alpha: float):
        """
        BGD(梯度下降算法)
        :param alpha: 学习率
        :return: None
        """
        gradient_increasment = []
        for (data, result) in zip(self.train_data, self.train_result):
            predict = self.softmax(data.dot(self.theta))
            error = result - predict
            data = np.reshape(data, (1, len(data)))
            error = np.reshape(error, (1, len(error)))
            g = data.T.dot(error)
            gradient_increasment.append(g)
        avg_g = np.average(gradient_increasment, 0)
        self.theta = self.theta + alpha * avg_g

    def SGD(self, alpha: float):
        """
        SGD(随机梯度下降算法)
        :param alpha: 学习率
        :return: None
        """
        self.Shuffle_Sequence()
        for (data, result) in zip(self.train_data, self.train_result):
            predict = self.softmax(data.dot(self.theta))
            error = result - predict
            data = np.reshape(data, (1, len(data)))
            error = np.reshape(error, (1, len(error)))
            g = data.T.dot(error)
            self.theta = self.theta + alpha * g

    def MBGD(self, alpha: float, batch_size: int):
        """
        MBGD（小批量梯度下降算法）
        :param alpha: 学习率
        :param batch_size: 小批量样本规模
        :return: None
        """
        self.Shuffle_Sequence()
        for start in np.arange(0, self.train_data.shape[0], batch_size):
            end = min(start + batch_size, self.train_data.shape[0])
            gradient_increasment = []
            mini_train_data = self.train_data[start:end]
            mini_train_result = self.train_result[start:end]
            for (data, result) in zip(mini_train_data, mini_train_result):
                predict = self.softmax(data.dot(self.theta))
                error = result - predict
                data = np.reshape(data, (1, len(data)))
                error = np.reshape(error, (1, len(error)))
                g = data.T.dot(error)
                gradient_increasment.append(g)
            avg_g = np.average(gradient_increasment, 0)
            self.theta = self.theta + alpha * avg_g

    def train_BGD(self, times: int, alpha: float):
        """
        利用BGD算法进行迭代
        :param times: 迭代次数
        :param alpha: 学习率
        :return: 每一次迭代的损失
        """
        Cost = []
        Cost.append(self.cost())
        for i in np.arange(times):
            self.BGD(alpha)
            Cost.append(self.cost())
        return Cost

    def train_SGD(self, times: int, alpha: float):
        """
        利用SGD算法进行迭代
        :param times: 迭代次数
        :param alpha: 学习率
        :return: 每一次迭代的损失
        """
        Cost = []
        Cost.append(self.cost())
        for i in np.arange(times):
            self.SGD(alpha)
            Cost.append(self.cost())
        return Cost

    def train_MBGD(self, times: int, alpha: float, batch_size: int):
        """
        利用BGD算法进行迭代
        :param batch_size: 批量最小规模
        :param times: 迭代次数
        :param alpha: 学习率
        :return: 每一次迭代的损失
        """
        Cost = []
        Cost.append(self.cost())
        for i in np.arange(times):
            self.MBGD(alpha, batch_size)
            Cost.append(self.cost())
        return Cost
