# -*- coding: UTF-8 -*-
"""
@Project ：pythonProject \n 
@File    ：BP.py \n
@Author  ：guo \n
@Date    ：2022/9/29 上午10:05 \n
"""
import numpy as np


class BPNN:
    def __init__(self, input_n, hidden_n, output_n, input_hidden_weight=None,
                 hidden_threshold=None, hidden_output_weights=None, output_threshold=None):
        """
        这是BP神经网络的构造函数 \n
        :param input_n: 输入层神经元个数
        :param hidden_n: 隐藏层神经元个数
        :param output_n: 输出层神经元个数
        :param input_hidden_weight: 输入层和隐含层之间的权重
        :param hidden_threshold: 隐含层的阈值
        :param hidden_output_weights: 隐含层与输出层之间的权重
        :param output_threshold: 输出层的阈值
        """
        self.train_label = None
        self.train_data = None
        self.input_n = input_n
        self.hidden_n = hidden_n
        self.output_n = output_n
        self.input_cells = np.zeros(self.input_n).reshape((1, self.input_n))
        self.hidden_cells = np.zeros(self.hidden_n).reshape((1, self.hidden_n))
        self.output_cells = np.zeros(self.output_n).reshape((1, self.output_n))
        self.input_hidden_weight = input_hidden_weight
        self.hidden_threshold = hidden_threshold
        self.hidden_output_weights = hidden_output_weights
        self.output_threshold = output_threshold

        if self.input_hidden_weight is None:
            self.input_hidden_weight = np.random.randn(input_n, hidden_n)
        if self.hidden_threshold is None:
            self.hidden_threshold = np.random.randn(1, hidden_n)
        if self.hidden_output_weights is None:
            self.hidden_output_weights = np.random.randn(hidden_n, output_n)
        if self.output_threshold is None:
            self.output_threshold = np.random.randn(1, output_n)

    def Init(self, Train_data, Train_label):
        """
        初始化训练数据集
        :param Train_data: 训练数据集
        :param Train_label: 训练标签集
        :return: None
        """
        self.train_data = Train_data
        self.train_label = Train_label

    def predict(self, input_data: np.ndarray):
        """
        BP神经网络的向前传播函数 \n
        :param input_data: 输入数据
        :return: 输出层预测结果
        """
        self.input_cells = input_data
        self.hidden_cells = input_data.dot(self.input_hidden_weight) + self.hidden_threshold
        self.hidden_cells = self.Sigmoid(self.hidden_cells)
        self.output_cells = self.hidden_cells.dot(self.hidden_output_weights) + self.output_threshold
        self.output_cells = self.Sigmoid(self.output_cells)
        return self.output_cells

    def Gradient(self, train_label, predict_label):
        """
        计算在反向传播过程中相关参数的梯度增量，损失函数为训练标签与预测标签之间的均方误差 \n
        :param train_label: 训练数据标签
        :param predict_label: BP神经网络的预测结果
        :return: 隐藏层与输出层之间权重的梯度增量，输出层阈值的梯度增量，输入层与隐藏层之间权重的梯度增量，隐藏层阈值的梯度增量
        """
        error = predict_label - train_label
        g = predict_label * (1 - predict_label) * error
        # 输出层阈值的梯度增量
        output_threshold_gradient_increasement = g
        # 隐藏层与输出层之间权重的梯度增量
        hidden_output_weights_gradient_increasement = self.hidden_cells.T.dot(g)

        e = self.hidden_cells * (1 - self.hidden_cells) * (g.dot(self.hidden_output_weights.T))
        # 隐藏层阈值的梯度增量
        hidden_threshold_gradient_increasement = e
        # 输入层与隐藏层之间权重的梯度增量
        input_hidden_weights_gradient_increasement = self.input_cells.T.dot(e)

        return hidden_output_weights_gradient_increasement, output_threshold_gradient_increasement, \
               input_hidden_weights_gradient_increasement, hidden_threshold_gradient_increasement

    def back_propagate(self, hidden_output_weights_gradient_increasement, output_threshold_gradient_increasement,
                       input_hidden_weights_gradient_increasement, hidden_threshold_gradient_increasement,
                       learning_rate):
        """
        利用误差反向传播算法对BP神经网络的模型参数进行迭代更新的函数 \n
        :param hidden_output_weights_gradient_increasement: 隐含层与输出层之间权重的梯度增量
        :param output_threshold_gradient_increasement: 输出层阈值的梯度增量
        :param input_hidden_weights_gradient_increasement: 输入层与隐含层之间权重的梯度增量
        :param hidden_threshold_gradient_increasement: 隐含层阈值的梯度增量
        :param learning_rate: 学习率
        :return: None
        """
        alpha = learning_rate
        self.hidden_output_weights = self.hidden_output_weights - alpha * hidden_output_weights_gradient_increasement
        self.output_threshold = self.output_threshold - alpha * output_threshold_gradient_increasement
        self.input_hidden_weight = self.input_hidden_weight - alpha * input_hidden_weights_gradient_increasement
        self.hidden_threshold = self.hidden_threshold - alpha * hidden_threshold_gradient_increasement

    def BGD(self, learning_rate):
        hidden_output_weights_gradient_increasements = []
        hidden_threshold_gradient_increasements = []
        input_hidden_weight_gradient_increasements = []
        output_threshold_gradient_increasements = []
        pred_label = []
        for (t_data, t_label) in zip(self.train_data, self.train_label):
            t_data = np.reshape(t_data, (1, len(t_data)))
            t_label = np.reshape(t_label, (1, len(t_label)))
            pred = self.predict(t_data)
            pred_label.append(pred)
            hidden_output_weights_gradient_increasement, \
            output_threshold_gradient_increasement, \
            input_hidden_weight_gradient_increasement, \
            hidden_threshold_gradient_increasement = self.Gradient(t_label, pred)

            hidden_output_weights_gradient_increasements.append(hidden_output_weights_gradient_increasement)
            output_threshold_gradient_increasements.append(output_threshold_gradient_increasement)
            input_hidden_weight_gradient_increasements.append(input_hidden_weight_gradient_increasement)
            hidden_threshold_gradient_increasements.append(hidden_threshold_gradient_increasement)

        hidden_threshold_gradient_increasements_avg = np.average(hidden_threshold_gradient_increasements, 0)
        output_threshold_gradient_increasements_avg = np.average(output_threshold_gradient_increasements, 0)
        input_hidden_weight_gradient_increasements_avg = np.average(input_hidden_weight_gradient_increasements, 0)
        hidden_output_weights_gradient_increasements_avg = np.average(hidden_output_weights_gradient_increasements, 0)

        self.back_propagate(hidden_output_weights_gradient_increasements_avg,
                            output_threshold_gradient_increasements_avg,
                            input_hidden_weight_gradient_increasements_avg,
                            hidden_threshold_gradient_increasements_avg,
                            learning_rate)
        return self.cost(self.train_label, pred_label)

    def train_BGD(self, train_data, train_label, times, learning_rate):
        self.Init(train_data, train_label)
        Mse = []
        for i in np.arange(times):
            m = self.BGD(learning_rate)
            Mse.append(m)
        return np.array(Mse)

    def cost(self, Train_label, Predict_label):
        """
        训练数据集的平均训练损失函数
        :param Train_label: 训练标签集
        :param Predict_label: 预测结果集
        :return: 训练数据集的均方误差
        """
        return np.average(((Train_label - Predict_label) ** 2) / 2)

    def Sigmoid(self, x):
        """
        激活函数，Sigmoid函数，值域0~1 \n
        :param x:自变量
        :return:函数值
        """
        return 1 / (1 + np.exp(-x))
