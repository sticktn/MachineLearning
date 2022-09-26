# -*- coding: UTF-8 -*-
"""
@Project ：PycharmProjects
@File    ：logistic_example.py
@Author  ：guo
@Date    ：2022/9/22 下午8:13
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import Logistic_Regression
import numpy as np
from matplotlib import pyplot as plt


def main():
    breast_cancer = load_breast_cancer()
    data, result = breast_cancer.data, breast_cancer.target
    # 数据处理
    data = MinMaxScaler().fit_transform(data)
    data = revise_data(data)
    train_data, test_data, train_result, test_result = train_test_split(data, result, test_size=0.25)
    theta = np.random.randn(train_data.shape[1])

    BGD_logistic = Logistic_Regression.Logistic_Regression(train_data, train_result, theta)
    SGD_logistic = Logistic_Regression.Logistic_Regression(train_data, train_result, theta)
    MBGD_logistic = Logistic_Regression.Logistic_Regression(train_data, train_result, theta)

    times = 10000
    alpha = 0.1
    batch_size = 20

    BGD_cost, BGD_ac = BGD_logistic.train_BGD(times, alpha, test_data, test_result)
    SGD_cost, SGD_ac = SGD_logistic.train_SGD(times, alpha, test_data, test_result)
    MBGD_cost, MBGD_ac = MBGD_logistic.train_MBGD(times, alpha, test_data, test_result, batch_size)

    length = range(len(BGD_cost))
    plt.xlabel("times")
    plt.ylabel('cost')
    plt.plot(length, BGD_cost, 'red', label='BGD_cost')
    plt.plot(length, SGD_cost, 'pink', label='SGD_cost')
    plt.plot(length, MBGD_cost, 'blue', label='MBGD_cost')
    plt.legend()

    plt.show()
    # print(max(AC))
    # print(AC.index(max(AC)))
    # print(logistic.ac(test_data, test_result))


def revise_data(data):
    """
    给数据集的第一列添加一列1，为后续的回归系数的常数项做准备。
    @param data: 特征数据集
    @return: 加维后的数据集
    """
    data1 = np.ones((data.shape[0], data.shape[1] + 1))
    data1[:, 1:] = data
    return data1


if __name__ == '__main__':
    main()
