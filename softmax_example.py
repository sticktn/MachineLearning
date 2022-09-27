# -*- coding: UTF-8 -*-
"""
@Project ：PycharmProjects \n 
@File    ：softmax_example.py \n
@Author  ：guo \n
@Date    ：2022/9/25 下午12:02 \n
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Softmax_Regression import Softmax_Regression
from Softmax_Regression import change_onehot
from matplotlib import pyplot as plt


def main():
    data, result = read()
    data[:, 1:] = MinMaxScaler().fit_transform(data[:, 1:])  # 数据归一化
    result = change_onehot(result, 4)
    train_data, test_data, train_result, test_result = train_test_split(data, result, test_size=0.25, shuffle=10)
    row, col = np.shape(train_data)[1], 4
    theta = np.random.randn(row, col)
    BGD_softmax = Softmax_Regression(train_data, train_result, theta)
    SGD_softmax = Softmax_Regression(train_data, train_result, theta)
    MBGD_softmax = Softmax_Regression(train_data, train_result, theta)
    cost_BGD = BGD_softmax.train_BGD(1000, 0.1)
    cost_SGD = SGD_softmax.train_SGD(1000, 0.1)
    cost_MBGD = MBGD_softmax.train_MBGD(1000, 0.1, 20)

    length = len(cost_MBGD)
    plt.xlabel('times')
    plt.ylabel('cost')
    plt.plot(range(length), cost_BGD, c='blue')
    plt.plot(range(length), cost_SGD, c='red')
    plt.plot(range(length), cost_MBGD, c='black')
    plt.gray()
    plt.show()


def read() -> tuple[np.ndarray, np.ndarray]:
    """
    读取数据集
    :return:数据集和数据结果
    """
    df = pd.read_csv('voice_data.txt', sep='\s+', header=None)
    ndarray = np.array(df)
    result = ndarray[:, 0].copy()
    ndarray[:, 0] = 1
    data = ndarray
    return data, result


if __name__ == '__main__':
    main()
