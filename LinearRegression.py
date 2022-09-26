"""
正则方程
"""
import numpy.linalg
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def main():
    data = pd.read_csv('data', sep='\s+', names=['x1', 'x2', 'x3'])
    data = np.array(data)
    input_data = data[:, :2]
    result = data[:, 2]

    l = input_data.shape[0]
    pred = []
    for j in range(l):
        pred.append(lwlrPoint(input_data[j, :], input_data, result, 0.01))
    pred = np.array(pred)

    # plt.figure(1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(input_data[:, 1], result, '.')
    plt.plot(input_data[:, 1], pred, '.')
    plt.show()


def lwlrPoint(point, input_data, result, k=0.1):
    """
    对point加权
    :param k: 高斯权重函数参数
    :param point: 观测点
    :param input_data: X
    :param result: Y
    :return: point对应的预测值
    """
    length = input_data.shape[0]
    m = np.zeros((length, length))
    for i in range(length):
        diff = point - input_data[i, :]
        m[i, i] = np.exp(diff.dot(diff.T) / (-2.0 * k ** 2))
    XTWX = input_data.T.dot(m).dot(input_data)
    inv = np.linalg.inv(XTWX)

    XTWY = input_data.T.dot(m).dot(result)
    theta = inv.dot(XTWY)
    return point.dot(theta)


if __name__ == '__main__':
    main()
