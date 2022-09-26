# -*- coding: UTF-8 -*-
'''
@Project ：pythonProject 
@File    ：lwlr.py
@Author  ：guo
@Date    ：2022/9/15 下午3:39 
'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def main():
    data = pd.read_csv('data', sep='\s+', names=['x1', 'x2', 'y'])
    data = np.array(data)
    input_data = data[:, :2]
    result = data[:, 2]
    # theta0 = regular_equation(input_data, result)  # 线性回归的参数
    # w = reW(input_data.dot(theta0), result)
    # theta = regular_equation(input_data, result, w)
    # pred_value = w.dot(input_data).dot(theta)
    p = lwlrTest(input_data, input_data, result, 0.01)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(input_data[:, 1], result, '.')
    # plt.plot(input_data[:, 1], input_data.dot(theta0))
    # plt.plot(input_data[:, 1], pred_value, '.')
    plt.plot(input_data[:, 1], p, '.')
    plt.show()


def Gaussian_Weight(data, k):
    """
    高斯权重生成函数
    @param data: 输入数据
    @param k: 带宽系数
    @return: 高斯权重
    """
    sum = data * data
    return np.exp(sum / (-2.0 * (k ** 2)))


# 计算某个样本点的预测值
def lwlr(testPoint, dataMat, labelMat, k=1.0):  # k为高斯核函数的参数
    xMat = np.mat(dataMat)
    yMat = np.mat(labelMat).T
    m = np.shape(xMat)[0]  # m=200
    weights = np.mat(np.eye(m))  # 初始化权重矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
        # 对testPoint样本计算出权重矩阵 200*200 且只有对角元素有值
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0:
        print('矩阵不可逆')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))  # 计算ws的估计值 2*1矩阵
    return testPoint * ws  # testPoint样本点的预测值yHat


# 计算整个样本的预测值
def lwlrTest(testArr, dataMat, labelMat, k=1.0):
    # testArr为待测试的矩阵,例子中即为dataMat
    m = np.shape(testArr)[0]  # 待测矩阵的行数 dataMat共200行
    yHat = np.zeros(m)  # 初始化预测值 1*m数值为0的矩阵
    for i in range(m):
        yHat[i] = lwlr(testArr[i], dataMat, labelMat, k)
    return yHat  # ndarray


if __name__ == '__main__':
    main()
