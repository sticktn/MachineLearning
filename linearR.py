# -*- coding: UTF-8 -*-
"""
@Project ：PycharmProjects \n 
@File    ：linearR.py \n
@Author  ：guo \n
@Date    ：2023/3/4 上午11:17 \n
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def BGD(learning_rate, theta, input, label):
    grad = []
    for i, j in zip(input, label):
        pred = i.dot(theta.T)
        grad.append((j - pred) * i)
    grad = np.average(grad, axis=0).reshape(1, 2)
    theta = theta + learning_rate * grad
    return theta


def train_BGD(times, learning_rate, theta, input, label):
    for _ in range(times):
        theta = BGD(learning_rate, theta, input, label)
    return theta


data = pd.read_csv('data', sep='\s+', names=['x1', 'x2', 'x3'])
data = np.array(data)
input_data = data[:, :2]
result = data[:, 2]

theta = np.random.rand(1,input_data.shape[1])

theta = train_BGD(10000,0.001,theta,input_data,result)

plt.plot(input_data[:,1],input_data.dot(theta.T))
plt.scatter(input_data[:,1],result,c='y')
plt.show()
