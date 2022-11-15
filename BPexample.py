# -*- coding: UTF-8 -*-
"""
@Project ：pythonProject \n 
@File    ：BPexample.py \n
@Author  ：guo \n
@Date    ：2022/10/1 下午1:22 \n
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from BP import BPNN
import time

def main():
    start_time = time.time()
    data, result = read()
    data[:, 1:] = MinMaxScaler().fit_transform(data[:, 1:])  # 数据归一化
    result = change_onehot(result, 4)
    train_data, test_data, train_result, test_result = train_test_split(data, result, test_size=0.25, shuffle=10)
    input_n = train_data.shape[1]
    output_n = train_result.shape[1]
    hidden_n = int(round((input_n * output_n) * 2.0 / 3))
    BGD_BPNN = BPNN(input_n,hidden_n,output_n)
    cost = BGD_BPNN.train_BGD(train_data,train_result,1000,0.001)
    """
    如果cost先减小后变大可能是过拟合现象
    """
    plt.xlabel('times')
    plt.ylabel('cost')
    plt.plot(range(len(cost)),cost)
    end_time = time.time()
    print(end_time - start_time)
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


if __name__ == '__main__':
    main()
