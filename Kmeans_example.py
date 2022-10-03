# -*- coding: UTF-8 -*-
"""
@Project ：pythonProject \n 
@File    ：Kmeans_example.py \n
@Author  ：guo \n
@Date    ：2022/10/3 上午10:32 \n
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

from Kmeans import Kmeans


def main():
    data, label = load_iris().data, load_iris().target
    data = MinMaxScaler().fit_transform(data)

    col = np.shape(data)[1]
    Col = ["sepal length", "sepal width ", "petal length", "petal width"]
    for i in range(0, col - 1):
        for j in range(i + 1, col):
            plt.scatter(data[:, i], data[:, j])
            plt.grid(True)
            plt.xlabel(Col[i])
            plt.ylabel(Col[j])
            plt.show()
            # plt.tight_layout()

    K = np.arange(1, 11)
    Dist = []
    Labels = []
    Centroids = []
    for k in K:
        kmeans = Kmeans(data,k)
        Label, centroids, dist = kmeans.cluster()
        Dist.append(dist)
        Labels.append(Label)
        Centroids.append(centroids)
        print("k=%d下的质心：" % (k))
        print(Centroids)
    plt.plot(K, Dist)
    plt.grid(True)
    plt.xlabel("k")
    plt.ylabel("聚类后的距离")
    plt.show()

    # 对k=3的聚类结果进行可视化
    # 遍历所有数据及其聚类标签
    colors = ['r','g',"b"]
    markers = ['o','*','x']
    for i in np.arange(0,col-1):
        for j in np.arange(i + 1, col):
            # 画每簇数据
            for (index, (c, m)) in enumerate(zip(colors, markers)):
                d = data[Labels[2] == index]
                plt.scatter(d[:, i], d[:, j], c=c, marker=m, alpha=0.5)
            # 画聚类质心
            for centroid in Centroids[2]:
                plt.scatter(centroid[i], centroid[j], c="k", marker="+",s=100)
            # 画面属性设置
            plt.xlabel(Col[i])
            plt.ylabel(Col[j])
            plt.grid(True)
            plt.show()



if __name__ == '__main__':
    main()
