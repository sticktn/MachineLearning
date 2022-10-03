# -*- coding: UTF-8 -*-
"""
@Project ：pythonProject \n 
@File    ：Kmeans.py \n
@Author  ：guo \n
@Date    ：2022/10/1 下午9:16 \n
"""
import numpy as np


class Kmeans:
    def __init__(self, Data, k):
        """
        Kmeans算法的构造函数 \n
        :param Data: 聚类数据集
        :param k: 聚类个数
        """
        self.data = Data
        row = Data.shape[0]
        self.label = np.array([0] * row)
        self.centroids = Data[np.random.choice(Data.shape[0], k, True)]
        self.K = k

    def EcludDistance(self, dataA, dataB):
        """
        计算两个向量之间的欧氏距离 \n
        :param dataA: A向量
        :param dataB: B向量
        :return: A,B向量之间的欧氏距离
        """
        return np.sqrt(np.sum((dataA - dataB) ** 2))

    def cluster(self):
        """
        聚类函数
        :return: None
        """
        newdist = 0
        lastdist = 1
        while np.abs(newdist - lastdist) > 10E-6:
            lastdist = newdist
            for i, d in enumerate(self.data):
                distance = []
                for centroid in self.centroids:
                    distance.append(self.EcludDistance(d, centroid))
                self.label[i] = np.argmin(distance)

            for i in np.arange(self.K):
                cluster_data = self.data[self.label == i]
                size = len(cluster_data)
                if size != 0:
                    self.centroids[i] = np.sum(cluster_data, 0) / size

            newdist = 0

            for data, label in zip(self.data, self.label):
                newdist += self.EcludDistance(data, self.centroids[label])
        return self.label, self.centroids, newdist
