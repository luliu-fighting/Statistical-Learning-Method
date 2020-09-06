#Author:liulu
#Date:2020.09.06
#参考代码：https://github.com/Dod-o/Statistical-Learning-Method_Code
#           https://github.com/fengdu78/lihang-code
'''
数据集：iris
数据集简介：iris数据集大小为150*5，前四列为特征，最后一列为标签；一个有三种标签，每个标签对应50个样本；
在此程序中，为了画图方便，只用了前两个特征sepal length 和 sepal width, 并且只用了前100个样本，也就是只
用了两类标签，实现二分类。

此程序对权值的更新采用了牛顿法
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载数据
def loadData():
    print("start to load data")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    # 由于原来的列标签后面都带有单位cm，所以这里重新定义列标签
    df.columns = ['sepal lengrh', 'sepal width', 'petal length', 'petal width', 'label']
    # pd.DataFrame类型转换为numpy数组，并切片出前两列特征和和最后一列类标签
    data = np.array(df.iloc[:100, [0, 1, -1]])
    print("loading completed")
    return data[:,:-1],data[:,-1]


class LogisticRegression:
    #初始化权值，学习率和最大迭代次数
    def __init__(self,learning_rate = 0.01, max_iter = 200):
        self.w = np.zeros((3,1),dtype=np.float)  #如果要用iris数据集的全部特征，将这里的3改为5即可
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    #定义sigmoid函数
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    #训练,牛顿法
    def train(self, train_data, train_label):
        print("start to train")
        #因为输入的train_label是个1维数组，无法进行转置，所以将其扩充为2维
        train_label = np.array([train_label]).T
        for iter in range(self.max_iter):
            #一行一行的求解
            for i in range(len(train_data)):
                # 因为取出的x是个1维数组，无法进行转置，所以将其扩充为2维
                x = np.array([train_data[i]])
                y = train_label[i]
                wx = np.dot(x, self.w)
                #计算一阶导数
                gradient = np.dot(x.T, (y - self.sigmoid(wx)))
                #计算Hessian矩阵
                Hessian = -np.dot(x.T, self.sigmoid(wx)).dot((1 - self.sigmoid(wx))).dot(x)
                #权值更新
                self.w -= self.learning_rate * np.linalg.pinv(Hessian).dot(gradient)
        print("training completed")
        print("w is {}".format(self.w))

    #预测
    def predict(self, x):
        wx = np.dot(x, self.w)
        result = self.sigmoid(wx)
        if result > 0.5:
            return 1
        else:
            return 0

    #测试，给出正确率
    def test(self, test_data, test_label):
        print("start to test")
        error_count = 0
        for i in range(len(test_data)):
            x = test_data[i]
            y = test_label[i]
            if y != self.predict(x):
                error_count += 1
        acc = 1 - error_count/len(test_data)
        print("testing completed")
        print("Accrucy is {:.3%}".format(acc))

if __name__ ==  "__main__":
    #加载数据
    X, Y = loadData()
    #因为将b合并到了w中，所以相应的要扩充输入向量，w[-1]为b
    X_data = np.hstack([X, np.ones((len(X),1))])
    #划分训练集和测试集
    train_data, test_data, train_label, test_label = train_test_split(X_data, Y, test_size=0.3)
    clf = LogisticRegression()
    clf.train(train_data, train_label)
    clf.test(test_data, test_label)

    #画图
    plt.scatter(X[:50, 0], X[:50, 1], label='label0')
    plt.scatter(X[50:, 0], X[50:, 1], label='label1')
    plt.legend()
    x_points = x_ponits = np.arange(4, 8)
    y_points = -(clf.w[0]*x_ponits + clf.w[2])/clf.w[1]
    plt.plot(x_ponits, y_points)
    plt.show()