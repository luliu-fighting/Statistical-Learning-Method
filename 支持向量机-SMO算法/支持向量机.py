#Author:liulu
#Date: 2020.9.20

#本程序实现了用序列最小最优化算法求解支持向量机
#核函数为高斯核

'''
参考代码：
    https://github.com/Dod-o/Statistical-Learning-Method_Code
    https://github.com/fengdu78/lihang-code

数据集：
    Mnist
    训练集数量：60000(实际使用：2000)
    测试集数量：10000（实际使用：600)

运行结果：
    正确率：95.33%
    运行时长：79.094s
'''


import numpy as np
import pandas as pd
import time

#加载数据集
def loadData():
    print("start to load data")
    df_train = pd.read_csv("D:/jupyter/Statistical-Learning-Method_Code-master/Statistical-Learning-Method_Code-master/Mnist/mnist_train/mnist_train.csv")
    df_test = pd.read_csv("D:/jupyter/Statistical-Learning-Method_Code-master/Statistical-Learning-Method_Code-master/Mnist/mnist_test/mnist_test.csv")

    data_train = np.array(df_train.iloc[:2000, :])
    data_test = np.array(df_test.iloc[:600, :])

    # 将特征的取值从0-255调整为0-1
    # 数据集label共有10类（0-9），为了实现二类分类，将0作为一类，非0的其他类归为一类
    trainfeature = data_train[:, 1:] / 255
    trainlabel = np.array([1 if i == 0 else -1 for i in data_train[:, 0]])
    testfeature = data_test[:, 1:] / 255
    testlabel = np.array([1 if i == 0 else -1 for i in data_test[:, 0]])

    print("completed!")
    return trainfeature, trainlabel, testfeature, testlabel

#定义SVM类
class SVM:
    #初始化
    def __init__(self, trainfeature, trainlabel, C=0.8):

        #惩罚参数C的设置是多次运行得出了最优值，当C=10时，正确率在91%左右，C=0.8时，
        #正确率在95%左右，C=0.7时，正确率为70%左右

        self.trainfeature = trainfeature
        self.trainlabel = trainlabel
        self.m, self.n = self.trainfeature.shape
        # 惩罚参数，C越大，对误分类的惩罚越大
        self.C = C
        # 核函数的计算，使用的是高斯核函数，初始化时将所有的核函数的值提前算好
        self.k = self.calckernel()
        self.b = 0
        # 初始化所有α为0
        self.alpha = np.zeros(self.m)
        # SMO运算过程中的Ei
        self.E = [self.calcEi(i) for i in range(self.m)]

    #计算核函数
    def calckernel(self, sigma=10):
        #高斯核函数
        # 初始化高斯核结果矩阵 大小 = 训练集长度m * 训练集长度m
        k = np.zeros((self.m, self.m))
        for i in range(self.m):
            x = self.trainfeature[i]
            # 因为得到的高斯核矩阵应是对称的，所有小循环的j从i开始遍历
            for j in range(i, self.m):
                z = self.trainfeature[j]
                #式（7.90）
                result = np.exp(-np.linalg.norm(x-z)**2 / (2*sigma**2))
                #同时给对称的位置赋值
                k[i, j] = result
                k[j, i] = result

        return k

    #计算g(xi)
    def calcGxi(self, i):
        #式（7.104）
        gxi = self.b
        for j in range(self.m):
            #在“7.2.3支持向量”开头第一句话有说到“对应于α > 0的样本点
            # (xi, yi)的实例xi称为支持向量”。也就是说只有支持向量的α是大于0的，在求和式内的
            # 对应的αi*yi*K(xi, xj)不为0，非支持向量的αi*yi*K(xi, xj)必为0，也就不需要参与
            # 到计算中。也就是说，在g(xi)内部求和式的运算中，只需要计算α>0的部分，其余部分可
            # 忽略。因为支持向量的数量是比较少的，这样可以再很大程度上节约时间
            # 从另一角度看，抛掉支持向量的概念，如果α为0，αi*yi*K(xi, xj)本身也必为0，从数学
            # 角度上将也可以扔掉不算
            if self.alpha[j] == 0:
                continue
            else:
                gxi += self.alpha[j] * self.trainlabel[j] * self.k[j, i]

        return gxi

    #计算序列最小最优化算法的（SMO）中的E(i)
    def calcEi(self, i):
        #式（7.105）
        return self.calcGxi(i) - self.trainlabel[i]

    #判断样本是否满足KKT条件，用于SMO中第一个变量的选择
    def isSatisfyKKT(self, i):
        y_g = self.calcGxi(i) * self.trainlabel[i]
        #式（7.111）-式（7.113）
        if (self.alpha[i] == 0) and (y_g >= 1):
            return True
        elif (0 < self.alpha[i] < self.C) and (y_g == 1):
            return True
        elif (self.alpha[i] == self.C) and (y_g <= 1):
            return True
        else:
            return False

    #SMO中两个变量的选择
    def getAlpha(self):
        #根据P147的中间那段话
        #先找出满足条件0<alpha[i]<C的样本点，检验是否满足KKT条件
        index_list = np.array([i for i in range(self.m) if 0 < self.alpha[i] < self.C])
        #如果index_list中的样本均满足KKT条件，则遍历剩下的样本点，检验是否满足KKT条件
        index_other = np.array([i for i in range(self.m) if i not in index_list])
        index = np.hstack([index_list, index_other])
        for i in index:
            #在Debug过程中发现hstack指令会将int型变为float型，所以要进行调整
            i = int(i)
            if self.isSatisfyKKT(i):
                continue
            #选择样本中违反KKT条件最严重的样本点作为第一个变量
            E1 = self.E[i]
            #根据P147倒数第二段选择第二个变量
            if E1 >= 0:
                j = self.E.index(min(self.E))
            else:
                j = self.E.index(max(self.E))
            return i, j

    #训练
    def train(self, max_iter=1000):
        print('start to train')
        iter = 0
        #设置参数来控制循环，当参数不再更新时终止循环
        parameterChanged = 1
        while (iter < max_iter) and (parameterChanged > 0):
            iter += 1
            parameterChanged = 0
            #变量的选择
            i1, i2 = self.getAlpha()
            #P143，对alpha的值进行限制，课本上说的是剪辑
            if self.trainlabel[i1] == self.trainlabel[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]

            eta = self.k[i1, i1] + self.k[i2, i2] - 2*self.k[i1, i2]
            # 式（7.106）
            #先赋值再剪辑
            #当前值self.alpha[i2]即为公式中的alpha2_old
            alpha2_new = self.alpha[i2] + self.trainlabel[i2]*(E1 - E2)/eta
            if alpha2_new < L:
                alpha2_new = L
            elif alpha2_new > H:
                alpha2_new = H
            #式（7.109）
            alpha1_new = self.alpha[i1] + self.trainlabel[i1]*self.trainlabel[i2]*(self.alpha[i2] - alpha2_new)
            #式（7.115）
            b1_new = -E1 - self.trainlabel[i1] * self.k[1, 1] * (alpha1_new - self.alpha[i1]) - \
                     self.trainlabel[i2] * self.k[2, 1] * (alpha2_new - self.alpha[i2]) + self.b
            #式（7.116）
            b2_new = -E2 - self.trainlabel[i1] * self.k[1, 2] * (alpha1_new - self.alpha[i1]) - \
                     self.trainlabel[i2] * self.k[2, 2] * (alpha2_new - self.alpha[i2]) + self.b

            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # 选择中点，P148倒数第二段
                b_new = (b1_new + b2_new) / 2

            #判断参数是否发生了变化
            if np.abs(alpha2_new - self.alpha[i2]) >= 0.0001:
                parameterChanged += 1

            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new
            #重新计算E(i)
            self.E[i1] = self.calcEi(i1)
            self.E[i2] = self.calcEi(i2)

        print('training completed!')

    #两个样本单独计算核函数，在预测单个样本标签的时候用的
    def calcSinglKernel(self, x1, x2, sigma=10):
        return np.exp(-np.linalg.norm(x1-x2)**2 / (2*sigma**2))

    #预测
    def predict(self, x):
        result = self.b

        for i in range(self.m):
            result += self.alpha[i] * self.trainlabel[i] * self.calcSinglKernel(x, self.trainfeature[i])

        return np.sign(result)

    #用测试集测试，给出正确率
    def test(self, testfeature, testlabel):
        print('start to test')
        right_count = 0
        for i in range(len(testfeature)):
            result = self.predict(testfeature[i])
            if result == testlabel[i]:
                right_count += 1
        print("testing compelted!")
        print("accrucy is {:.2%}".format(right_count / len(testfeature)))


if __name__ == "__main__":
    start = time.time()
    #加载训练集和测试集数据
    trainfeature, trainlabel, testfeature, testlabel = loadData()
    #类实例化
    svm = SVM(trainfeature, trainlabel, C=0.8)
    #训练
    svm.train()
    #测试
    svm.test(testfeature, testlabel)
    #给出程序运行时间
    print("time span:{:.3f} seconds".format(time.time() - start))






