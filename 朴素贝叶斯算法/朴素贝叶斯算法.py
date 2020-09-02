#Author:liulu
#Date:2020.09.02
#参考代码：https://github.com/Dod-o/Statistical-Learning-Method_Code
'''
数据集：Mnist
训练集数量：60000
测试集数量：10000
------------------------------
运行结果：
    正确率：84.35%
    运行时长：102s
'''
import numpy as np
import time

def Dataload(file):
    # 加载离线数据集
    data = []
    label = []
    print("start to load file")
    f = open(file)
    # 遍历文件中的每一行
    for line in f.readlines():
        # 获取当前行，并按“，”切割成字段放入列表中
        # strip：去掉每行字符串首尾指定的字符（默认空格或换行符）
        # split：按照指定的字符将字符串切割成每个字段，返回列表形式
        curline = line.strip().split(',')
        # 为了方便计算，将数据进行了二值化处理，大于128的转换成1，小于的转换成0。即每一个特征的取值只有两种。
        data.append([1 if int(i) > 128 else 0 for i in curline[1:]])
        #因为读出来的数据为字符类型，一定要将数据变为整数类型，否则后续计算无法完成。
        label.append(int(curline[0]))
    print("file loading completed!")
    return data,label


class NavieBayes:
    def __init__(self):
        #初始化
        #数据集中手写图片为28*28，转换为向量是784维。CSV文件是已经转换后的数据。
        self.feature_num = 784
        # 设置类别数目，0-9共十个类别
        self.class_num = 10

    def getPriorProbability(self,traindata,trainlabel):
        #计算先验概率P(Y)，P(X|Y)
        # 初始化先验概率分布存放数组，后续计算得到的P(Y = 0)放在Py[0]中，以此类推
        Py = np.zeros((self.class_num,1))
        for i in range(self.class_num):
            # 这里用的是贝叶斯估计，对应课本式（4.11）
            Py[i] = (np.sum(trainlabel == i)+1)/(len(trainlabel)+10)
        # 程序运行中很可能会向下溢出无法比较，因为值太小了。所以人为把值进行log处理。log在定义域内是一个递增函数，也就是说log（x）中，
        # x越大，log也就越大，单调性和原数据保持一致。所以加上log对结果没有影响。此外连乘项通过log以后，可以变成各项累加，简化了计算。
        Py = np.log(Py)

        Ix_y = np.zeros((self.class_num,self.feature_num,2))
        Px_y = np.zeros((self.class_num,self.feature_num,2))
        for i in range(len(traindata)):
            label = trainlabel[i]
            x = traindata[i]
            for j in range(self.feature_num):
                #获取y=label，第j个特征以及每个特征取值对应的个数
                Ix_y[label][j][x[j]] += 1

        for i in range(self.class_num):
            for j in range(self.feature_num):
                # 获取y=label，第j个特征值为0的个数
                Ix_y0 = Ix_y[i][j][0]
                # 获取y=label，第j个特征值为1的个数
                Ix_y1 = Ix_y[i][j][1]
                # 分别计算对于y= label，x第j个特征值为0和1的条件概率分布
                #这里用的是贝叶斯估计，对应课本式（4.10）
                Px_y[i][j][0] = np.log((Ix_y0 + 1)/(Ix_y0 + Ix_y1 + 2))
                Px_y[i][j][1] = np.log((Ix_y1 + 1) / (Ix_y0 + Ix_y1 + 2))

        return Py,Px_y

    def getPostProbability(self,Py,Px_y,x):
        #求所需要的后验概率P(Y|X)

        # 建立存放所有标记的估计概率数组
        P = [0]*self.class_num
        for i in range(self.class_num):
            sum = 0
            for j in range(self.feature_num):
                # 在训练过程中对概率进行了log处理，所以这里原先应当是连乘所有概率，最后比较哪个概率最大
                # 但是当使用log处理时，连乘变成了累加，所以使用sum
                sum += Px_y[i][j][x[j]]
            #对应于课本（4.7）
            P[i] = sum + Py[i]
        # P.index(max(P))：找到该概率最大值对应的索引（索引值和标签值相等）
        return P.index(max(P))

    def test(self,Py,Px_y,testdata,testlabel):
        #利用测试集数据进行测试，给出准确率
        error_count = 0
        for i in range(len(testdata)):
            x = testdata[i]
            label_Test = testlabel[i]
            label_Predict = self.getPostProbability(Py,Px_y,x)
            if label_Test != label_Predict:
                error_count +=1
        accuracy = 1 - error_count/len(testdata)
        print("Accuracy is {}".format(accuracy))


if __name__ == "__main__":
    start = time.time()
    traindata, trainlabel = Dataload('D:/jupyter/Statistical-Learning-Method_Code-master/Statistical-Learning-Method_Code-master/Mnist/mnist_train/mnist_train.csv')
    testdata, testlabel = Dataload('D:/jupyter/Statistical-Learning-Method_Code-master/Statistical-Learning-Method_Code-master/Mnist/mnist_test/mnist_test.csv')
    clf = NavieBayes()
    Py,Px_y = clf.getPriorProbability(traindata,trainlabel)
    clf.test(Py,Px_y,testdata,testlabel)
    end = time.time()
    print("time span is {}".format(end - start))

