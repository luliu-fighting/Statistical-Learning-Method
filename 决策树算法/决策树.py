#Author:liulu
#Date:2020.09.04
#参考代码：https://github.com/fengdu78/lihang-code

'''
此程序利用ID3算法生成了决策树，并未对决策树进行剪枝。
代码实现了李航《统计学习方法》的例5.1
因为数据量太少，所以并未计算准确率，只是对输入进行了预测。
'''


import numpy as np
import pandas as pd
import time
from collections import Counter
from math import log


# 定义节点类 二叉树
class Node:
    #初始化
    def __init__(self, root=True, label=None, feature_name=None, feature=None):
        self.root = root   #True表示此节点为叶子节点，无分支了
        self.label = label  #节点的类标记
        self.feature_name = feature_name  #特征的名称
        self.feature = feature   #特征的取值
        self.tree = {}
        self.result = {
            'label:':self.label,
            'feature:':self.feature,
            'tree:':self.tree
        }

    #自定义显示
    def __repr__(self):
        return '{}'.format(self.result)

    #增加分支节点
    def add_node(self,val,node):
        self.tree[val] = node

    #递归的预测实例的类
    def predict(self,features):
        if self.root == True:
            return self.label
        return self.tree[features[self.feature]].predict(features)


#定义决策树类
class DecisionTree:
    #初始化
    def __init__(self,threshold=0.1):
        self.threshold = threshold
        self._tree = {}

    #计算数据集的经验熵
    def calc_HD(self,dataset):
        dataset = np.array(dataset)
        data_length = len(dataset)
        label = dataset[:,-1]  #要求数据集的最后一列为标记列
        label_count = Counter(label)  #得到一个字典，key为类别，value为对应的类的数量
        HD = -sum([(p/data_length)*log(p/data_length,2) for p in label_count.values()]) #课本式（5.7）
        return HD

    #计算特征A对数据集的经验条件熵
    def calc_HDA(self,dataset,A=0):
        data_length = len(dataset)
        feature_sets = {}
        #构建以特征A的取值划分的各个数据子集
        for i in range(data_length):
            feature = dataset[i][A]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(dataset[i])
        HDA = 0
        #课本式（5.8）
        for D in feature_sets.values():
            HDA += (len(D)/data_length)*self.calc_HD(D)
        return HDA

    #找出信息增益最大的特征
    def calcBestFeature(self,dataset):
        feature_count = len(dataset[0]) - 1    #特征的个数
        GDA = [0]*feature_count
        for i in range(feature_count):
            #计算信息增益，课本式（5.9）
            GDA[i] = self.calc_HD(dataset) - self.calc_HDA(dataset,A=i)
        max_GDA = max(GDA)     #最大的信息增益
        best_feature = GDA.index(max_GDA)    #最大的信息增益对应的特征索引
        return best_feature,max_GDA

    #利用ID3算法递归生成决策树
    def createTree(self,train_data):
        label_train = train_data.iloc[:,-1]     #要求输入的数据为pd.DataFrame类型
        features = train_data.columns[:-1]      #并要求最后一列为类别标记列

        # 1,若D中实例属于同一类Ck，则T为单节点树，并将类Ck作为结点的类标记，返回T
        if len(label_train.value_counts()) == 1:
            return Node(root=True, label=label_train.iloc[0])

        # 2, 若A为空，则T为单节点树，将D中实例树最大的类Ck作为该节点的类标记，返回T
        if len(features) == 0:
            return Node(root=True, label=label_train.value_counts().index[0])

        # 3,计算最大信息增益 ，Ag为信息增益最大的特征
        best_feature, max_GDA = self.calcBestFeature(np.array(train_data))
        best_feature_name = features[best_feature]

        # 4,Ag的信息增益小于阈值eta,则置T为单节点树，并将D中是实例数最大的类Ck作为该节点的类标记，返回T
        if max_GDA < self.threshold:
            return Node(root=True, label=label_train.value_counts().index[0])

        # 5,根据Ag的取值构建数据子集
        node_tree = Node(root=False, feature_name=best_feature_name, feature=best_feature)
        feature_list = train_data[best_feature_name].value_counts().index
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[best_feature_name] ==
                                          f].drop([best_feature_name], axis=1)
            # 6, 递归生成决策树
            sub_tree = self.createTree(sub_train_df)
            node_tree.add_node(f, sub_tree)
        return node_tree

    #训练，生成决策树
    def fit(self, train_data):
        print("start to fit")
        self._tree = self.createTree(train_data)
        print("DecisionTree completed!")
        return self._tree

    #预测实例的类
    def predict(self, x_data):
        return self._tree.predict(x_data)

    #计算准确率
    def accrucy(self,test_data):
        error_count = 0
        x_data = np.array(test_data.iloc[:,1:])
        label_test = np.array(test_data.iloc[:,0])
        for i in range(len(test_data)):
            if label_test[i] != self.predict(x_data[i]):
                error_count += 1
        accrucy = 1 - error_count/len(test_data)
        print("Accrucy is {}".format(accrucy))

#主程序
if __name__=='__main__':
    start = time.time()
    #例5.1的数据
    datasets = [['青年', '否', '否', '一般', '否'],
                ['青年', '否', '否', '好', '否'],
                ['青年', '是', '否', '好', '是'],
                ['青年', '是', '是', '一般', '是'],
                ['青年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '好', '否'],
                ['中年', '是', '是', '好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '好', '是'],
                ['老年', '是', '否', '好', '是'],
                ['老年', '是', '否', '非常好', '是'],
                ['老年', '否', '否', '一般', '否'],
                ]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']
    train_data = pd.DataFrame(datasets, columns=labels)
    clf = DecisionTree()
    tree = clf.fit(train_data)
    print(tree)
    end = time.time()
    print("time span : {}".format(end - start))
    print(clf.predict(['老年', '否', '否', '一般']))