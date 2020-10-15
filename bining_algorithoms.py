#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

data = np.random.randn(100)
#等频分箱
data_bins = pd.qcut(data, q=5)
print(data_bins.value_counts())
#等距分箱
data_bins = pd.cut(data,bins=5)
print(data_bins.value_counts())


# In[2]:


#决策树分箱
from sklearn.tree import DecisionTreeClassifier
def optimal_binning_boundary(x, y):
    '''
        利用决策树获得最优分箱的边界值列表,利用决策树生成的内部划分节点的阈值，作为分箱的边界
    '''
    boundary = []  # 待return的分箱边界值列表

    x = x.fillna(-1).values  # 填充缺失值
    y = y.values

    clf = DecisionTreeClassifier(criterion='entropy',  # “信息熵”最小化准则划分
                                 max_leaf_nodes=6,  # 最大叶子节点数
                                 min_samples_leaf=0.05)  # 叶子节点样本数量最小占比

    clf.fit(x, y)  # 训练决策树
	
    #tree.plot_tree(clf) #打印决策树的结构图
    #plt.show()

    n_nodes = clf.tree_.node_count #决策树的节点数
    children_left = clf.tree_.children_left #node_count大小的数组，children_left[i]表示第i个节点的左子节点
    children_right = clf.tree_.children_right #node_count大小的数组，children_right[i]表示第i个节点的右子节点
    threshold = clf.tree_.threshold #node_count大小的数组，threshold[i]表示第i个节点划分数据集的阈值

    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  # 非叶节点
            boundary.append(threshold[i])

    boundary.sort()

    min_x = x.min()
    max_x = x.max() + 0.1  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本
    boundary = [min_x] + boundary + [max_x]

    return boundary


# In[4]:


def best_ks_box(data, var_name, box_num):
    data = data[[var_name, 'isDefault']]
    """
    KS值函数
    """
    def ks_bin(data_, limit):
        g = data_.iloc[:, 1].value_counts()[0]
        b = data_.iloc[:, 1].value_counts()[1]
        data_cro = pd.crosstab(data_.iloc[:, 0], data_.iloc[:, 1])
        data_cro[0] = data_cro[0] / g
        data_cro[1] = data_cro[1] / b
        data_cro_cum = data_cro.cumsum()
        ks_list = abs(data_cro_cum[1] - data_cro_cum[0])
        ks_list_index = ks_list.nlargest(len(ks_list)).index.tolist()
        for i in ks_list_index:
            data_1 = data_[data_.iloc[:, 0] <= i]
            data_2 = data_[data_.iloc[:, 0] > i]
            if len(data_1) >= limit and len(data_2) >= limit:
                break
        return i
    
    
    """
    区间选取函数
    """

    def ks_zone(data_, list_):
        list_zone = list()
        list_.sort()
        n = 0
        for val in list_:
            m = sum(data_.iloc[:, 0] <= val) - n
            n = sum(data_.iloc[:, 0] <= val)
            print(val,' , m:',m,' n:',n)
            list_zone.append(m)
        #list_zone[i]存放的是list_[i]-list[i-1]之间的数据量的大小
        list_zone.append(50000 - sum(list_zone))
        print('sum ',sum(list_zone[:-1]))
        print('list zone ',list_zone)
        #选取最大数据量的区间
        max_index = list_zone.index(max(list_zone))
        if max_index == 0:
            rst = [data_.iloc[:, 0].unique().min(), list_[0]]
        elif max_index == len(list_):
            rst = [list_[-1], data_.iloc[:, 0].unique().max()]
        else:
            rst = [list_[max_index - 1], list_[max_index]]
        return rst

    data_ = data.copy()
    limit_ = data.shape[0] / 20  # 总体的5%
    """"
    循环体
    """
    zone = list()
    for i in range(box_num - 1):
        #找出ks值最大的点作为切点，进行分箱
        ks_ = ks_bin(data_, limit_)
        zone.append(ks_)
        new_zone = ks_zone(data, zone)
        data_ = data[(data.iloc[:, 0] > new_zone[0]) & (data.iloc[:, 0] <= new_zone[1])]

    zone.append(data.iloc[:, 0].unique().max())
    zone.append(data.iloc[:, 0].unique().min())
    zone.sort()
    return zone


# In[5]:


#卡方分箱
# 计算2*2列联表的卡方值
def get_chi2_value(arr):
    rowsum = arr.sum(axis=1)  # 对行求和
    colsum = arr.sum(axis=0)  # 对列求和
    n = arr.sum()
    emat = np.array([i * j / n for i in rowsum for j in colsum])
    arr_flat = arr.reshape(-1)
    arr_flat = arr_flat[emat != 0]  # 剔除了期望为0的值,不参与求和计算，不然没法做除法！
    emat = emat[emat != 0]  # 剔除了期望为0的值,不参与求和计算，不然没法做除法！
    E = (arr_flat - emat) ** 2 / emat
    return E.sum()

# 自由度以及分位点对应的卡方临界值
def get_chi2_threshold(percents, nfree):
    return chi2.isf(percents, df=nfree)

# 计算卡方切分的切分点
def get_chimerge_cutoff(ser, tag, max_groups=None, threshold=None):
    freq_tab = pd.crosstab(ser, tag)
    cutoffs = freq_tab.index.values  # 保存每个分箱的下标
    freq = freq_tab.values  # [M,N_class]大小的矩阵，M是初始箱体的个数，N_class是目标变量类别的个数
    while True:
        min_value = None #存放所有对相邻区间中卡方值最小的区间的卡方值
        min_idx = None #存放最小卡方值的一对区间中第一个区间的下标
        for i in range(len(freq) - 1):
            chi_value = get_chi2_value(freq[i:(i + 2)]) #计算第i个区间和第i+1个区间的卡方值
            if min_value == None or min_value > chi_value:
                min_value = chi_value
                min_idx = i
        if (max_groups is not None and max_groups < len(freq)) or (
                threshold is not None and min_value < get_chi2_threshold(threshold, len(cutoffs)-1)):
            tmp = freq[min_idx] + freq[min_idx + 1] #合并卡方值最小的那一对区间
            freq[min_idx] = tmp
            freq = np.delete(freq, min_idx + 1, 0) #删除被合并的区间
            cutoffs = np.delete(cutoffs, min_idx + 1, 0)
        else:
            break
    return cutoffs


# In[ ]:




