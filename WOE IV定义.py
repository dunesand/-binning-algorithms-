#!/usr/bin/env python
# coding: utf-8

# In[1]:


def WOE(data, feat, label):

    bin_values = data[feat].unique()
    good_total_num = len(data[data[label]==1])
    bad_total_num = len(data[data[label]==0])

    woe_dic = {}
    df = pd.DataFrame()
    for i,val in enumerate(bin_values):
        good_num = len(data[(data[feat]==val) & (data[label]==1)])
        bad_num = len(data[(data[feat]==val) & (data[label]==0)])
        df.loc[i,feat] = val
        df.loc[i, feat+'_woe'] = np.log( (good_num/good_total_num) / ((bad_num/bad_total_num+0.0001)) )
        woe_dic[val] = np.log( (good_num/good_total_num) / ((bad_num/bad_total_num+0.0001)) )

    return woe_dic,df


# In[ ]:


def IV(data, woe_dic, feat, label):
    good_total_num = len(data[data[label] == 1])
    bad_total_num = len(data[data[label] == 0])
    bin_values = data[feat].unique()
    feat_IV = 0
    for val in bin_values:
        woe = woe_dic[val]
        good_num = len(data[(data[feat] == val) & (data[label] == 1)])
        bad_num = len(data[(data[feat] == val) & (data[label] == 0)])

        feat_IV += ((good_num/good_total_num)-(bad_num/bad_total_num))*woe

    return feat_IV

