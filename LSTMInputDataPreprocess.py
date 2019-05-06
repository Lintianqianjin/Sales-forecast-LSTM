###本文件用于处理 将要输入LSTM网络的源数据###
import numpy as np
import json
import os
import regex as re

def loadSeq2seqData(steps=1, outputSize = 1,burstType = 'dist',normType = 'norm',containDist=True, containDate=True, containMonth = True,filepath = r'train2.txt'):

    data = np.loadtxt(r'dataset\\'+filepath,
                      encoding='utf-8', skiprows=0, delimiter=',')  # 读取数据文件
    seq_train = data[:,:7]
    seq_pred = data[:, -5:]
    date = data[:,-24:-17]
    month = data[:,-17:-5]

    if burstType == 'dist':
        distToBurst = data[:, 7:35]
        norm_distToBurst = distanceToBurstDayNormalize(distToBurst)
    elif burstType == 'onehot':
        distToBurst = data[:, 7:23]
        norm_distToBurst = distToBurst
    else:
        distToBurst = data[:, 7:23]
        norm_distToBurst = distToBurst

    if normType == 'norm':
        norm_seq_train,norm_seq_pred,maxSale,minSale = singleCommoditySalesNormalize(seq_train,seq_pred)
        stacks = [norm_seq_train]
        if containDist:
            stacks.append(norm_distToBurst)
        if containDate:
            stacks.append(date)
        if containMonth:
            stacks.append(month)

        stacks = tuple(stacks)

        normed_x = np.hstack(stacks)
        normed_y = norm_seq_pred

        X = list()
        Y = list()
        for index in range(len(normed_x)-steps+1):
            x = np.array(normed_x[index:index + steps]).reshape((normed_x.shape[1],))
            X.append(x)
            y = np.array(normed_y[index+steps-1]).reshape((normed_y.shape[1],))
            Y.append(y)

        X = np.array(X)
        Y = np.array(Y)
        return X,Y,maxSale,minSale
    else:
        norm_seq_train, norm_seq_pred = singleCommoditySalesLog(seq_train, seq_pred)
        stacks = [norm_seq_train]
        if containDist:
            stacks.append(norm_distToBurst)
        if containDate:
            stacks.append(date)
        if containMonth:
            stacks.append(month)

        stacks = tuple(stacks)

        normed_x = np.hstack(stacks)
        normed_y = norm_seq_pred

        X = list()
        Y = list()
        for index in range(len(normed_x) - steps + 1):
            x = np.array(normed_x[index:index + steps]).reshape((normed_x.shape[1],))
            X.append(x)
            y = np.array(normed_y[index + steps - 1]).reshape((normed_y.shape[1],))
            Y.append(y)

        X = np.array(X)
        Y = np.array(Y)
        return X, Y


def mergeCommodities():
    newUp = json.load(open(r'D:\Course Plus\CCMATH\第十二届华中地区数学建模大赛B题\货物上新时间统计.json', 'r', encoding='utf-8'))
    BaseDir = r'D:\Course Plus\CCMATH\第十二届华中地区数学建模大赛B题\训练数据1.5\训练数据1.5'
    merged_train = open(r'dataset\train3.txt','w',encoding='utf-8')
    merged_test = open(r'dataset\test3.txt','w',encoding='utf-8')
    merged_val = open(r'dataset\val3.txt','w',encoding='utf-8')
    dataFileList = os.listdir(BaseDir)
    for index, upDate in enumerate(newUp):
        type = newUp[index][0]
        if type == '2018-03-13':
            commodities = [name.split('-')[0] for name in newUp[index][1]]
            commodities.remove('SS81066')
            commoditiesTrain = commodities[:14]
            commoditiesTest = commodities[14:]
            break
    else:
        print('error: no update in dir')
        exit()

    for datafile in dataFileList:
        if re.search('SS\d{5}',datafile).group() in commoditiesTrain:
            lines = open(os.path.join(BaseDir, datafile),encoding='gbk').readlines()[1:]
            for line in lines:
                merged_train.write(line)
        if re.search('SS\d{5}', datafile).group() in commoditiesTest:
            lines = open(os.path.join(BaseDir, datafile), encoding='gbk').readlines()[1:]
            for line in lines:
                merged_test.write(line)
        if re.search('SS\d{5}', datafile).group() == 'SS81066':
            lines = open(os.path.join(BaseDir, datafile), encoding='gbk').readlines()[1:]
            for line in lines:
                merged_val.write(line)


def singleCommoditySalesLog(X, Y):
    if type(X) != type(np.array([0])):
        X = np.array(X)
    if type(Y) != type(np.array([0])):
        Y = np.array(Y)

    X = np.log10(X+1)/5
    Y = np.log10(Y+1)/5

    return X,Y

def singleCommoditySalesNormalize(X,Y):
    # X为输入的时间序列数据，Y为输出的时间序列数据

    # 基于假设各商品的变化趋势是类似的，至少同一上新日期的商品的变化趋势是类似的
    # 差别体现在商品类别带来的不同造成的绝对销量的差异
    # 因此归一化后，同一上新日期的数据能一起送入LSTM模型训练
    # 归一化的方式是
    #              当天销售量-该商品销售期间最小值
    #          ————————————————————————————————————
    #          该商品销售期间的最大值-该商品销售期间最小值

    if type(X) != type(np.array([0])):
        X = np.array(X)
    if type(Y) != type(np.array([0])):
        Y = np.array(Y)

    maxSale = max([X.max(),Y.max()])
    minSale = min([X.min(),Y.min()])

    X = (X-minSale)/(maxSale-minSale)
    Y = (Y-minSale)/(maxSale-minSale)
    return X,Y,maxSale,minSale

def distanceToBurstDayNormalize(X):
    # X为将要预测的序列的第一天距离设置的爆点的距离的矩阵
    # 归一化的方式是 x/365
    if type(X) != type(np.array([0])):
        X = np.array(X)

    X = X/365

    return X

def argNormSalesAmount(X,maxSale,minSale):
    if type(X) != type(np.array([0])):
        X = np.array(X)
    return X*(maxSale-minSale)+minSale

def argLogSalesAmount(X):
    if type(X) != type(np.array([0])):
        X = np.array(X)
    return np.power(10,X*5)-1
if __name__ == '__main__':
    pass
    mergeCommodities()
    # X = [[1,2],[2,3],[3,4],[5,6]]
    # Y = [[2],[3],[4],[7]]
    # X,Y = singleCommoditySalesNormalize(X,Y)
    # print(X)
    # print(Y)

    # X = [[0,0,0,0,0,0,1,5,-5,-8,-224,321]]
    # X = distanceToBurstDayNormalize(X)
    # print(X)
    # X, Y,maxSale,minSale = loadSeq2seqData()
    # print(Y)
    # print(argNormSalesAmount(Y,maxSale,minSale))
    # list = [5930,321,228,207,214,245,209,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,172,177,171,125,66]
    # print(list[])