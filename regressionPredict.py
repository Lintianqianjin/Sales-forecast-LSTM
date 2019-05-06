import numpy as np  # numpy库
import math
import matplotlib.pyplot as plt

from sklearn.linear_model import BayesianRidge, LinearRegression,Lasso,Ridge  # 批量导入要实现的回归算法
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法

from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法

import pandas as pd  # 导入pandas

###todo:自定义模型划分开始###
# 按时间跨度划分
def diySetSplit(X, Y, test_size=0.2):
    X_train=X[:int(len(X)*(1-test_size))+1]
    X_test=X[int(len(X)*(1-test_size))+1:]
    y_train=Y[:int(len(Y)*(1-test_size))+1]
    y_test=Y[int(len(Y)*(1-test_size))+1:]
    return X_train,X_test,y_train,y_test

###todo:自定义模型划分结束###


### todo : 定义全局用到的变量 开始###
loop_times = 1
test_size=0.2
model_metrics_list = np.zeros((4,4))
figure_fit_flag = False
model_names = ['线性回归','支持向量回归', '梯度增强回归' ,'岭回归']  # 不同模型的名称列表
### todo : 定义全局用到的变量 结束###

###todo : 导入数据 开始###
data = np.loadtxt(r'D:\Course Plus\CCMATH\第十二届华中地区数学建模大赛B题\回归数据\train_data.txt',
                      encoding='utf-8',skiprows=1,delimiter=',')  # 读取数据文件

data = np.log10(data+1)
# data = data/10000

X = data[:, :-1]  # 分割自变量
Y = data[:, -1]  # 分割因变量
###todo : 导入数据 结束###

###todo : 循环训练模型 开始###

for random_state in range(loop_times):
    print(random_state)

    #分割训练集和测试集开始
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,random_state=15)
    #分割训练集和测试集结束

    ###todo : 定义模型 开始###
    # model_br = BayesianRidge(n_iter=2, tol=0.01, alpha_1=0.1, alpha_2=0.00001, lambda_1=0.00001,
    #                          lambda_2=50)  # 建立贝叶斯岭回归模型对象
    model_lr = LinearRegression(fit_intercept=True, normalize=True)  # 建立普通线性回归模型

    model_svr_rbf = SVR(kernel='rbf', C=500, gamma=0.001, tol=0.0001,
                epsilon=0.0001)  # 建立支持向量机回归模型对象
    # model_svr_linear = SVR(kernel='linear', C=10,  tol=0.000001,
    #                 epsilon=0.000001)  # 建立支持向量机回归模型对象
    model_gbr = GradientBoostingRegressor(loss='ls',learning_rate = 0.1,tol=0.1,max_features='auto',
                                          n_estimators=100,max_depth=2)  # 建立梯度增强回归模型对象
    model_rg = Ridge(alpha=10, solver='sag') #建立岭回归
    ###todo : 定义模型 结束###

    #模型索引开始
    model_dic = [model_lr,model_svr_rbf, model_gbr, model_rg]  # 不同回归模型对象的集合
    #模型索引结束

    #定义预测对象，训练集或验证集
    if figure_fit_flag:
        preX = X_train
        y_check = y_train
    else:
        preX = X_test
        y_check = y_test

    pre_y_list = []  # 各个回归模型预测的y值列表

    ###todo: 训练模型开始###
    for model in model_dic:  # 读出每个回归模型对象
        model.fit(X_train, y_train)
        pre_y_list.append(model.predict(preX))  # 将回归训练中得到的预测y存入列表
    ###todo: 训练模型结束###

    n_samples, n_features = X.shape  # 总样本量,总特征数
    model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集

    ###todo:评价模型开始###
    for i in range(len(model_dic)):  # 循环每个模型索引
        for mindex,m in enumerate(model_metrics_name):  # 循环每个指标对象
            tmp_score = m(y_check, pre_y_list[i])  # 计算每个回归指标结果
            model_metrics_list[i][mindex] += tmp_score  # 将结果存入回归评估指标列表
    ###todo:评价模型结束###

    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.figure(figsize=(16,9))  # 创建画布
    # plt.plot(np.arange(X_test.shape[0]), y_test, color='k', label='真实值',linewidth = 1)  # 画出原始值的曲线
    # color_list = ['r', 'b', 'g', 'y', 'c']  # 颜色列表
    # linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
    # for i, pre_y in enumerate(pre_y_list):  # 读出通过回归模型预测得到的索引及结果
    #     if figure_fit_flag:
    #         shape0 = X_train.shape[0]
    #     else:
    #         shape0 = X_test.shape[0]
    #     plt.plot(np.arange(shape0), pre_y_list[i], color_list[i], label=model_names[i],linewidth = 1)  # 画出每条预测结果线
    # # plt.title('锰收得率拟合效果示意图',fontsize=24)  # 标题
    # #plt.title('碳收得率拟合效果示意图', fontsize=24)  # 标题
    # plt.legend(loc='lower left',fontsize=14)  # 图例位置
    # plt.ylabel('lg(销售量+1)',fontsize = 24)  # y轴标题
    # plt.xlabel('样本',fontsize =24)
    # plt.xticks(fontsize = 24)
    # plt.yticks(fontsize = 24)
    # # plt.show()  # 展示图像
    # plt.savefig('regression.png',dpi=300)
###todo : 循环训练模型 结束###

###todo:打印效果 开始

df2 = pd.DataFrame(model_metrics_list / loop_times, index=model_names,
                   columns=['ev', 'mae', 'mse', 'r2'])  # 建立回归指标的数据框
print(f'samples: {n_samples} \t pastDays: {n_features}')  # 打印输出样本量和特征数量
print(70 * '-')  # 打印分隔线
print('regression metrics:')  # 打印输出标题
print(df2)  # 打印输出回归指标的数据框
print(70 * '-')  # 打印分隔线
print('short name \t full name')  # 打印输出缩写和全名标题
print('ev \t explained_variance')
print('mae \t mean_absolute_error')
print('mse \t mean_squared_error')
print('r2 \t r2')
print(70 * '-')  # 打印分隔线

###todo:打印效果 结束