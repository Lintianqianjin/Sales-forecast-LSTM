# 导入库
import numpy as np  # numpy库
from sklearn.linear_model import BayesianRidge, LinearRegression,Lasso,Ridge  # 批量导入要实现的回归算法
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法
from sklearn.preprocessing import PolynomialFeatures# 导入多项式
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import pandas as pd  # 导入pandas
import matplotlib.pyplot as plt  # 导入图形展示库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import math


###todo : 导入数据 开始###
data = np.loadtxt(r'D:\Course Plus\华中数模\第十二届华中地区数学建模大赛B题\回归数据\train_data.txt',
                      encoding='utf-8',skiprows=1,delimiter=',')  # 读取数据文件

data = np.log10(data+1)

X = data[:, :-1]  # 分割自变量
Y = data[:, -1]  # 分割因变量
###todo : 导入数据 结束###

# model_br = BayesianRidge(n_iter=2, tol=0.01, param_1=0.1, param_2=0.00001, lambda_1=0.00001,
#                              lambda_2=50)  # 建立贝叶斯岭回归模型对象
# model_lr = LinearRegression(fit_intercept=True, normalize=True)  # 建立普通线性回归模型对
#
# model_svr = SVR(kernel='rbf', C=3400, param=0.0016, tol=0.001,
#                 epsilon=0.14)
# model_svr = SVR(kernel='linear', C=10,  tol=0.000001,
#                 epsilon=0.000001)
param = [0.00001,0.0001,0.001,0.01,0.1,1,10,100]
# param = [1,10,100,1000]
param_log = []
for a in param:
    param_log.append(math.log10(a))
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]
measure_name = ['解释方差','平均绝对误差','均方误差','拟合优度']
param_score = np.zeros((4,len(param)))
for index_param, cur_param in enumerate(param):
    print(index_param)
    #初始化模型
    # model = SVR(kernel='rbf', C=1000, gamma=0.001, tol=0.0001,
    #                 epsilon=0.0001)
    # model = GradientBoostingRegressor(loss='ls',learning_rate = 0.1,tol=0.1,max_features='auto',n_estimators=100,max_depth=3,subsample=0.85)
    model = Ridge(alpha=cur_param, solver='sag')
    model_score_list=np.zeros(4)  # 将结果存入回归评估指标列表
    for random_state in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=random_state)
        model.fit(X_train, y_train)
        pre_y_list = (model.predict(X=X_test))  # 将回归训练中得到的预测y存入列表
        for mindex,m in enumerate(model_metrics_name):  # 循环每个指标对象
            tmp_score = m(y_test, pre_y_list)  # 计算每个回归指标结果
            model_score_list[mindex]+=tmp_score

    model_score_list = model_score_list/100
    for score_index,mean_score in enumerate(model_score_list):
        param_score[score_index][index_param] = mean_score
print(param_score)

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(18,10))  # 创建画布
color_list = ['r', 'b', 'g', 'y', 'c']  #l 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
for i, scores in enumerate(param_score):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(param_log, scores, color_list[i], label=measure_name[i])  # 画出每条预测结果线

plt.legend(loc='upper left',fontsize =24)  # 图例位置
plt.ylabel('衡量指标值',fontsize = 24)  # y轴标题

plt.xlabel('lg(Param)',fontsize = 24)
plt.xticks(fontsize = 24)
plt.yticks(fontsize = 24)

plt.show()  # 展示图