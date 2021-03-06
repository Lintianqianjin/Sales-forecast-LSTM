# Sales-forecast-LSTM,based on Keras
使用LSTM预测商品销量，考虑销量激增点影响
## 数据来源
2019年华中数模大赛
## 数据预处理
关键问题在于让网络能学习到销量激增点，如双十一，这样的信息；
经过测试，最终发现效果最好的就是先从数据中识别这样的激增日，然后one-hot编码添加到输入数据中。
其它的处理包括归一化什么的，让数据的绝对数值不要太大就行，利于训练。
## 网络结构
比较简单的一个尝试，使用三层LSTM，激活函数用tanh,每一层接Dropout，最后接Dense。
![](https://github.com/Lintianqianjin/Sales-forecast-LSTM/blob/master/img/model_structure_sample.png)
## 效果示例
某次预测的结果。训练集是14个商品，验证集是2个商品，把表现最好的模型在测试集上测试。
以下是在测试集1个商品上的预测结果，可以说泛化能力是挺不错的。销量的绝对误差大约是9，均方根误差大概是50。
![](https://github.com/Lintianqianjin/Sales-forecast-LSTM/blob/master/img/predict_sample.png)
