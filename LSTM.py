import keras
from keras.layers import Input,LSTM,Dense,Dropout,Activation,ConvLSTM2D,BatchNormalization
# from keras.layers.convolutional_recurrent impor
from keras.utils.vis_utils import plot_model
from keras.initializers import glorot_uniform,truncated_normal
import numpy as np
from keras.models import Model,Sequential
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.models import load_model
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint


import matplotlib.pyplot as plt
import time,datetime
import math

###导入自己的文件 开始###

from LSTMInputDataPreprocess import *

###导入自己的文件 结束###

def root_mean_squared_error(y_pred,y_true):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def loadData():
    ###todo : 导入数据 开始###
    data = np.loadtxt(r'D:\Course Plus\CCMATH\第十二届华中地区数学建模大赛B题\回归数据\train_data.txt',
                      encoding='utf-8', skiprows=0, delimiter=',')  # 读取数据文件

    data = np.log10(data + 1)

    X = data[:, :-1]  # 分割自变量
    Y = data[:, -1]  # 分割因变量
    ###todo : 导入数据 结束###

    return X,Y

def buildConvLSTM(output_size=1, steps = 1,dropout=0.3):
    model = Sequential()
    # model.add(Input())
    # model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
    #            input_shape=(None, 40, 40, 1),
    #            padding='same', return_sequences=True))

    model.add(ConvLSTM2D(filters=16, kernel_size=(1,1),padding='same',input_shape= (None,1,1,5),use_bias=True,
                         bias_initializer='glorot_uniform',return_sequences=True,
                        activation='tanh'))
    # input_shape = (inputShape[0], inputShape[1])
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(ConvLSTM2D(filters=16, kernel_size=(1,1),padding='same', use_bias=True,
                         bias_initializer='glorot_uniform',return_sequences=True, activation='tanh'))
    model.add(Dropout(dropout))
    model.add(ConvLSTM2D(filters=16, kernel_size=(1,1),padding='same', use_bias=True,
                         bias_initializer='glorot_uniform',return_sequences=True,activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    # model.add(LSTM(64, activation='tanh', return_sequences=True))
    # model.add(Dropout(dropout))
    model.add(Dense(units=output_size,use_bias=True,bias_initializer='glorot_uniform',activation='linear'))
    # model.add(Activation('linear'))
    model.compile(loss=root_mean_squared_error, optimizer='adam', metrics=['mae'])
    model.summary()
    curT = datetime.datetime.utcfromtimestamp(time.time()).strftime("%d-%H-%M-%S")
    plot_model(model, to_file=f'{curT}modelConvLSTM.png', show_shapes=True)
    return model

def buildLSTM(inputShape=(), output_size=1, neurons=256,dropout=0.3):
    model = Sequential()
    # use_bias = True, bias_initializer = 'glorot_uniform',
    model.add(LSTM(neurons, return_sequences=True,
                   use_bias=True, bias_initializer='glorot_uniform',
                   input_shape=(inputShape[0], inputShape[1]),
                   activation='tanh'))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons, return_sequences=True, activation='tanh',
                   use_bias=True, bias_initializer='glorot_uniform'))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons, activation='tanh', return_sequences=True,
                   use_bias=True, bias_initializer='glorot_uniform'))
    # model.add(BatchNormalization())
    model.add(Dropout(dropout))
    # model.add(LSTM(64, activation='tanh', return_sequences=True))
    # model.add(Dropout(dropout))
    model.add(Dense(units=output_size, activation='linear'))
    # model.add(Activation('linear'))
    # K.set_value(model.optimizer.lr, 0.01)
    model.compile(loss=root_mean_squared_error, optimizer='adam', metrics=['mae'])
    model.summary()
    curT = datetime.datetime.utcfromtimestamp(time.time()).strftime("%d-%H-%M-%S")
    plot_model(model, to_file=f'{curT}_modelLSTM.png', show_shapes=True)
    return model

def genConvLSTMData(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=724)

    X_train = np.array([[[[x]]] for index,x in enumerate(X_train)])
    y_train = np.array([[[[[y]]]] for index,y in enumerate(y_train)])
    X_test = np.array([[[[x]]] for index,x in enumerate(X_test)])
    y_test = np.array([[[[[y]]]] for index,y in enumerate(y_test)])

    return X_train, y_train, X_test, y_test

def genLSTMData(X,Y,type = None,burstType='onehot'):
    if type == 'oneCommodity':
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=724)
    else:
    #multi commodity
        X_train = X
        y_train = Y
        X_test,y_test = loadSeq2seqData(steps=1,containDist=True,containMonth=False,containDate=False,normType='log',filepath='test3.txt',burstType=burstType)

    X_train = np.array([[x] for x in X_train])
    y_train = np.array([[y] for y in y_train])
    X_test = np.array([[x] for x in X_test])
    y_test = np.array([[y] for y in y_test])

    print(f'训练X数据 {X_train.shape} 测试X数据 {X_test.shape}')
    print(X_train[0])
    print(f'训练标签数据 {y_train.shape} 测试标签数据 {y_test.shape}')
    print(y_train[0])

    time.sleep(2)

    return X_train,y_train,X_test,y_test

def trainSaveModel(isSaveModel = False,modelType = 'LSTM',outPutSize = 1,steps = 1,neurons = 256,datSetType = 'multiCommodity',burstType = 'onehot'):
    X, Y = loadSeq2seqData(steps=steps,containDist=True,containMonth=False,containDate=False,normType='log',burstType=burstType,filepath='train3.txt')
    # print(X[0])
    if modelType == 'LSTM':
        model = buildLSTM(inputShape=(steps, X.shape[-1]), output_size=outPutSize, neurons=neurons, dropout=0.25)
        X_train, y_train, X_test, y_test = genLSTMData(X, Y,type=datSetType,burstType=burstType)
    elif modelType == 'ConvLSTM':
        model = buildConvLSTM(steps=1, output_size=outPutSize, dropout=0.25)
        X_train, y_train, X_test, y_test = genConvLSTMData(X, Y)
    else:
        print('默认LSTM')
        model = buildLSTM(inputShape=(steps, X.shape[-1]), output_size=outPutSize, neurons=neurons, dropout=0.25)
        X_train, y_train, X_test, y_test = genLSTMData(X, Y,type=datSetType,burstType=burstType)

    loss = []
    mae = []
    vloss = []
    vmae = []

    epochs = 24
    for i in range(epochs):
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
        # filepath = "model/weights-improvement-{epoch:02d}-{val_loss:.4f}.h5"
        # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
        #                              mode='min')
        # keras.optimizers.Adam(lr=0.0012, beta_1=0.9, beta_2=0.9, epsilon=1e-08, amsgrad=True)

        log = model.fit(X_train, y_train,batch_size=4, epochs=1, verbose=0, validation_data=(X_test, y_test))
        loss.append(log.history['loss'])
        mae.append(log.history['mean_absolute_error'    ])
        val_l = log.history['val_loss']
        vloss.append(log.history['val_loss'])
        val_a = log.history['val_mean_absolute_error']
        vmae.append(log.history['val_mean_absolute_error'])
        print('epoch ' + str(i))
        print('rmse'+str(val_l[0]))
        print('mae'+str(val_a[0]))
        if isSaveModel and val_l[0]<0.0535:
            model.save(f'model/3layersLSTM_{i}_vl_{val_l}_va_{val_a}.h5')
    # plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot([epoch for epoch in range(1, epochs + 1)], loss, label='训练集均方根误差')
    plt.plot([epoch for epoch in range(1, epochs + 1)], mae, label='训练集绝对误差')
    plt.plot([epoch for epoch in range(1, epochs + 1)], vloss, label='测试集均方根误差')
    plt.plot([epoch for epoch in range(1, epochs + 1)], vmae, label='测试集绝对误差')
    plt.legend(fontsize=14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel('训练集迭代次数', fontsize=16)
    plt.ylabel('均方根误差/平均绝对误差', fontsize=16,labelpad=10)
    # plt.show()
    curT = datetime.datetime.utcfromtimestamp(time.time()).strftime("%d-%H-%M-%S")
    plt.tight_layout()
    plt.savefig(curT+'-train-test-loss.png',dpi=300)

def pltFig(modelFile = None,return_real_sale = False,steps=1,predLength = 5):
    model = load_model(modelFile,custom_objects={'root_mean_squared_error':root_mean_squared_error})
    X, Y = loadSeq2seqData(steps=steps,containDist=True,containMonth=False,containDate=False,normType='log',filepath='val3.txt',burstType='onehot')
    # 用于画图的横坐标
    n_samples = len(Y)
    # 用于画图的纵坐标
    ori_y = [y.mean() for y in Y]

    X = np.array([[x] for x in X])
    pred_Y = np.array(model.predict(X)).flatten().reshape(n_samples,predLength)

    pred_Y = [y.mean() for y in pred_Y]

    if return_real_sale:
        ori_y =argLogSalesAmount(ori_y)
        pred_Y =argLogSalesAmount(pred_Y)
    # print(pred_Y)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(22,11))
    plt.plot([index for index in range(n_samples)], ori_y,label = '真实值')
    plt.plot([index for index in range(n_samples)], pred_Y, label = '预测值')
    plt.legend(fontsize = 32)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlabel('天', fontsize=28)
    if return_real_sale:
        plt.ylabel('销量', fontsize=32, labelpad=10)
    else:
        plt.ylabel('lg(销量+1)/5', fontsize=32, labelpad=5)
    # plt.show()
    curT = datetime.datetime.utcfromtimestamp(time.time()).strftime("%d-%H-%M-%S")
    plt.tight_layout()
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
    plt.savefig(f"{curT} model_predict.png",dpi = 300)

    mae = np.mean(abs(np.array(ori_y) - np.array(pred_Y)))
    rmse = math.sqrt(sum([(y-x)**2 for x,y in zip(ori_y,pred_Y)])/len(ori_y))

    print('mae'+str(mae))
    print('rmse'+str(rmse))

if __name__ == '__main__':
    # trainSaveModel(isSaveModel=True,modelType='LSTM',steps=1,outPutSize=5,neurons=96,datSetType = 'multiCommodity')
    # build_model(inputShape=(1, 5), output_size=1, neurons=128, dropout=0.25)
    mf = r'3layersLSTM_15_vl_[0.05285577177826507]_va_[0.04395105661780622].h5'
    pltFig(modelFile=f'D:\Course Plus\CCMATH\code\model\\{mf}',
                     return_real_sale=False,steps=1,predLength = 5)