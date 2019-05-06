import regex as re
import matplotlib.pyplot as plt
from matplotlib.ticker import  FormatStrFormatter
from matplotlib.ticker import  MultipleLocator
import math
import numpy as np
import os
import json

class drawSaleFig():
    def __init__(self):
        pass

    def loadData(self,file=None):
        id = re.search('SS.*?$',file).group()
        x = list()
        y = list()

        for index,line in enumerate(open(file,'r',encoding='utf-8')):
            partions = line.strip().split(',')
            x.append(int(partions[1])-5840)
            y.append(math.log10(int(partions[2])+1))

        # y = list(math.log10(np.array(y)+1))

        return id, x ,y

    def loadData2(self,file=None):
        id = re.search('SS.*?-',file).group().strip('-')
        x = list()
        y = list()

        for index,line in enumerate(open(file,'r',encoding='utf-8')):
            if index ==0 :
                continue
            partions = line.strip().split(',')
            x.append(int(partions[0])-5840)
            y.append(math.log10(int(partions[1])+1))

        # y = list(math.log10(np.array(y)+1))

        return id, x ,y

    def drawFig(self,x=None,y=None,id=''):
        plt.figure(figsize=(50,10))
        plt.plot(x, y, marker='o', mec='r', mfc='w', label=id)
        plt.xticks(rotation = 90)
        plt.xlabel('date',fontsize = 40)
        plt.ylabel('sales',fontsize = 40)
        # plt.show()
        plt.savefig(f'D:\\Course Plus\\CCMATH\\第十二届华中地区数学建模大赛B题\\商品趋势图\\{id}.png',dpi=300)
        plt.close()

if __name__ == '__main__':

    newUp = json.load(open(r'D:\Course Plus\CCMATH\第十二届华中地区数学建模大赛B题\货物上新时间统计.json','r',encoding='utf-8'))

    BaseDir = r'D:\Course Plus\CCMATH\第十二届华中地区数学建模大赛B题\上新日-该上新日最后一天'
    fileList = os.listdir(BaseDir)

    for index, upDate in enumerate(newUp):
        draw_file = [name.split('-')[0] for name in newUp[index][1]]
        dsp = drawSaleFig()
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(20, 10))
        for file in fileList:
            print(file)
            if file.split('-')[0] in draw_file:
                id,x,y = dsp.loadData2(os.path.join(BaseDir,file))
                plt.plot(x[:100],y[:100], label = id)

        plt.legend(fontsize = 18)
        plt.xticks(rotation=90)
        xminorLocator = MultipleLocator(5)
        ymajorLocator = MultipleLocator(0.1)
        plt.xlabel('距离2017/1/1的天数', fontsize=28,labelpad=5)
        plt.ylabel('lg(销量+1)', fontsize=28,labelpad=20)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        # plt.show()
        plt.savefig('D:\\Course Plus\\CCMATH\\第十二届华中地区数学建模大赛B题\\商品趋势图按上新日（截断用于展示）\\'+newUp[index][0]+'.png',dpi = 300)
        plt.close()