# -*- coding: utf-8 -*-
# 张嘉玮
# 20190502
# 自61，2016011528

import torch
import torch.autograd
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
import math

# random.seed(428)
def Model(T):
    """
    原模型
    :param T: 温度（单位为K）
    :return: 电阻（单位：千欧）
    """

    beta = 3100
    R_T_0 = 22
    T_0 = 25 + 273.15

    R_T = R_T_0*math.exp(beta*(1/T-1/T_0))

    return R_T

def Plot2(Ts,Rs_real,Rs_poly,n):
    """
    任务2
    将回归模型和原数据画在一张图上
    :param Ts: 温度
    :param Rs_real: 原始数据
    :param Rs_poly: 回归得到的的数据
    :param n:回归模型最高次项
    :return:
    """
    plt.title('Resistance--Temperature:n = '+str(n+1))
    plt.plot(Ts, Rs_real, label='R_Real(noised)', linewidth=2, color='orange', marker='o',
             markerfacecolor='purple', markersize=4)
    plt.plot(Ts, Rs_poly, label='R_Poly', linewidth=2, color='lime', marker='o',
             markerfacecolor='purple', markersize=4)
    plt.xlabel('T(C)')
    plt.ylabel('R(Kohm)')
    plt.legend(loc='upper right')
    plt.show()

def PlotLoss(loss,lrs,n,Train = True):
    """
    Loss曲线
    :param loss:
    :param Train:
    :return:
    """
    N = [i + 1 for i in range(len(loss[0]))]
    scatterColors = ['orangered', 'lawngreen', 'orange', 'blueviolet', 'gold', 'turquoise']
    if Train:
        plt.title('Loss--Epoch Of Train,n='+str(n))
    else:
        plt.title('Loss--Epoch Of Train,n='+str(n))
    for j,lr in enumerate(lrs):
        colour = scatterColors[j% len(scatterColors)]
        plt.plot(N, loss[j], label='Lr:'+str(lr),  color=colour)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

def Plot4(MSE_Errors_Train,MSE_Errors_Test):
    """
    对比两个误差
    :param MSE_Errors_Train:
    :param MSE_Errors_Test:
    :return:
    """
    N = [i + 1 for i in range(len(MSE_Errors_Train))]
    plt.title('MES_Error--N')
    plt.plot(N, MSE_Errors_Train, label='MSE_Errors_Train', linewidth=2, color='lawngreen', marker='o',
             markerfacecolor='purple', markersize=4)
    plt.plot(N, MSE_Errors_Test, label='MSE_Errors_Test', linewidth=2, color='red', marker='o',
             markerfacecolor='purple', markersize=4)
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.legend(loc='upper right')
    plt.show()

def Plot5(Error,Train = True,Nums = 10,ns = 6):
    """
    进行10次实验的结果可视化；
    分为横向和纵向比较
    :param Error:
    :param Train:
    :param Nums:
    :param ns:
    :return:
    """
    Num = [i+1 for i in range(Nums)]
    n = [i+1 for i in range(ns)]
    scatterColors = ['orange','lawngreen', 'purple', 'brown', 'black', 'blue', 'green', 'yellow', 'red']

    if Train:
        plt.title('MES_Error--N Of Train')
    else:
        plt.title('MES_Error--N Of Test')

    for i in Num:
        color = scatterColors[(i-1) % len(scatterColors)]
        plt.plot(n, Error[i-1], label='MSE_Errors_Time_'+str(i), linewidth=2, color=color, marker='o',
                 markerfacecolor='purple', markersize=4)

    plt.xlabel('N')
    plt.ylabel('Error')
    plt.legend(loc='upper right')
    plt.show()


    if Train:
        plt.title('MES_Error--N Of Train')
    else:
        plt.title('MES_Error--N Of Test')
    Error = np.array(Error)
    Error_T = np.transpose(Error)
    for i in n:
        color = scatterColors[(i - 1) % len(scatterColors)]
        error = Error_T[i - 1]
        plt.plot(Num,error, label='N(poly)_' + str(i), linewidth=2, color=color, marker='o',
                 markerfacecolor='purple', markersize=4)

    plt.xlabel('Times')
    plt.ylabel('Error')
    plt.legend(loc='upper right')
    plt.show()

def Data(SIGMA = 0.5):
    """
    数据生成
    :param SIGMA:
    :return:
    """

    Ts_C = [i for i in range(0, 101, 2)]
    Ts_K = [i + 273.15 for i in Ts_C]
    Rs_KOhm = [Model(T) for T in Ts_K]
    MU = 0
    Rs_KOhm_Noised = [i + random.gauss(MU, SIGMA) for i in Rs_KOhm]


    Ts_C_Train = Ts_C[10:41]
    Ts_C_Test = Ts_C[0:10] + Ts_C[41:]
    Rs_KOhm_Train = Rs_KOhm_Noised[10:41]
    Rs_KOhm_Test = Rs_KOhm[0:10] + Rs_KOhm[41:]

    Mean = np.zeros(4)
    Var = np.zeros(4)

    Mean[0] = np.mean(Ts_C_Train)
    Var[0] = np.var(Ts_C_Train)

    Mean[1] = np.mean(Rs_KOhm_Train)
    Var[1] = np.var(Rs_KOhm_Train)

    Mean[2] = np.mean(Ts_C_Test)
    Var[2] = np.var(Ts_C_Test)

    Mean[3] = np.mean(Rs_KOhm_Test)
    Var[3] = np.var(Rs_KOhm_Test)

    Ts_C_Train_N = np.zeros(len(Ts_C_Train))
    Rs_KOhm_Train_N = np.zeros(len(Rs_KOhm_Train))
    for i in range(len(Ts_C_Train)):
        Ts_C_Train_N[i] = (Ts_C_Train[i] - Mean[0])/np.sqrt(Var[0])
        Rs_KOhm_Train_N[i] = (Rs_KOhm_Train[i] - Mean[1])/np.sqrt(Var[1])

    Ts_C_Test_N = np.zeros(len(Ts_C_Test))
    Rs_KOhm_Test_N = np.zeros(len(Rs_KOhm_Test))
    for i in range(len(Ts_C_Test)):
        Ts_C_Test_N[i] = (Ts_C_Test[i] - Mean[2])/np.sqrt(Var[2])
        Rs_KOhm_Test_N[i] = (Rs_KOhm_Test[i] - Mean[3])/np.sqrt(Var[3])

    return np.array(Ts_C_Train),np.array(Ts_C_Train_N),\
           np.array(Ts_C_Test),np.array(Ts_C_Test_N),\
           np.array(Rs_KOhm_Train),np.array(Rs_KOhm_Train_N),\
           np.array(Rs_KOhm_Test),np.array(Rs_KOhm_Test_N),\
           Mean,Var

def NormalBack(Y,m,v):
    """
    去归一化
    :param Y:
    :param m:
    :param v:
    :return:
    """
    Y_new = np.zeros(len(Y))
    for i in range(len(Y)):
        Y_new[i] = m + Y[i]*np.sqrt(v)
    return Y_new

def DataT(Ts_C,n):
    """
    多项式数据准备
    :param Ts_C:
    :param n:
    :return:
    """
    Ts_C_Set = np.stack([Ts_C **i for i in range(1,n+1)],axis=1)
    return np.float64(Ts_C_Set)


def ModelPrint(ploy):
    """
    模型输出
    :param ploy:
    :return:
    """

    n = len(ploy)
    Model = 'R(t) = '+str(ploy[0])+'t**'+str(n-1)
    for i,modulus in enumerate(ploy[1:]):
        if modulus>=0:
            Model = Model + ' + '+ str(modulus)+'t**'+str(n-2-i)
        else:
            Model = Model + ' - '+ str(abs(modulus)) + 't**' + str(n - 2 - i)
    print(Model)


def MultiLinear(Ts_C,w,b):
    return torch.mm(Ts_C,w)+b

def Loss(Rs_Poly , Rs_train):
    return torch.mean((Rs_Poly-Rs_train)**2)

def MSE(Rs_KOhm_reality,Rs_KOhm_Poly):
    """
    求两组数据的均方差
    :param Rs_KOhm_reality:
    :param Rs_KOhm_Poly:
    :return:
    """
    return sum([(R_real-R_poly)**2 for R_real,R_poly in zip(Rs_KOhm_reality,Rs_KOhm_Poly)])/len(Rs_KOhm_reality)


def Task6():
    """
    任务6
    :return:
    """
    # lrs = [0.00001,0.0001,0.001,0.01]
    lrs = [0.0001]
    MES_Error_Train = []
    MES_Error_Test = []
    for i in range(1,7):
        print("多项式最高次项：",i)
        loss_data = [[] for _ in lrs]
        for j,lr in enumerate(lrs):
            Ts_C_Train,Ts_C_Train_N, Ts_C_Test,Ts_C_Test_N, Rs_KOhm_Train,Rs_KOhm_Train_N, Rs_KOhm_Test,Rs_KOhm_Test_N ,Mean,Var= Data(SIGMA=0.5)

            Ts_C_Train_Perpare = DataT(Ts_C_Train_N,i)
            Ts_C_Train_Perpare = torch.from_numpy(Ts_C_Train_Perpare).float()

            Ts_C_Test_Perpare = DataT(Ts_C_Test_N,i)
            Ts_C_Test_Perpare = torch.from_numpy(Ts_C_Test_Perpare).float()

            Rs_KOhm_Train_N = torch.from_numpy(Rs_KOhm_Train_N).float().unsqueeze(1)
            Rs_KOhm_Test_N = torch.from_numpy(Rs_KOhm_Test_N).float().unsqueeze(1)

            w = Variable(torch.rand(i,1),requires_grad=True)
            # w = Variable(torch.zeros(i, 1), requires_grad=True)
            b = Variable(torch.zeros(1),requires_grad=True)

            Ts_C_Train_Perpare = Variable(Ts_C_Train_Perpare)
            Rs_KOhm_Train_N = Variable(Rs_KOhm_Train_N)

            Ts_C_Test_Perpare = Variable(Ts_C_Test_Perpare)
            Rs_KOhm_Test_N = Variable(Rs_KOhm_Test_N)

            for e in range(10000):
                Rs_KOhm_Train_Poly = MultiLinear(Ts_C_Train_Perpare,w,b)
                loss = Loss(Rs_KOhm_Train_Poly,Rs_KOhm_Train_N)

                if e !=0:
                    w.grad.data.zero_()
                    b.grad.data.zero_()

                loss.backward()

                w.data = w.data - lr*w.grad.data
                b.data = b.data - lr*b.grad.data

                loss_data[j].append(float(loss))
                if (e+1)%20 == 0:
                    print('epoch {}, MSE_Loss: {:.5f}'.format(e + 1, float(loss)))

            Rs_KOhm_Train_Poly = MultiLinear(Ts_C_Train_Perpare,w,b)

            Rs_KOhm_Train_Poly = NormalBack(Rs_KOhm_Train_Poly,Mean[1],Var[1])
            Rs_KOhm_Train_Poly.tolist()
            # Plot2(Ts_C_Train,Rs_KOhm_Train,Rs_KOhm_Train_Poly,i-1)


            Rs_KOhm_Test_Poly = MultiLinear(Ts_C_Test_Perpare,w,b)
            Rs_KOhm_Test_Poly = NormalBack(Rs_KOhm_Test_Poly,Mean[3],Var[3])
            Rs_KOhm_Test_Poly.tolist()
            # Plot2(Ts_C_Test,Rs_KOhm_Test,Rs_KOhm_Test_Poly,i-1)

            MES_Error_Train.append(MSE(Rs_KOhm_Train,Rs_KOhm_Train_Poly))
            MES_Error_Test.append(MSE(Rs_KOhm_Test,Rs_KOhm_Test_Poly))
        # PlotLoss(loss_data,lrs,i)
    Plot4(MES_Error_Train,MES_Error_Test)
    return MES_Error_Train,MES_Error_Test


# Task6()

Errors_Train = []
Errors_Test = []
for i in range(10):
    e1,e2 = Task6()
    Errors_Train.append(e1)
    Errors_Test.append(e2)

Plot5(Errors_Train)
Plot5(Errors_Test,False)
