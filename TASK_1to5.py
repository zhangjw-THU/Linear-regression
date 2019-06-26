# -*- coding: utf-8 -*-
# 张嘉玮
# 20190502
# 自61，2016011528

# K = 273.15 + C
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.autograd
import torch.nn.functional as F
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.patches as mpatches

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

def Plot1(Ts,Rs):
    """
    任务1中的画图
    :param Ts: 温度：1*n
    :param Rs: 电阻：1*n
    :return:
    """

    plt.title('Resistance--Temperature')
    plt.plot(Ts, Rs, label='R_KOhm', linewidth=2, color='orange', marker='o',
             markerfacecolor='purple', markersize=4)
    plt.xlabel('T(C)')
    plt.ylabel('R(Kohm)')
    plt.legend(loc='upper right')
    plt.show()

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

def Plot2_for_3(Ts,Rs_real,Rs_poly_60_times):
    for i in range(6):
        plt.title('Resistance--Temperature:n = ' + str(i + 1))
        for j in range(10):
            plt.plot(Ts,Rs_poly_60_times[i+6*j],label='R_Poly', linewidth=2, color='lime', marker='o',
             markerfacecolor='purple', markersize=4)
        plt.plot(Ts, Rs_real, label='R_Real', linewidth=3, color='orange', marker='o',
                 markerfacecolor='purple', markersize=4)
        plt.xlabel('T(C)')
        plt.ylabel('R(Kohm)')
        # plt.legend(loc='upper right')
        plt.show()

def Plot3(MSE_Errors,Train = True):
    """
    任务2
    :param MSE_Errors:误差
    :param Train:训练or测试
    :return:
    """
    N = [i+1 for i in range(len(MSE_Errors))]

    if Train:
        plt.title('MES_Error--N Of Train')
        colour = 'lawngreen'
    else:
        plt.title('MES_Error--N Of Test')
        colour = 'red'
    plt.plot(N,MSE_Errors, label='MSE_Errors', linewidth=2, color=colour, marker='o',
             markerfacecolor='purple', markersize=4)
    plt.xlabel('N')
    plt.ylabel('Error')
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

def Plot6(Errors,Noise,Train=True):
    """
    添加不同的误差的实验结果
    :param Errors:
    :param Noise:
    :param Train:
    :return:
    """
    n = [i + 1 for i in range(len(Errors[0]))]
    scatterColors = ['orange', 'lawngreen', 'purple', 'brown', 'orangered', 'hotpink', 'salmon', 'yellow', 'red']
    if Train:
        plt.title('MES_Error--N Of Train')
    else:
        plt.title('MES_Error--N Of Test')

    for i,noise in enumerate(Noise):
        color = scatterColors[i % len(scatterColors)]
        plt.plot(n, Errors[i], label='Noise_' + str(noise), linewidth=2, color=color, marker='o',
                 markerfacecolor='purple', markersize=4)

    plt.xlabel('N')
    plt.ylabel('Error')
    plt.legend(loc='upper right')
    plt.show()


def Plot7(Errors,Noises,Nums = 10,ns = 6,Train=True):
    """
    改变噪声强度，进行的10次实验
    :param Errors:
    :param Noises:
    :param Nums:
    :param ns:
    :param Train:
    :return:
    """
    if Train:
        plt.title('MES_Error--N Of Train')
    else:
        plt.title('MES_Error--N Of Test')
    Num = [i + 1 for i in range(Nums)]
    n = [i + 1 for i in range(ns)]
    scatterColors = ['orange', 'lawngreen', 'purple']#, 'brown', 'orangered', 'hotpink', 'salmon', 'yellow', 'red']
    labels = [str(Noises[0]), str(Noises[1]), str(Noises[2])]
    patches = [mpatches.Patch(color=scatterColors[i], label="{:s}".format(labels[i])) for i in range(len(scatterColors))]

    for j,Error in enumerate(Errors):
        for i in Num:
            color = scatterColors[j% len(scatterColors)]
            plt.plot(n, Error[i - 1], label='Noise_' + str(Noises[j]), linewidth=2, color=color, marker='o',
                     markerfacecolor='purple', markersize=4)

    plt.xlabel('N')
    plt.ylabel('Error')
    plt.legend(handles=patches, bbox_to_anchor=(0.75,1), ncol=3)
    plt.show()


def Plot8(Errors,Nums,Train = True):
    """

    :param Errors:
    :param Nums:
    :param Train:
    :return:
    """
    n = [i + 1 for i in range(len(Errors[0]))]
    scatterColors = ['orange', 'lawngreen', 'purple', 'brown', 'orangered', 'hotpink', 'salmon', 'yellow', 'red']
    if Train:
        plt.title('MES_Error--n(poly) of Train')
    else:
        plt.title('MES_Error--n(poly) of Test')
    for i,num in enumerate(Nums):
        color = scatterColors[i % len(scatterColors)]
        plt.plot(n, Errors[i], label= 'Train_Num:'+ str(num), linewidth=2, color=color, marker='o',
                 markerfacecolor='purple', markersize=4)

    plt.xlabel('N')
    plt.ylabel('Error')
    plt.legend(loc='upper right')
    plt.show()

def Plot9(Errors,Noises,Nums = 10,ns = 6,Train=True):
    """

    :param Errors:
    :param Noises:
    :param Nums:
    :param ns:
    :param Train:
    :return:
    """
    if Train:
        plt.title('MES_Error--N Of Train')
    else:
        plt.title('MES_Error--N Of Test')
    Num = [i + 1 for i in range(Nums)]
    n = [i + 1 for i in range(ns)]
    scatterColors = ['orange', 'lawngreen', 'purple', 'brown', 'orangered']#, 'hotpink', 'salmon', 'yellow', 'red']
    labels = [str(i) for i in Noises]
    patches = [mpatches.Patch(color=scatterColors[i], label="{:s}".format(labels[i])) for i in range(len(scatterColors))]

    for j,Error in enumerate(Errors):
        for i in Num:
            color = scatterColors[j% len(scatterColors)]
            plt.plot(n, Error[i - 1], label='TrainSet_Num:' + str(Noises[j]), linewidth=2, color=color, marker='o',
                     markerfacecolor='purple', markersize=4)

    plt.xlabel('N')
    plt.ylabel('Error')
    plt.legend(handles=patches, bbox_to_anchor=(0.75,1), ncol=3)
    plt.show()

def ModelPrint(ploy):

    n = len(ploy)
    Model = 'R(t) = '+str(ploy[0])+'t**'+str(n-1)
    for i,modulus in enumerate(ploy[1:]):
        if modulus>=0:
            Model = Model + ' + '+ str(modulus)+'t**'+str(n-2-i)
        else:
            Model = Model + ' - '+ str(abs(modulus)) + 't**' + str(n - 2 - i)
    print(Model)

def LeastSquare(Ts_C,Rs_KOhm,n):
    """
    最小二乘拟合
    :param Ts_C:
    :param Rs_KOhm:
    :param n:
    :return: 拟合所得系数
    """
    return np.polyfit(Ts_C,Rs_KOhm,deg=n)

def PolyVal(poly,Ts_C):
    """
    求拟合结果
    :param poly:
    :param Ts_C:
    :return:
    """
    return np.polyval(poly,Ts_C)

def MSE(Rs_KOhm_reality,Rs_KOhm_Poly):
    """
    求两组数据的均方差
    :param Rs_KOhm_reality:
    :param Rs_KOhm_Poly:
    :return:
    """
    return sum([(R_real-R_poly)**2 for R_real,R_poly in zip(Rs_KOhm_reality,Rs_KOhm_Poly)])/len(Rs_KOhm_reality)

"""
任务1：以2C作为间隔（步长），画出该种热敏电阻在温度范围为0C~100C间阻值随温度变化的特性曲线
"""
def Task1():
    """
    任务1
    :return:
    """
    Ts_C = [i for i in range(0,101,2)]
    Ts_K = [i + 273.15 for i in Ts_C]

    Rs_KOhm = [Model(T) for T in Ts_K]
    Plot1(Ts_C,Rs_KOhm)

"""
任务2：多项式拟合
"""
# 数据准备
def Task2(MU = 0,SIGMA = 0.5*0.5):
    """
    任务2
    :param MU: 高斯噪声的参数
    :param SIGMA:
    :return:
    """
    Ts_C = [i for i in range(0, 101, 2)]
    Ts_K = [i + 273.15 for i in Ts_C]
    Rs_KOhm = [Model(T) for T in Ts_K]
    Ts_C_20to80 = [i for i in range(20,81,2)]
    Ts_K_20to80 = [i + 273.15 for i in Ts_C_20to80]

    Ts_C_0to18 = [i for i in range(0,19,2)]
    Ts_C_82to100 = [i for i in range(82,101,2)]
    Ts_C_0to18_and_82to100 = Ts_C_0to18 + Ts_C_82to100
    Ts_K_0to18_and_82to100 = [i + 273.15 for i in Ts_C_0to18_and_82to100]

    Rs_KOhm_20to80 = [Model(i) for i in Ts_K_20to80]
    Rs_KOhm_0to18_and_82to100 = [Model(i) for i in Ts_K_0to18_and_82to100]


    Rs_KOhm_20to80_Noised = [ i + random.gauss(MU,SIGMA) for i in Rs_KOhm_20to80]

    Polys = []
    for i in range(1,7,1):
        Polys.append(LeastSquare(Ts_C_20to80,Rs_KOhm_20to80_Noised,i))

    MSE_Error_20to80 = []
    MSE_Error_0to18_and_82to100 = []
    for n,poly in enumerate(Polys):
        Rs_KOhm_20to80_Poly = PolyVal(poly,Ts_C_20to80)
        ModelPrint(poly)
        # Plot2(Ts_C_20to80,Rs_KOhm_20to80_Noised,Rs_KOhm_20to80_Poly,n)

        MSE_Error_20to80.append(MSE(Rs_KOhm_20to80_Noised,Rs_KOhm_20to80_Poly))

        Rs_KOhm_0to18_and_82to100_Poly = PolyVal(poly,Ts_C_0to18_and_82to100)
        MSE_Error_0to18_and_82to100.append(MSE(Rs_KOhm_0to18_and_82to100,Rs_KOhm_0to18_and_82to100_Poly))
        # Plot2(Ts_C_0to18_and_82to100,Rs_KOhm_0to18_and_82to100,Rs_KOhm_0to18_and_82to100_Poly,n)
    # Plot3(MSE_Error_20to80)
    # print(MSE_Error_20to80)
    # Plot3(MSE_Error_0to18_and_82to100,False)
    # print(MSE_Error_0to18_and_82to100)
    # Plot4(MSE_Error_20to80,MSE_Error_0to18_and_82to100)

    return MSE_Error_20to80,MSE_Error_0to18_and_82to100

def Task2_For_Task5(Ts_C_Train ,Ts_C_Test ,Rs_KOhm_Train ,Rs_KOhm_Test ):
    """
    任务5当中的任务2
    :param Ts_C_Train:
    :param Ts_C_Test:
    :param Rs_KOhm_Train:
    :param Rs_KOhm_Test:
    :return:
    """
    Polys = []
    for i in range(1, 7, 1):
        Polys.append(LeastSquare(Ts_C_Train, Rs_KOhm_Train, i))

    MSE_Error_Train = []
    MSE_Error_Test = []
    for poly in Polys:
        Rs_KOhm_Poly = PolyVal(poly, Ts_C_Train)
        # ModelPrint(poly)
        # Plot2(Ts_C_20to80,Rs_KOhm_20to80_Noised,Rs_KOhm_20to80_Poly)
        MSE_Error_Train.append(MSE(Rs_KOhm_Train, Rs_KOhm_Poly))

        Rs_KOhm_Test_Poly = PolyVal(poly, Ts_C_Test)
        MSE_Error_Test.append(MSE(Rs_KOhm_Test, Rs_KOhm_Test_Poly))


    return MSE_Error_Train, MSE_Error_Test

"""
任务三：重复相应内容10次（每次重新添加噪声模拟不同批次实验数据）
"""
def Task3(MU = 0,SIGMA = 0.5):
    """
    任务3
    :param MU:
    :param SIGMA:
    :return:
    """
    Ts_C_0to100 = [i for i in range(0,101,2)]
    Ts_K_0to100 = [i+273.15 for i in Ts_C_0to100]
    Ts_C_20to80 = [i for i in range(20, 81, 2)]
    Ts_K_20to80 = [i + 273.15 for i in Ts_C_20to80]
    Ts_C_0to18 = [i for i in range(0, 19, 2)]
    Ts_C_82to100 = [i for i in range(82, 101, 2)]
    Ts_C_0to18_and_82to100 = Ts_C_0to18 + Ts_C_82to100
    Ts_K_0to18_and_82to100 = [i + 273.15 for i in Ts_C_0to18_and_82to100]
    Rs_KOhm_0to100 = [Model(i) for i in Ts_K_0to100]
    Rs_KOhm_20to80 = [Model(i) for i in Ts_K_20to80]
    Rs_KOhm_0to18_and_82to100 = [Model(i) for i in Ts_K_0to18_and_82to100]

    Errors_Train = np.zeros((10,6))
    Errors_Test = np.zeros((10,6))

    Rs_KOhm_0to100_Poly_10_times = []
    for j in range(10):
        Rs_KOhm_20to80_Noised = [i + random.gauss(MU, SIGMA) for i in Rs_KOhm_20to80]

        Polys = []
        for i in range(1,7,1):
           Polys.append(LeastSquare(Ts_C_20to80, Rs_KOhm_20to80_Noised, i))

        for i,poly in enumerate(Polys):
            Rs_KOhm_20to80_Poly = PolyVal(poly, Ts_C_20to80)
            Errors_Train[j][i] = MSE(Rs_KOhm_20to80_Noised, Rs_KOhm_20to80_Poly)

            Rs_KOhm_0to100_Poly = PolyVal(poly,Ts_C_0to100)
            Rs_KOhm_0to100_Poly_10_times.append(Rs_KOhm_0to100_Poly)

            Rs_KOhm_0to18_and_82to100_Poly = PolyVal(poly, Ts_C_0to18_and_82to100)
            Errors_Test[j][i] = MSE(Rs_KOhm_0to18_and_82to100, Rs_KOhm_0to18_and_82to100_Poly)
    #
    # Plot5(Errors_Train)
    # Plot5(Errors_Test,False)

    Plot2_for_3(Ts_C_0to100,Rs_KOhm_0to100,Rs_KOhm_0to100_Poly_10_times)
    return Errors_Train,Errors_Test


def Task3_For_Task5(Ts_C_Train ,Ts_C_Test ,Rs_KOhm_Train ,Rs_KOhm_Test,MU = 0,SIGMA = 0.5):
    """
    任务5当中的任务2
    :param Ts_C_Train:
    :param Ts_C_Test:
    :param Rs_KOhm_Train:
    :param Rs_KOhm_Test:
    :param MU:
    :param SIGMA:
    :return:
    """
    Errors_Train = np.zeros((10, 6))
    Errors_Test = np.zeros((10, 6))
    for j in range(10):
        Rs_KOhm_20to80_Noised = [i + random.gauss(MU, SIGMA) for i in Rs_KOhm_Train]

        Polys = []
        for i in range(1, 7, 1):
            Polys.append(LeastSquare(Ts_C_Train, Rs_KOhm_20to80_Noised, i))

        for i, poly in enumerate(Polys):
            Rs_KOhm_20to80_Poly = PolyVal(poly, Ts_C_Train)
            Errors_Train[j][i] = MSE(Rs_KOhm_20to80_Noised, Rs_KOhm_20to80_Poly)

            Rs_KOhm_0to18_and_82to100_Poly = PolyVal(poly, Ts_C_Test)
            Errors_Test[j][i] = MSE(Rs_KOhm_Test, Rs_KOhm_0to18_and_82to100_Poly)

    return Errors_Train, Errors_Test
"""
任务4：改变噪声强度（通过改变所加噪声的标准偏差实现），重复2），3）内容，观察并讨论数据中不同噪声强度给拟合（学习）带来的影响；
"""
def Task4():
    """
    任务4
    :return:
    """
    Noise_Task2 = [0.5,0.8,1,1.5,2]
    Noise_Task3 = [0.5,2,5]
    Error_Trains_Task2 = []
    Error_Tests_Task2 = []
    Error_Trains_Task3 = []
    Error_Tests_Task3 = []
    for noise in Noise_Task2:
        e1,e2 = Task2(SIGMA=noise)
        Error_Trains_Task2.append(e1)
        Error_Tests_Task2.append(e2)
    for noise in Noise_Task3:
        e3,e4 = Task3(SIGMA=noise)
        Error_Trains_Task3.append(e3)
        Error_Tests_Task3.append(e4)

    Plot6(Error_Trains_Task2, Noise_Task2)
    Plot6(Error_Tests_Task2,Noise_Task2,False)
    Plot7(Error_Trains_Task3,Noises=Noise_Task3)
    Plot7(Error_Tests_Task3,Noises=Noise_Task3,Train=False)


"""
任务5：将实验数据温度20C~80C范围进行调整（扩大或缩小），重复2），3）内容（需要对训练集及测试集范围进行对应调整），观察并讨论由于采用不同规模训练数据给拟合（学习）结果带来的影响
"""

def Task5():
    """
    任务5
    :return:
    """
    TrainNums = [26,30,35,40,45]
    Ts_C = [i for i in range(0, 101, 2)]
    Ts_K = [i + 273.15 for i in Ts_C]
    Rs_KOhm = [Model(T) for T in Ts_K]
    MU = 0
    SIGMA = 0.5
    Rs_KOhm_Noised = [i + random.gauss(MU, SIGMA) for i in Rs_KOhm]

    Error_Trains = []
    Error_Test = []
    Error_Trains_Task3 = []
    Error_Tests_Task3 = []
    for nums in TrainNums:
        nums_half = int(nums/2)
        Ts_C_Train = Ts_C[25-nums_half:25+nums_half+1]
        Ts_C_Test = Ts_C[0:25-nums_half]+Ts_C[25+nums_half+1:]
        Rs_KOhm_Train = Rs_KOhm_Noised[25-nums_half:25+nums_half+1]
        Rs_KOhm_Test = Rs_KOhm[0:25-nums_half]+Ts_C[25+nums_half+1:]

        e1,e2 = Task2_For_Task5(Ts_C_Train,Ts_C_Test,Rs_KOhm_Train,Rs_KOhm_Test)
        Error_Trains.append(e1)
        Error_Test.append(e2)
        e3,e4=Task3_For_Task5(Ts_C_Train, Ts_C_Test, Rs_KOhm_Train, Rs_KOhm_Test, MU=0, SIGMA=0.5)
        Error_Trains_Task3.append(e3)
        Error_Tests_Task3.append(e4)

    Plot8(Error_Trains,TrainNums)
    Plot8(Error_Test,TrainNums,False)
    Plot9(Error_Trains_Task3, Noises=TrainNums)
    Plot9(Error_Tests_Task3, Noises=TrainNums, Train=False)




# Task1()
# Task2()
# Task3()
Task4()
# Task5()