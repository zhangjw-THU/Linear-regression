# 多项式回归模型
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
torch.manual_seed(2018)
# 定义一个多变量函数

w_target = np.array([0.5, 3, 2.4])  # 定义参数
b_target = np.array([0.9])  # 定义参数
f_des = 'y = {:.2f} + {:.2f} * x + {:.2f} * x^2 + {:.2f} * x^3'.format(
    b_target[0], w_target[0], w_target[1], w_target[2])  # 打印出函数的式子
print(f_des)
MU = 0
SIGMA = 5
# 画出这个函数的曲线
x_sample = np.arange(-3, 3.1, 0.1)
y_sample = b_target[0] + w_target[0] * x_sample + w_target[1] * x_sample ** 2 + w_target[2] * x_sample ** 3

y_samples_noise = [random.gauss(MU,SIGMA) for i in range(len(y_sample))]
y_samples_noise = np.array(y_samples_noise)
y_sample = y_sample+y_samples_noise
x_train = np.stack([x_sample ** i for i in range(1, 4)], axis=1)
print(x_train.dtype)
x_train = torch.from_numpy(x_train).float()  # 转换成 float tensor
y_train = torch.from_numpy(y_sample).float().unsqueeze(1)  # 转化成 float tensor

# 定义参数和模型
w = Variable(torch.randn(3, 1), requires_grad=True)
b = Variable(torch.zeros(1), requires_grad=True)

# 将 x 和 y 转换成 Variable
x_train = Variable(x_train)
y_train = Variable(y_train)


def multi_linear(x):
    return torch.mm(x, w) + b


# 计算误差
def get_loss(y_, y):
    return torch.mean((y_ - y_train) ** 2)


# 进行 100 次参数更新
for e in range(100):
    y_pred = multi_linear(x_train)
    loss = get_loss(y_pred, y_train)
    if e != 0:
        w.grad.data.zero_()
        b.grad.data.zero_()
    loss.backward()

    # 更新参数
    w.data = w.data - 0.001 * w.grad.data
    b.data = b.data - 0.001 * b.grad.data
    if (e + 1) % 20 == 0:
        print('epoch {}, Loss: {:.5f}'.format(e + 1, float(loss)))
# 画出更新之后的结果
y_pred = multi_linear(x_train)
plt.plot(x_train.data.numpy()[:, 0], y_pred.data.numpy(), label='fitting curve', color='r')
plt.plot(x_train.data.numpy()[:, 0], y_sample, label='real curve', color='b')
plt.legend()
plt.show()
plt.close()