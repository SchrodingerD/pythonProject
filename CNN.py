# import torch
# from torch_geometric.data import Data
# edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long)
# x = torch.tensor([[-1],[0],[1]],dtype=torch.float)
# data = Data(x=x,edge_index=edge_index)
# print(data)

# 用梯度下降法求解函数z=x^2+y^2的最小值
import matplotlib.pyplot as plt
import numpy as np
import math
def solution1(u=0.1):
    xdata = []
    ydata = []
    tdata = []
    print('当前学习率：{}'.format(u))
    x, y, u = 3, 2, u
    for t in range(20):
        z = x ** 2 + y ** 2
        orgx, orgy = x, y
        xdata.append(orgx)
        ydata.append(orgy)
        tdata.append(t)
        xt, yt = x * 2, y * 2
        xz, yz = -u * xt, -u * yt
        x, y = x + xz, y + yz
        print('loop:{},当前的坐标位置:({:+.4f},{:+.4f}),梯度值:({:+.4f},{:+.4f}),步长:({:+.4f},{:+.4f}),函数值:({:+.4f})'.format(t, orgx, orgy, xt, yt, xz, yz, z))
    return xdata, ydata, tdata

def drawtrack(xdata, ydata, tdata):
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(xdata, ydata, "ob")
    for i in range(0, len(xdata)):
        ax.text(xdata[i], ydata[i] + 0.1, tdata[i])
    # ax.spines['right'].set_position(('data', 3.0))
    # ax.spines['top'].set_position(('data', 2.0))
    # ax.spines['bottom'].set_position(('data', 0))
    # ax.spines['left'].set_position(('data', 0))
    plt.title("求梯度")
    plt.show()

if __name__ == '__main__':
    xdata, ydata, tdata = solution1(0.4)
    drawtrack(xdata, ydata, tdata)
    # xdata, ydata, tdata = solution1(0.1)
    # drawtrack(xdata, ydata, tdata)
    # xdata, ydata, tdata = solution1(0.01)
    # drawtrack(xdata, ydata, tdata)