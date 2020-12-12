# 根据sizes,bedroom预测price of house
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

data = pd.read_csv('./ex1data2.txt',names=['sizes', 'bedroom', 'price'])
# print(data.head())

#散点图
# data.plot.scatter('sizes', 'price', c='b', label='sizes', s=30)
# plt.show()
# data.plot.scatter('bedroom', 'price', c='b', label='bedroom', s=30)
# plt.show()
# 数据标准化
data = (data-data.mean())/data.std()
# print(data.head())
data.insert(0, 'ones', 1)
X = data.iloc[:,0:-1]
Y = data.iloc[:,-1]
X = X.values
Y = Y.values
# print(X.shape)  # (47,3)
# print(Y.shape)  # (47,)
Y=Y.reshape(-1,1)
# print(Y.shape)  # (47,1)

# 损失函数
def costFunction(X, Y, theta):
    inner = np.power(X@theta-Y, 2)
    return np.sum((inner)/2*len(X))

theta = np.zeros((3,1))
cost_init = costFunction(X, Y, theta)
# print(cost_init)  # 1081.0000000000005

#梯度下降
def gradientDescent(X, Y, theta, alpha,iters):
    costs = []
    for i in range(iters):
        theta = theta - alpha/len(X)* (X.T@(X@theta-Y))
        # theta = theta - (alpha * X.T @ (X @ theta - Y))/len(X)
        cost = costFunction(X, Y, theta)
        costs .append(cost)
        # if i%100 == 0:
        #     print(cost)
    return theta, costs

# 不同学习率比较
al_alpha = [0.003, 0.03, 0.02, 0.001, 0.01]
iters = 2000
fig, ax = plt.subplots()

for alpha in al_alpha:
    theta_final, costs = gradientDescent(X, Y, theta, alpha, iters)
    ax.plot(np.arange(iters), costs, label=alpha)
    ax.legend()
ax.set(xlabel='number of iters', ylabel='costs', title='cost vs iters')
# plt.show()


x1 = np.linspace(X[:,1].min(), X[:,1].max(), 100)
x2 = np.linspace(X[:,2].min(), X[:,2].max(), 100)
x1, x2 = np.meshgrid(x1, x2)      # 生成网格点矩阵将两个一维数组变成二维矩阵
print(theta_final.shape)  # (3,1)
y = theta_final[0,0] + theta_final[1,0]*x1 + theta_final[2,0]*x2

fig = plt.figure()
Ax = Axes3D(fig)  # 将图像转化为3D
Ax.plot_surface(x1, x2, y, rstride=1, cstride=1, cmap=cm.viridis, label='prediction')  #  rstride行跨度，cstride列跨度  cmap颜色映射表

Ax.scatter(X[:100,1],X[:100,2],Y[:100,0],c='y')
# Ax.scatter(X[100:250,1],X[100:250,2],Y[100:250,0],c='r')
# Ax.scatter(X[250:,1],X[250:,2],Y[250:,0],c='b')
Ax.set_zlabel('price')  # 坐标轴
Ax.set_ylabel('bedroom')
Ax.set_xlabel('feet')
# plt.show()
# 梯度下降过程可视化
fig, bx = plt.subplots(figsize=(8, 6))
bx.plot(np.arange(iters), costs, 'r')
# 设置坐标轴
bx.set_xlabel('Iterations')
bx.set_ylabel('Cost')
# 设置标题
bx.set_title('Error vs. Training Epoch')
plt.show()