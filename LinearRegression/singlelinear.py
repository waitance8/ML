# 根据population，预测profit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./ex1data1.txt', names=['population', 'profit'])
# print(data)

data.plot.scatter('population', 'profit', c='b', label='population', s=30)
# plt.show()
# 处理数据
data.insert(0, 'ones', 1)
X = data.iloc[:, 0:-1]
Y = data.iloc[:, -1]
# print(X)
# print(Y)
# 把数据转换成数组形式
X = X.values
Y = Y.values
# print(X)
# print(Y)
# print(X.shape)  # (97, 2)
# print(Y.shape)
Y = Y.reshape(-1, 1)  # (97, 1)
# print(Y.shape)

#损失函数
def costfunction(X, Y, theta):
    inner = np.power((X@theta-Y), 2)
    return np.sum((inner)/(2*len(X)))

theta = np.zeros((2,1))
cost_init = costfunction(X, Y, theta)
# print(cost_init)  # 32.072733877455676

#梯度下降
def gradientDescent(X, Y, theta, alpha):
    costs = []
    for i in range(iters):
        theta = theta - alpha/len(X) * (X.T @ (X @ theta - Y))
        cost = costfunction(X, Y, theta)
        costs.append(cost)
    return theta, costs

alpha = 0.02
iters = 2000
fig, ax = plt.subplots()
# fig表示整个图，ax表示实例化的对象
theta_final, costs = gradientDescent(X, Y, theta, alpha)
# 可视化损失函数
ax.plot(np.arange(iters), costs)
ax.set(xlabel='number of iters', ylabel='costs', title='cost vs iters' )
plt.show()

# 拟合曲线 划分自变量population
x = np.linspace(Y.min(), Y.max(), 100)
y_pred = theta_final[0,0] + theta_final[1,0] * x
fig, ax = plt.subplots()
#真实样本
ax.scatter(X[:,1], Y, label='train data')
#预测结果
ax.plot(x,y_pred,label='predict')
ax.legend()
plt.show()
