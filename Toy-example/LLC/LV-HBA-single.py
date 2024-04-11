import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

# 定义目标函数和更新函数
def F(x, y1, y2):
    result = 0.5 * norm(x - y2)**2 + 0.5 * norm(y1 - np.ones_like(y1))**2
    return result

def dtheta1(theta1, x, y1, lambda1, gamma1):
    result = theta1 - x + lambda1 * np.ones_like(theta1) + (1 / gamma1) * (theta1 - y1)
    return result

def dtheta2(theta2, y2, lambda1, gamma1):
    result = np.ones_like(theta2) + lambda1 * np.ones_like(theta2) + (1 / gamma1) * (theta2 - y2)
    return result

def dlambda1(x, theta1, theta2, gamma2, lambda1, z):
    result = -np.dot(np.ones_like(x).T, x) - np.dot(np.ones_like(theta1).T, theta1) - np.dot(np.ones_like(theta2).T, theta2) + (1 / gamma2) * (lambda1 - z)
    return result

def dx(c, x, y1, y2, theta1, lambda1):
    result = (x - y2) / c - y1 + theta1 - lambda1 * np.ones_like(theta1)
    return result

def dy1(c, x, y1, gamma1, theta1):
    result = (y1 - np.ones_like(y1)) / c + y1 - x - (y1 - theta1) / gamma1
    return result

def dy2(c, x, y2, gamma1, theta2):
    result = (y2 - x) / c + np.ones_like(y2) - (y2 - theta2) / gamma1
    return result

def dz(lambda1, gamma2, z):
    result = -(lambda1 - z) / gamma2
    return result

# 初始化参数和变量
n = 100
one_n = np.ones((n, 1))
one_3n = np.ones((3 * n, 1))

xstar = -0.3 * one_n
y1star = 0.7 * one_n
y2star = -0.4 * one_n

N = 10000000
eta = 0.03
alpha = 0.002
beta = 0.002
x = 2 * one_n
y1 = 2 * one_n
y2 = 2 * one_n

theta1 = one_n
theta2 = one_n
lambda1 = 10
z = 10

gamma1 = 10
gamma2 = 10
r = 1

convergenceX = np.zeros(N + 1)
convergenceY = np.zeros(N + 1)
Value = np.zeros(N + 1)
convergenceY[0] = norm(np.concatenate((y1, y2)) - np.concatenate((y1star, y2star))) / norm(np.concatenate((y1star, y2star)))
convergenceX[0] = norm((x - xstar)) / norm(xstar)
Value[0] = F(x, y1, y2)

for i in range(N):
    c = (i + 1)**0.3
    x_origin = x.copy()
    theta1_origin = theta1.copy()
    theta2_origin = theta2.copy()

    theta1 = theta1 - eta * dtheta1(theta1, x_origin, y1, lambda1, gamma1)
    theta2 = theta2 - eta * dtheta2(theta2, y2, lambda1, gamma1)

    lambda1 = min(max(lambda1 - eta * dlambda1(x_origin, theta1_origin, theta2_origin, gamma2, lambda1, z), 0), r)

    x_m = x_origin - alpha * dx(c, x_origin, y1, y2, theta1, lambda1)
    y1_m = y1 - alpha * dy1(c, x_origin, y1, gamma1, theta1)
    y2_m = y2 - alpha * dy2(c, x_origin, y2, gamma1, theta2)

    atx = np.dot(one_3n.T, np.concatenate((x_m, y1_m, y2_m)))
    if atx < 0:
        atx = 0
    w_m = np.concatenate((x_m, y1_m, y2_m))
    w = w_m - (atx / (3 * n)) * one_3n

    x = w[:n]
    y = w[100:]
    y1 = y[:100]
    y2 = y[100:]

    z = min(max(z - beta * dz(lambda1, gamma2, z), 0), r)

    Value[i + 1] = F(x, y1, y2)
    convergenceX[i + 1] = norm(x - xstar) / norm(xstar)
    convergenceY[i + 1] = norm(np.concatenate((x, y1, y2)) - np.concatenate((xstar, y1star, y2star))) / norm(np.concatenate((xstar, y1star, y2star)))

print("Min Value:", min(Value))
print("Min convergenceX:", min(convergenceX))
print("Min convergenceY:", min(convergenceY))

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(convergenceX, label='X convergenceX Convergence')
plt.xlabel('Iteration')
plt.ylabel('||x^k-x||/||x||')
plt.title('X Convergence Convergence')
plt.axhline(0, linestyle='--', color='red')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(convergenceY, label='Y Sequence Convergence')
plt.xlabel('Iteration')
plt.ylabel('||y^k-y||/||y||')
plt.title('Y Sequence Convergence')
plt.axhline(0, linestyle='--', color='red')
plt.legend()

plt.tight_layout()
plt.show()

# 绘制目标函数值随迭代次数的变化图
plt.figure(figsize=(8, 6))
plt.plot(Value, label='F(x^k,y^k)')
plt.xlabel('Iteration')
plt.ylabel('F(x^k,y^k)')
plt.title('Objective Function Value Convergence')
plt.axhline(5, linestyle='--', color='red')
plt.legend()
plt.show()