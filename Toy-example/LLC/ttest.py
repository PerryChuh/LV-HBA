import numpy as np
import scipy as sp
import torch
import time
import scipy.linalg
start_time = int(time.time())


def proj(x, A, b=None):
    if A.ndim < 2:
        raise ValueError("Matrix A must be at least 2-dimensional.")
    I = np.eye(x.shape[0])
    # 矩阵乘法
    A1 = A @ A.T
    # 计算矩阵的 Moore-Penrose 伪逆
    A_ = np.linalg.pinv(A1)
    if b is None:
    #     return np.matmul(I-A_ @ A,x)
    # else:
    #     return np.matmul(I-A_ @ A,x)-A_@b
    # 也是矩阵乘法
        return np.matmul(I - A.T @ A_ @ A, x)
    else:
        return np.matmul(I - A.T @ A_ @ A, x) - A.T @ A_ @ b

def F(x, y1, y2):
    result = 0.5*(np.linalg.norm((x-y2)))**2 + 0.5*(np.linalg.norm(y1-e1))**2
    return result
def f(x, y1, y2):
    result = 0.5*(np.linalg.norm(y1))**2 - x.T @ y1 + e1.T @ y2
    return result


def F_x(x, y1, y2):
    result = x-y2
    return result

def F_y(x, y1, y2):
    result = np.concatenate((y1-e1, y2-x), axis=0)
    return result

def f_x(x, y1, y2):
    result = -y1
    return result

def f_y(x, y1, y2):
    result = np.concatenate((y1-x, e1), axis=0)
    return result

def g_x(x, y1, y2):
    result = e1
    return result

def g_y(x, y1, y2):
    result = e2
    return result

def g(x, y1, y2):
    result = e1.T@x+e1.T@y1+e1.T@y2
    return result


def fun(n):
    x = 1*np.ones((100, 1))
    # x = x_opt
    y1 = 1*np.ones((100, 1))
    # y1 = y1_opt
    y2 = 1*np.ones((100, 1))
    # y2 = y2_opt
    theta1 = np.ones((100, 1))
    theta2 = np.ones((100, 1))
    lambda1 = 1
    lambda2 = 1
    z1 = 1
    z2 = 1
    for k in range(n):
        x_k = x
        y1_k = y1
        y2_k = y2
        z1_k = z1
        z2_k = z2
        theta1_k = theta1
        theta2_k = theta2
        lambda1_k = lambda1
        lambda2_k = lambda2



        # dk_0 = f_y(x, theta1, theta2) + lambda1 * g_y(x, theta1, theta2) + (theta - y) / gamma1
        dk_01 = theta1_k - x_k + (lambda1_k - lambda2_k) * e1 + (theta1_k - y1_k) / gamma1
        dk_02 = e1 + (lambda1_k - lambda2_k) * e1 + (theta2_k - y2_k) / gamma1

        # dk_1 = - g(x, theta1, theta2) + (lambda1 - z) / gamma2
        dk_11 = - e1.T @ x_k - e1.T @ theta1_k - e1.T @ theta2_k + (lambda1_k - z1_k) / gamma2
        dk_12 = e1.T @ x_k + e1.T @ theta1_k + e1.T @ theta1_k + (lambda2_k - z2_k) / gamma2
        # update theta, lambda

        theta1_k1 = theta1_k - (eta * dk_01)
        theta2_k1 = theta2_k - (eta * dk_02)
        lambda1_k1 = lambda1_k - (eta * dk_11)
        lambda2_k1 = lambda2_k - (eta * dk_12)

        # projection onto box: [0,r]
        lambda1_k1 = min(max(lambda1_k1, 0), r)
        lambda2_k1 = min(max(lambda2_k1, 0), r)
        # lambda1 = 0 if lambda1 < 0 else (r if lambda1 > r else lambda1)
        theta1 = theta1_k1
        theta2 = theta2_k1
        lambda1 = lambda1_k1
        lambda2 = lambda2_k1

        # calc d1 d2 d3, and update x, y, z respectively
        ck = (k+1)**0.3
        # dkx = F_x(x, y1, y2) / ck + f_x(x, y1, y2) - f_x(x, theta1, theta2) - lambda1 * g_x(x, theta1, theta2)
        dkx = (1/ck) * (x_k - y2_k) - y1_k + theta1_k1 - (lambda1_k1-lambda2_k1) * e1
        # dky = F_y(x, y1, y2) / ck + f_y(x, y1, y2) - (y - theta) / gamma1
        dky_1 = (y1_k - e1)/ck + y1_k - x_k - (y1_k - theta1_k1)/gamma1
        dky_2 = (y2_k - x_k)/ck + e1 - (y2_k-theta2_k)/gamma1
        # dky = np.concatenate((dky_1, dky_2),axis=0)
        dkz_1 = -(lambda1_k1 - z1_k) / gamma2
        dkz_2 = -(lambda2_k1 - z2_k) / gamma2

        x_k1 = x_k - alpha * dkx
        y1_k1 = y1_k - alpha * dky_1
        y2_k1 = y2_k - alpha * dky_2

        '''
        # 更新x，y
        x = x - (e1 @ e1.T @ x) / 300
        y1 = y1 - (e1 @ e1.T @ y1) / 300
        y2 = y2 - (e1 @ e1.T @ y2) / 300
        print(g(x, y1, y2))
        '''

        # projection onto C:={[x;y1;y2]^T e3=0}
        # w = w - ((e3.T@w)/(e3.T@e3))@e3
        w = np.concatenate((x_k1, y1_k1, y2_k1), axis=0)
        scale = np.dot(e3.T, np.concatenate((x_k1, y1_k1, y2_k1)))
        w = w - scale / (300) * e3


        x_k1 = w[:100]
        y_k1 = w[100:]
        # update y1, y2
        y1_k1 = y_k1[:100]
        y2_k1 = y_k1[100:]
        # print(round(float(g(x_k1, y1_k1, y2_k1)), 5))

        z1_k1 = z1_k - (beta * dkz_1)
        # projection onto box: [0,r]
        z1_k1 = min(max(z1_k1, 0), r)
        # z = 0 if z < 0 else (r if z > r else z)

        z2_k1 = z2_k - (beta * dkz_2)
        # projection onto box: [0,r]
        z2_k1 = min(max(z2_k1, 0), r)
        # z = 0 if z < 0 else (r if z > r else z)


        x = x_k1
        y1 = y1_k1
        y2 = y2_k1
        z1 = z1_k1
        z2 = z2_k1
        y_k1 = np.concatenate((y1_k1, y2_k1), axis=0)
        arrF[0, k] = F(x_k1, y1_k1, y2_k1)
        arrX[0, k] = np.linalg.norm(x_k1-x_opt, 2) / np.linalg.norm(x_opt, 2)
        arrY[0, k] = np.linalg.norm(y_k1-y_opt, 2) / np.linalg.norm(y_opt, 2)


if __name__ == '__main__':
    alpha = 0.002
    beta = 0.002
    eta = 0.03
    gamma1 = 10
    gamma2 = 10
    r = 1
    e1 = np.ones((100, 1))
    e2 = np.ones((200, 1))
    e3 = np.ones((300, 1))
    E = np.eye(100)
    x_opt = -0.3 * e1
    y1_opt = 0.7 * e1
    y2_opt = -0.4 * e1
    y_opt = np.concatenate((y1_opt, y2_opt), axis=0)
    print(F(x_opt, y1_opt, y2_opt))
    iteration = 5000
    arrF = np.zeros((1, iteration))
    arrX = np.zeros((1, iteration))
    arrY = np.zeros((1, iteration))
    fun(arrF.shape[1])

    tmp_time = int(time.time())
    import matplotlib.pyplot as plt

    plt.plot(arrF.flatten())
    plt.xlabel('iteration')
    plt.ylabel('F(xk,yk)')
    plt.title('F value')
    plt.grid(True)
    plt.show()

    plt.plot(arrX.flatten())
    plt.xlabel('')
    plt.ylabel('X')
    plt.title('X convergence')
    plt.grid(True)
    plt.show()

    plt.plot(arrY.flatten())
    plt.xlabel('Index')
    plt.ylabel('Y')
    plt.title('Y convergence')
    plt.grid(True)
    plt.show()



    '''
    x_values = range(len(res1))
    # 画出图像
    plt.plot(x_values, res1)
    plt.xlabel('Iterations')  # 设置 x 轴标签
    plt.ylabel('Log of Norm Ratio')  # 设置 y 轴标签
    plt.title('x_Convergence Plot')  # 设置标题
    plt.grid(True)  # 添加网格线
    plt.show()  # 显示图像

    y_values = range(len(res2))
    # 画出图像
    plt.plot(y_values, res2)
    plt.xlabel('Iterations')  # 设置 x 轴标签
    plt.ylabel('Log of Norm Ratio')  # 设置 y 轴标签
    plt.title('y_Convergence Plot')  # 设置标题
    plt.grid(True)  # 添加网格线
    plt.show()  # 显示图像
    '''