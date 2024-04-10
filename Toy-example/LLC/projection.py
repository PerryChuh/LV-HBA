import numpy as np
import scipy as sp
import torch
import time
import scipy.linalg

start_time = int(time.time())


# A_ = np.linalg.pinv(A)

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


def proj1(x, b=None):
    if b is None:
        return x - sum(x) / len(x)
    else:
        return x - (sum(x) + b) / len(x)


e1 = np.ones((100, 1))
e2 = np.ones((200, 1))
e3 = np.ones((300, 1))
E = np.eye(100)
O = np.zeros((100, 100))
A = np.concatenate((E, O), axis=1)
B = np.concatenate((O, E), axis=1)


def F(x, y):
    return 0.5 * ((x - B @ y).T @ (x - B @ y)) + 0.5 * ((A @ y - e1).T @ (A @ y - e1))


def F_x(x, y):
    return x - B @ y


def F_y(x, y):
    return (A.T @ (A @ y - e1)) - B.T @ (x - B @ y)


def f_x(x, y):
    return -A @ y


def f_y(x, y):
    return A.T @ A @ y - A.T @ x + B.T @ e1


def g_x(x, y):
    return e1


def g_y(x, y):
    return e2


def g(x, y):
    return e1.T @ x + e2.T @ y


def fun(n, alpha=0.008, beta=0.02, eta=0.01, _lambda=1, gamma1=10, gamma2=10, u=100, seed=1):
    rng = np.random.default_rng(seed)
    x_opt = -0.3 * e1
    y1_opt = 0.7 * e1
    y2_opt = -0.4 * e1
    y_opt = np.concatenate((y1_opt, y2_opt), axis=0)
    # x = 10*np.ones((100,1))
    x = x_opt
    # y1 = 10*np.ones((100,1))
    y1 = y1_opt
    # y2 = 10*np.ones((100,1))
    y2 = y2_opt
    print(F(x_opt, y_opt))
    y = np.concatenate((y1, y2), axis=0)
    # x = np.full((10,1), 1)
    # y = np.full((20,1), 1)
    # x = proj1(x, sum(y))
    # y = proj1(y, sum(x))
    theta = np.ones((200, 1))
    # theta = proj1(theta, sum(x))

    # Z = np.arange(0, u+1)
    z = 10
    Ix = np.eye(x.shape[0])
    # fix y_opt and delete this
    C = Ix

    res1 = []
    res2 = []
    time_computation = []

    algorithm_start_time = time.time()
    w = np.concatenate((x, y), axis=0)
    list_k_time = np.array([0])
    for k in range(n):
        # clac d4
        # ck= 10000000*(k+1)*0.3
        # ck= 0.025*(k+1)**0.3
        ck = (k + 1) ** 0.3
        d4_0 = f_y(x, theta) + _lambda * g_y(x, theta) + (theta - y) / gamma1
        d4_1 = - g(x, theta) + (_lambda - z) / gamma2

        # update theta, lambda
        theta = theta - (eta * d4_0)
        y11_opt = x - e1
        theta1 = theta[:100]
        _lambda = _lambda - (eta * d4_1)
        # _lambda = -u if _lambda < -u else (u if _lambda > u else _lambda)
        # 投影到[0,1000] box上，u=1000
        _lambda = 0 if _lambda < 0 else (u if _lambda > u else _lambda)

        # proj

        # calc d1 d2 d3, and update x, y, z respectively

        d1 = F_x(x, y) / ck + f_x(x, y) - f_x(x, theta) - _lambda * g_x(x, theta)
        d2 = F_y(x, y) / ck + f_y(x, y) - (y - theta) / gamma1
        x = x - alpha * d1
        y = y - alpha * d2
        w = np.concatenate((x, y), axis=0)
        w = w - sum(w) / len(w)
        # 更新x，y
        # w = proj(w, e3.T)
        # w = w - ((e3.T@w)/(e3.T@e3))*e3
        w = w - (e3.T @ w) * e3
        # print(sum(w), sum((x - w[200:])**2) + .5 * sum((w[100:200] - 1)**2))
        x = w[:100]
        y = w[100:]
        d3 = - (_lambda - z) / gamma2
        t_z = z - (beta * d3)
        # z = -u if t_z < -u else (u if t_z > u else t_z)
        z = 0 if t_z < 0 else (u if t_z > u else t_z)
        y1 = w[100:200]
        y2 = w[200:]
        # y_opt = C@((Ix-torch.linalg.pinv(A@C)@(A@C))@x-torch.linalg.pinv(A@C)@(H@x))
        res1.append(np.log(np.linalg.norm(x - x_opt, 2) / np.linalg.norm(x_opt, 2)))
        # res2.append(np.log(np.linalg.norm(y-y_opt,2) / np.linalg.norm(y_opt,2)))
        # res.append(np.log(np.linalg.norm(x-x_opt,2)))
        # res.append(np.log((np.square(np.linalg.norm(theta1-y11_opt,2))))) # norm

        # res.append(F(x,y).detach().numpy())
        time_computation.append(time.time() - algorithm_start_time)
        y_gap1 = (np.linalg.norm(x - x_opt, 2) / np.linalg.norm(x_opt, 2))
        arrF[0, k] = F(x, y)
    return res1, res2, time_computation


if __name__ == '__main__':
    from pathlib import Path
    import os

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory

    # while True:
    #     A = np.random.rand(90,100)
    #     rank = np.linalg.matrix_rank(A)
    #     if rank == A.shape[0] and rank < A.shape[1]:
    #         print('A rank is', rank)
    #         np.save(os.path.join(str(ROOT), 'A.npy'), A)
    #         break

    # while True:
    #     B = np.random.rand(45,100)
    #     Z = np.zeros(B.shape)
    #     B = np.concatenate([B,Z],axis=0)
    #     rank = np.linalg.matrix_rank(B)
    #     if rank < B.shape[0]:
    #         print('B rank is', rank)
    #         np.save(os.path.join(str(ROOT), 'B.npy'), B)
    #         break

    # h = np.random.rand(90,100)
    # np.save(os.path.join(str(ROOT), 'h.npy'), h)
    # res, tc = fun(100, alpha=0.1, beta=0.1, eta=0.1, _lambda=0.1, gamma1=1.2, gamma2=1.2, u=100000, seed=1)
    arrF = np.zeros((1, 15000))
    arrX = np.zeros((1, 15000))
    arrY = np.zeros((1, 15000))
    res1, res2, tc = fun(15000)

    tmp_time = int(time.time())
    import matplotlib.pyplot as plt

    ax = plt.gca()
    plt.plot(tc, res1)
    # plt.plot(tc,res2)
    plt.show()

    plt.plot(arrF.flatten())
    plt.xlabel('Index')
    plt.ylabel('F(x, y)')
    plt.title('F(x, y) Variation')
    plt.grid(True)
    plt.show()

    x_values = range(len(res1))
    # 画出图像
    plt.plot(x_values, res1)
    plt.xlabel('Iterations')  # 设置 x 轴标签
    plt.ylabel('Log of Norm Ratio')  # 设置 y 轴标签
    plt.title('Convergence Plot')  # 设置标题
    plt.grid(True)  # 添加网格线
    plt.show()  # 显示图像

    y_values = range(len(res1))
    # 画出图像
    plt.plot(x_values, res1)
    plt.xlabel('Iterations')  # 设置 x 轴标签
    plt.ylabel('Log of Norm Ratio')  # 设置 y 轴标签
    plt.title('Convergence Plot')  # 设置标题
    plt.grid(True)  # 添加网格线
    plt.show()  # 显示图像