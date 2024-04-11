import numpy as np
import time

e1 = np.ones((100, 1))
e2 = np.ones((200, 1))
e3 = np.ones((300, 1))
E = np.eye(100)
O = np.zeros((100, 100))
A = np.concatenate((E, O), axis=1)
B = np.concatenate((O, E), axis=1)

def F(x, y1, y2):
    result = 0.5 * np.linalg.norm(x - y2)**2 + 0.5 * np.linalg.norm(y1 - e1)**2
    return result
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

N = 5000
arrF = np.zeros(N)
def fun(n, alpha=0.005, beta=0.002, eta=0.03, _lambda=1, gamma1=10, gamma2=10, u=1, seed=1):
    x_opt = -0.3 * e1
    y1_opt = 0.7 * e1
    y2_opt = -0.4 * e1
    y_opt = np.concatenate((y1_opt, y2_opt), axis=0)
    x = 10 * np.ones((100, 1))
    y1 = 10 * np.ones((100, 1))
    y2 = 10 * np.ones((100, 1))
    y = np.concatenate((y1, y2), axis=0)
    theta = np.ones((200, 1))
    z = 1
    res1 = []
    res2 = []
    for k in range(n):
        ck = (k + 1) ** 0.3
        d4_0 = f_y(x, theta) + _lambda * g_y(x, theta) + (theta - y) / gamma1
        d4_1 = - g(x, theta) + (_lambda - z) / gamma2

        # update theta, lambda
        theta = theta - (eta * d4_0)
        _lambda = _lambda - (eta * d4_1)
        _lambda = -u if _lambda < -u else (u if _lambda > u else _lambda)

        # calc d1 d2 d3, and update x, y, z respectively
        d1 = F_x(x, y) / ck + f_x(x, y) - f_x(x, theta) - _lambda * g_x(x, theta)
        d2 = F_y(x, y) / ck + f_y(x, y) - (y - theta) / gamma1
        x = x - alpha * d1
        y = y - alpha * d2
        w = np.concatenate((x, y), axis=0)
        w = w - (e3 @ e3.T) @ w / 300

        x = w[:100]
        y = w[100:]
        d3 = - (_lambda - z) / gamma2
        t_z = z - (beta * d3)
        z = 0 if t_z < 0 else (u if t_z > u else t_z)
        y1 = w[100:200]
        y2 = w[200:]
        res1.append(np.linalg.norm(x - x_opt, 2) / np.linalg.norm(x_opt, 2))
        res2.append(np.linalg.norm(y - y_opt, 2) / np.linalg.norm(y_opt, 2))
        arrF[k] = F(x, y1, y2)
    return res1, res2

if __name__ == '__main__':

    res1, res2 = fun(N)
    tmp_time = int(time.time())
    import matplotlib.pyplot as plt

    plt.subplot(2, 1, 1)
    plt.plot(res1, label='X convergenceX Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('||x^k-x||/||x||')
    plt.title('X Convergence Convergence')
    plt.axhline(0, linestyle='--', color='red')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(res2, label='Y Sequence Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('||y^k-y||/||y||')
    plt.title('Y Sequence Convergence')
    plt.axhline(0, linestyle='--', color='red')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 绘制目标函数值随迭代次数的变化图
    plt.figure(figsize=(8, 6))
    plt.plot(arrF, label='F(x^k,y^k)')
    plt.xlabel('Iteration')
    plt.ylabel('F(x^k,y^k)')
    plt.title('Objective Function Value Convergence')
    plt.axhline(5, linestyle='--', color='red')
    plt.legend()
    plt.show()

    print(arrF[N - 1])