import numpy as np
import matplotlib.pyplot as plt

# Objective functions
n = 100
e1 = np.ones((n, 1))
e2 = np.ones((2 * n, 1))
e3 = np.ones((3 * n, 1))

def F(x, y1, y2):
    result = 0.5 * np.linalg.norm(x - y2)**2 + 0.5 * np.linalg.norm(y1 - e1)**2
    return result

def f(x, y1, y2):
    result = 0.5 * np.linalg.norm(y1)**2 - np.dot(x.T, y1) + np.dot(e1.T, y2)
    return result

xstar = -0.3 * e1
y1star = 0.7 * e1
y2star = -0.4 * e1

print(F(xstar, y1star, y2star))


# Update
def dtheta1(theta1, x, y1, lambda1, lambda2, gamma1):
    result = theta1 - x + (lambda1 - lambda2) * e1 + (1 / gamma1) * (theta1 - y1)
    return result

def dtheta2(theta2, x, y2, lambda1, lambda2, gamma1):
    result = e1 + (lambda1 - lambda2) * e1 + (1 / gamma1) * (theta2 - y2)
    return result

def dlambda1(x, theta1, theta2, gamma2, lambda1, z1):
    result = -np.dot(e1.T, x) - np.dot(e1.T, theta1) - np.dot(e1.T, theta2) + (1 / gamma2) * (lambda1 - z1)
    return result

def dlambda2(x, theta1, theta2, gamma2, lambda2, z2):
    result = np.dot(e1.T, x) + np.dot(e1.T, theta1) + np.dot(e1.T, theta2) + (1 / gamma2) * (lambda2 - z2)
    return result

def dx(c, x, y1, y2, theta1, lambda1, lambda2):
    result = (x - y2) / c - y1 + theta1 + (lambda2 - lambda1) * e1
    return result

def dy1(c, y1, x, gamma1, theta1):
    result = (y1 - e1) / c + y1 - x - (y1 - theta1) / gamma1
    return result

def dy2(c, y2, x, gamma1, theta2):
    result = (y2 - x) / c + e1 - (y2 - theta2) / gamma1
    return result

def dz1(lambda1, gamma2, z1):
    result = -(lambda1 - z1) / gamma2
    return result

def dz2(lambda2, gamma2, z2):
    result = -(lambda2 - z2) / gamma2
    return result


# Iteration
N = 10000
iter = np.arange(N)
alpha = 0.002
beta = 0.002
eta = 0.03
gamma1 = 10
gamma2 = 10
r = 1



x = 10 * e1
y1 = 10 * e1
y2 = 10 * e1
theta1 = e1
theta2 = e1
lambda1 = 1
lambda2 = 1
z1 = 10
z2 = 10

convergenceX = np.zeros(N)
Fvalue = np.zeros(N)
convergenceY = np.zeros(N)
etw = np.ones(N)
convergenceY[0] = np.linalg.norm(np.concatenate((y1, y2)) - np.concatenate((y1star, y2star))) / np.linalg.norm(np.concatenate((y1star, y2star)))
convergenceX[0] = np.linalg.norm(xstar - x) / np.linalg.norm(xstar)
Fvalue[0] = F(x, y1, y2)

for i in iter:
    c = (i + 1) ** 0.3
    x_origin = x.copy()
    scale = np.dot(e3.T, np.concatenate((x, y1, y2)))
    theta1_origin = theta1.copy()
    theta2_origin = theta2.copy()

    theta1 = theta1 - eta * dtheta1(theta1, x_origin, y1, lambda1, lambda2, gamma1)
    theta2 = theta2 - eta * dtheta2(theta2, x_origin, y2, lambda1, lambda2, gamma1)
    lambda1 = min(max(lambda1 - eta * dlambda1(x_origin, theta1_origin, theta2_origin, gamma2, lambda1, z1), 0), r)
    lambda2 = min(max(lambda2 - eta * dlambda2(x_origin, theta1_origin, theta2_origin, gamma2, lambda2, z2), 0), r)

    x_m = x_origin - alpha * dx(c, x_origin, y1, y2, theta1, lambda1, lambda2)
    y1_m = y1 - alpha * dy1(c, y1, x_origin, gamma1, theta1)
    y2_m = y2 - alpha * dy2(c, y2, x_origin, gamma1, theta2)
    scale = np.dot(e3.T, np.concatenate((x_m, y1_m, y2_m)))
    x = x_m - scale / (3 * n) * e1
    y1 = y1_m - scale / (3 * n) * e1
    y2 = y2_m - scale / (3 * n) * e1

    z1 = min(max((z1 - beta * dz1(lambda1, gamma2, z1)), 0), r)
    z2 = min(max((z2 - beta * dz2(lambda2, gamma2, z2)), 0), r)
    convergenceX[i] = np.linalg.norm(x - xstar) / np.linalg.norm(xstar)
    convergenceY[i] = np.linalg.norm(np.concatenate((y1, y2)) - np.concatenate((y1star, y2star))) / np.linalg.norm(np.concatenate((y1star, y2star)))
    Fvalue[i] = F(x, y1, y2)
    etw[i] = (e3.T @ np.concatenate((x, y1, y2)))
    
print('max(etw):', max(etw))
print('min(Fvalue)', min(Fvalue))
print('min(convergenceY)', min(convergenceY))
print('min(convergenceX)', min(convergenceX))

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.plot(iter, convergenceX)
plt.xlabel('Iteration')
plt.ylabel('||xk-x||/||x||')
plt.title('Function convergenceX Convergence')

plt.subplot(1, 3, 2)
plt.plot(iter, convergenceY)
plt.xlabel('Iteration')
plt.ylabel('||yk-y||/||y||')
plt.title('Point Sequence Convergence')

plt.subplot(1, 3, 3)
plt.plot(iter, Fvalue)
plt.xlabel('Iteration')
plt.ylabel('F(xk,yk)')
plt.title('Function convergenceX')

plt.tight_layout()
plt.show()
