import os

import cvxpy as cp
import numpy as np
import time
import torch
import copy
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn import functional as F

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, random, time
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class LinearSVM(nn.Module):
    def __init__(self, input_size, n_classes, n_sample, device):
        super(LinearSVM, self).__init__()
        self.w = nn.Parameter(torch.ones(n_classes, input_size).to(device))
        self.b = nn.Parameter(torch.tensor(0.).to(device))
        self.xi = nn.Parameter(torch.ones(n_sample).to(device))
        self.C = nn.Parameter(torch.ones(n_sample).to(device))

    def forward(self, x):
        return F.linear(x, self.w, self.b)

    def loss_upper(self, y_pred, y_val):
        y_val_tensor = torch.Tensor(y_val).to(y_pred.device)
        x = torch.reshape(y_val_tensor, (y_val_tensor.shape[0], 1)) * y_pred / torch.linalg.norm(self.w)
        relu = nn.LeakyReLU()
        loss = torch.sum(relu(2 * torch.sigmoid(-5.0 * x) - 1.0))
        return loss

    def loss_lower(self):
        w2 = 0.5 * torch.linalg.norm(self.w) ** 2
        c_exp = torch.exp(self.C)
        xi_term = 0.5 * (torch.dot(c_exp, (self.xi) ** 2))
        loss = w2 + xi_term
        return loss

    def constrain_values(self, srt_id, y_pred, y_train):
        xi_sidx = srt_id
        xi_eidx = srt_id + len(y_pred)
        xi_batch = self.xi[xi_sidx:xi_eidx].to(y_pred.device)
        return 1 - xi_batch - y_train.view(-1) * y_pred.view(-1)

def run(seed, epochs, device):
    print("========run seed: {}=========".format(seed))

    data_list = []

    # f = open("fourclass.txt",encoding = "utf-8")
    f = open("gisette_scale", encoding="utf-8")
    a_list = f.readlines()
    f.close()
    for line in a_list:
        line1 = line.replace('\n', '')
        line2 = list(line1.split(' '))
        y = float(line2[0])
        x = [float(line2[i].split(':')[1]) if i < len(line2) and line2[i] != '' else 0 for i in range(1, 5001, 1)]
        data_list.append(x + [y])

    data_array = np.array(data_list)
    np.random.seed(seed)
    np.random.shuffle(data_array)

    z_train = data_array[:500, :-1]
    y_train = data_array[:500, -1]
    Corruption_rate = 0.4
    for i in range(500):
        value = np.random.choice([-1, 1], p=[Corruption_rate, 1 - Corruption_rate])
        y_train[i] *= value
    z_val = data_array[500:650, :-1]
    y_val = data_array[500:650, -1]
    z_test = data_array[650:, :-1]
    y_test = data_array[650:, -1]

    batch_size = 256
    data_train = TensorDataset(
        torch.tensor(z_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(
        dataset=data_train,
        batch_size=batch_size,
        shuffle=True)
    data_val = TensorDataset(
        torch.tensor(z_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32))
    val_loader = DataLoader(
        dataset=data_val,
        batch_size=batch_size,
        shuffle=True)
    data_test = TensorDataset(
        torch.tensor(z_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(
        dataset=data_test,
        batch_size=batch_size,
        shuffle=True)

    feature = 5000

    N_sample = y_train.shape[0]
    model = LinearSVM(feature, 1, N_sample, device).to(device)
    model.C.data.copy_(torch.Tensor(z_train.shape[0]).uniform_(-6.0, -5.0).to(device))
    model_theta = copy.deepcopy(model).to(device)

    alpha = 0.01
    beta = 0.1
    yita = 0.01
    gama1 = 0.1
    gama2 = 0.1

    lamda = torch.ones(N_sample).to(device)
    z = torch.ones(N_sample).to(device)

    params = [p for n, p in model.named_parameters() if n != 'C']
    params_theta = [p for n, p in model_theta.named_parameters() if n != 'C']

    x = cp.Variable(feature + 1 + 2 * N_sample)
    y = cp.Parameter(feature + 1 + 2 * N_sample)
    w = x[0:feature]
    b = x[feature]
    xi = x[feature + 1:feature + 1 + N_sample]
    C = x[feature + 1 + N_sample:]

    loss = cp.norm(x - y, 2) ** 2

    constraints = []
    for i in range(y_train.shape[0]):
        constraints.append(1 - xi[i] - y_train[i] * (cp.scalar_product(w, z_train[i]) + b) <= 0)

    obj = cp.Minimize(loss)
    prob = cp.Problem(obj, constraints)

    val_loss_list = []
    test_loss_list = []
    val_acc_list = []
    test_acc_list = []
    time_computation = []

    algorithm_start_time = time.time()

    for k in range(epochs):
        try:
            ck = (k + 1) ** 0.3
            # ck = 1 / ((k + 1) ** 0.3)
            model_theta.zero_grad()
            loss = model_theta.loss_lower()
            loss.backward()

            idx_glob = 0
            constr_glob_list = torch.ones(0).to(device)
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                log_probs = model_theta(images)
                cv = model_theta.constrain_values(idx_glob, log_probs, labels)
                lamda_batch = lamda[idx_glob:idx_glob + len(labels)].to(device)
                cv.backward(lamda_batch)
                constr_glob_list = torch.cat((constr_glob_list, cv), 0)
                idx_glob += len(labels)

            for i, p_theta in enumerate(params_theta):
                d4_theta = torch.zeros_like(p_theta.data)
                if p_theta.grad is not None:
                    d4_theta += p_theta.grad
                d4_theta += gama1 * (p_theta.data - params[i].data)
                p_theta.data.add_(d4_theta, alpha=-yita)

            lamda = lamda - yita * (-constr_glob_list + gama2 * (lamda - z))

            model_theta.zero_grad()

            loss = model_theta.loss_lower()
            loss.backward()

            idx_glob = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                log_probs = model_theta(images)
                cv = model_theta.constrain_values(idx_glob, log_probs, labels)
                lamda_batch = lamda[idx_glob:idx_glob + len(labels)]
                cv.backward(lamda_batch)
                idx_glob += len(labels)
            model.zero_grad()
            loss = model.loss_lower()
            loss.backward()

            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                log_probs = model(images)
                loss = model.loss_upper(log_probs, labels)
                loss.backward(torch.tensor(ck).to(device))

            for i, p in enumerate(params):
                d2 = torch.zeros_like(p.data)
                if p.grad is not None:
                    d2 += p.grad
                d2 += gama1 * (params_theta[i].data - p.data)
                p.data.add_(d2, alpha=-alpha)

            d1 = model.C.grad - model_theta.C.grad
            model.C.data.add(d1, alpha=-alpha)

            y_w = model.w.data.view(-1).detach().cpu().numpy()
            y_b = model.b.data.detach()
            y_xi = model.xi.data.view(-1).detach().numpy()
            y_C = model.C.data.view(-1).detach().numpy()

            y.value = np.concatenate((y_w, np.array([y_b.cpu()]), y_xi, y_C))

            prob.solve(solver='ECOS', abstol=2, reltol=2, max_iters=1800, feastol=2)
            C_solv = torch.Tensor(np.array(C.value)).to(device)
            w_solv = torch.Tensor(np.array([w.value])).to(device)
            b_solv = torch.tensor(b.value).to(device)
            xi_solv = torch.Tensor(np.array(xi.value)).to(device)

            model_theta.C.data.copy_(C_solv)
            model.C.data.copy_(C_solv)
            model.w.data.copy_(w_solv)
            model.b.data.copy_(b_solv)
            model.xi.data.copy_(xi_solv)

            number_right = 0
            val_loss = 0
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                log_probs = model(images)
                for i in range(len(labels)):
                    q = log_probs[i] * labels[i]
                    if q > 0:
                        number_right = number_right + 1
                val_loss += model.loss_upper(log_probs, labels)
            val_acc = number_right / len(y_val)
            val_loss /= 15.0

            number_right = 0
            test_loss = 0
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                log_probs = model(images)
                for i in range(len(labels)):
                    q = log_probs[i] * labels[i]
                    if q > 0:
                        number_right = number_right + 1
                test_loss += model.loss_upper(log_probs, labels)
            test_acc = number_right / len(y_test)
            test_loss /= 11.80
            print("val acc: {:.2f}".format(val_acc),
                  "val loss: {:.2f}".format(val_loss),
                  "test acc: {:.2f}".format(test_acc),
                  "test loss: {:.2f}".format(test_loss),
                  "round: {}".format(k))

            val_loss_list.append(val_loss.detach().cpu().numpy())
            test_loss_list.append(test_loss.detach().cpu().numpy())
            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)
            time_computation.append(time.time() - algorithm_start_time)
        except:
            break

    end_time = time.time()
    time_duaration = end_time - algorithm_start_time

    return val_loss_list, test_loss_list, val_acc_list, test_acc_list, time_computation, time_duaration


if __name__ == "__main__":
    if len(sys.argv) == 3:
        data_loop = int(sys.argv[1])
        epochs = int(sys.argv[2])
    else:
        print("Invalid params, run with default setting")
        data_loop = 9
        epochs = 80

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    val_loss_array = []
    test_loss_array = []
    val_acc_array = []
    test_acc_array = []
    time_duaration_array = []
    for seed in range(1, data_loop):
        val_loss_list, test_loss_list, val_acc_list, test_acc_list, time_computation, time_duaration = run(seed, epochs,
                                                                                                           device)
        val_loss_array.append(np.array(val_loss_list))
        test_loss_array.append(np.array(test_loss_list))
        val_acc_array.append(np.array(val_acc_list))
        test_acc_array.append(np.array(test_acc_list))
        time_computation = np.array(time_computation)
        time_duaration_array.append(time_duaration)

    max_length = max(len(lst) for lst in val_loss_array)

    val_loss_array = [np.pad(lst, (0, max_length - len(lst)), 'constant') if len(lst) < max_length else lst[:max_length]
                      for lst in val_loss_array]
    test_loss_array = [
        np.pad(lst, (0, max_length - len(lst)), 'constant') if len(lst) < max_length else lst[:max_length] for lst in
        test_loss_array]
    val_acc_array = [np.pad(lst, (0, max_length - len(lst)), 'constant') if len(lst) < max_length else lst[:max_length]
                     for lst in val_acc_array]
    test_acc_array = [np.pad(lst, (0, max_length - len(lst)), 'constant') if len(lst) < max_length else lst[:max_length]
                      for lst in test_acc_array]

    val_loss_array = np.array(val_loss_array)
    test_loss_array = np.array(test_loss_array)
    val_acc_array = np.array(val_acc_array)
    test_acc_array = np.array(test_acc_array)
    time_duaration_array = np.array(time_duaration_array)

    val_loss_mean = np.sum(val_loss_array, axis=0) / val_loss_array.shape[0]
    val_loss_sd = np.sqrt(np.var(val_loss_array, axis=0)) / 2.0
    test_loss_mean = np.sum(test_loss_array, axis=0) / test_loss_array.shape[0]
    test_loss_sd = np.sqrt(np.var(test_loss_array, axis=0)) / 2.0

    val_acc_mean = np.sum(val_acc_array, axis=0) / val_acc_array.shape[0]
    val_acc_sd = np.sqrt(np.var(val_acc_array, axis=0)) / 2.0
    test_acc_mean = np.sum(test_acc_array, axis=0) / test_acc_array.shape[0]
    test_acc_sd = np.sqrt(np.var(test_acc_array, axis=0)) / 2.0

    time_mean = np.sum(time_duaration_array, axis=0) / time_duaration_array.shape[0]
    print("*******************")
    print("Average running time for my algorithm: ", time_mean)
    print("Average test loss: ", test_loss_mean[-1])
    print("Average test acc: ", test_acc_mean[-1])
    print("*******************")

    plt.rcParams.update({'font.size': 18})
    plt.rcParams['axes.unicode_minus'] = False

    axis = np.arange(len(val_loss_mean))

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    plt.plot(axis, val_loss_mean, '-', label="Validation loss")
    ax.fill_between(axis, val_loss_mean - val_loss_sd, val_loss_mean + val_loss_sd, alpha=0.2)
    plt.plot(axis, test_loss_mean, '--', label="Test loss")
    ax.fill_between(axis, test_loss_mean - test_loss_sd, test_loss_mean + test_loss_sd, alpha=0.2)
    plt.title('Linear SVM')
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.legend(loc=0, numpoints=1)
    plt.savefig('new_run_usps_1.pdf')
    plt.show()

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    plt.plot(axis, val_acc_mean, '-', label="Validation accuracy")
    ax.fill_between(axis, val_acc_mean - val_acc_sd, val_acc_mean + val_acc_sd)
    plt.plot(axis, test_acc_mean, '--', label="Test accuracy")
    ax.fill_between(axis, test_acc_mean - test_acc_sd, test_acc_mean + test_acc_sd)
    plt.title('Linear SVM')
    plt.xlabel('Epochs')
    plt.ylabel("Accuracy")
    plt.ylim(0.64, 1.0)
    plt.legend(loc=0, numpoints=1)
    plt.savefig('new_run_usps_2.pdf')
    plt.show()

    end_time = time.time()
    print("time", end_time - start_time)
