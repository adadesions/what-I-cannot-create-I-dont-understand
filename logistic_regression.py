import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def h(x, w):
    inner = np.dot(w, np.transpose(x))
    return 1/(1+np.exp(-inner))


def cost(h, y):
    return -y*np.log10(h)-(1-y)*np.log10(1-h)


def J(X, Y, W):
    sum_cost = 0
    m = len(Y)-1
    for x, y in zip(X, Y):
        sum_cost += cost(h(x, W), y)
    error_cost = sum_cost/m
    print('Cost =', error_cost)
    return error_cost


def gradient(X, Y, W, alpha):
    Xt = np.transpose(X)
    G = np.array([h(x, W) for x in X])
    mul_mat = np.dot(Xt, G-Y)
    m = len(Y)
    W_update = W-(alpha/m)*mul_mat
    print('Updated Weight:', W_update)

    return W_update

def plot_fitted_model(W, start, end, offset=0.1):
    X = []
    for i in np.arange(start, end, offset):
        for j in np.arange(start, end, offset):
            X.append([i, j])

    y = [h(x, W) for x in X]
    Xt = np.transpose(X)

    return [Xt[0], Xt[1], y]


if __name__ == '__main__':
    X = [
        [1, 1],
        [0, 1],
        [1, 0],
        [0, 0],
        [0.5, 0],
        [0.24, 0.38],
        [1.2, 0.45],
    ]
    X = np.array(X)
    Xt = np.transpose(X)
    Y = np.array([0, 1, 1, 0, 1, 0, 0])
    W = np.array([0.1, 0.1])
    alpha = 0.5
    eps = 0.001
    e0 = J(X, Y, W)
    W0 = [W[0]]
    W1 = [W[1]]
    errors = [e0]
    for iters in np.arange(10000):
        print('======== iteration: {} ========'.format(iters))
        W = gradient(X, Y, W, alpha)
        e1 = J(X, Y, W)

        W0.append(W[0])
        W1.append(W[1])
        errors.append(e1)

        if np.abs(e1-e0) < eps:
            print('Min W:', W)
            break

        e0 = e1


    f_model = plot_fitted_model(W, 0, 1, 0.1)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(Xt[0], Xt[1], Y, marker='^')
    ax.plot_trisurf(f_model[0], f_model[1], f_model[2], cmap=cm.hot)

    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    ax2.plot_trisurf(W0, W1, errors, cmap=cm.cool)
    plt.show()
