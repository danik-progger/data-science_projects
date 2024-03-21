import numpy as np

# Initialize parameters
x = np.random.randn(10, 1)
y = 2 * x + np.random.randn()
w = 0.0
b = 0.0

# Hyperparameter
learning_rate = 0.01


def gradient_descend(x, y, w, b, learning_rate):
    dldw = 0.0
    dldb = 0.0
    N = x.shape[0]

    for xi, yi in zip(x, y):    # loss = (y - (wx + b))**2
        dldw += 2*(yi - (w*xi + b)) * (-xi)
        dldb += 2*(yi - (w*xi + b)) * (-1)

    w -= dldw * learning_rate / N
    b -= dldb * learning_rate / N

    return w, b


for epoch in range(400):
    w, b = gradient_descend(x, y, w, b, learning_rate)
    yhat = w*x + b
    loss = np.divide(np.sum((y - yhat)**2, axis=0), x.shape[0])
    print(f'{epoch} loss: {loss}   w: {w}   b: {b}')
