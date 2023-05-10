import numpy as np

    # Activation function
    def phi(x)
    y = np.exp(x)
    return y


def elm_train_model(Xd, Yd, model_data):
    Yd = np.reshape(Yd, (-1, 1))
    s_sz = len(Yd)
    Xd = np.reshape(Xd, (s_sz, -1))
    ndim = Xd.shape[1]

    # User choice parameters
    p = model_data.p
    lambda_ = model_data.lambda_
    N = model_data.nneurons


    # Create sparse weight matrix
    W = np.random.randn(ndim, N)
    Z = np.random.rand(ndim, N) > p
    W = W * Z

    # Bias vector
    bias = np.random.randn(1, N)

    # Solve least squares
    H = phi(np.dot(Xd, W) + bias)
    G = np.dot(H.T, H)
    G_dim = G.shape[0]
    beta = np.linalg.solve(G + (lambda_ * np.eye(G_dim)), np.dot(H.T, Yd))

    return W, bias, beta
