import util
import numpy as np
import matplotlib.pyplot as plt


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad


def plot_dataset(X, Y, save_path):
    plt.figure()
    plt.plot(X[Y == 1, -2], X[Y == 1, -1], 'bx')
    plt.plot(X[Y == -1, -2], X[Y == -1, -1], 'go')
    plt.savefig(save_path)


def plot_loss_function(X, Y, save_path):
    """
    Plot the loss function field and the optimization process.

    Inputs:
    - X: input features, shape (m, 3)
    - Y: ground truth labels, shape (m, )
    """
    # Plot the loss function field
    theta_0 = np.arange(-50, 0, 1)
    theta_1 = np.arange(0, 50, 1)
    theta_2 = np.arange(0, 50, 1)
    theta_0, theta_1, theta_2 = np.meshgrid(theta_0, theta_1, theta_2)    # shape (num_theta_0, num_theta_1, num_theta_2)
    
    theta = np.stack([theta_0, theta_1, theta_2], axis=-1)    # shape (num_theta_0, num_theta_1, num_theta_2, 3)
    num_theta_0, num_theta_1, num_theta_2, _ = theta.shape
    theta = np.reshape(theta, (num_theta_0 * num_theta_1 * num_theta_2, -1))   # shape (num_theta_0 * num_theta_1 * num_theta_2, 3)

    loss = np.sum(np.log(1 + np.exp(- np.expand_dims(Y, axis=-1) * (X @ theta.T))), axis=0)    # shape (num_theta_0 * num_theta_1 * num_theta_2, )
    loss = np.reshape(loss, (num_theta_0, num_theta_1, num_theta_2))

    print(np.sum(np.log(1 + np.exp(- Y * (X @ np.array([-0, 0, 0])))), axis=0))    
    print(np.sum(np.log(1 + np.exp(- Y * (X @ np.array([-20, 20, 20])))), axis=0))
    print(np.sum(np.log(1 + np.exp(- Y * (X @ np.array([-50, 50, 50])))), axis=0))
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(theta_0, theta_1, theta_2, s=0.1, c=loss, cmap=plt.cm.jet, vmin=16.5, vmax=25, alpha=0.1)

    # Plot the optimization process
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    i = 0
    theta_0_list = []
    theta_1_list = []
    theta_2_list = []
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad

        theta_0_list.append(theta[0])
        theta_1_list.append(theta[1])
        theta_2_list.append(theta[2])

        if np.linalg.norm(prev_theta - theta) < 1e-15 or theta[0] < -50:
            break

    ax.plot(theta_0_list, theta_1_list, theta_2_list)

    plt.savefig(save_path)


def main():
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    plot_dataset(Xa, Ya, "figures/p01_lr_ds1_a.png")
    plot_loss_function(Xa, Ya, "figures/p01_lr_ds1_a_loss.png")

    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    plot_dataset(Xb, Yb, "figures/p01_lr_ds1_b.png")
    plot_loss_function(Xb, Yb, "figures/p01_lr_ds1_b_loss.png")


if __name__ == "__main__":
    main()