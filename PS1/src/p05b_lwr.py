import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    model = LocallyWeightedLinearRegression(tau=tau)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_eval)
    
    mse = np.mean((y_pred - y_eval) ** 2)
    print("MSE: ", mse)

    plt.figure()
    plt.plot(x_train, y_train, "bx")
    plt.plot(x_eval, y_pred, "ro")
    plt.savefig(figure_path)
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x  # shape (m_train, n)
        self.y = y  # shape (m_train, )
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # y_pred = np.zeros(x.shape[0])
        # for i in range(x.shape[0]):
        #     w = np.exp(- np.linalg.norm(x[i] - self.x, axis=1) ** 2 / (2 * self.tau ** 2)) # shape (m, )
        #     W = np.diag(w)  # shape (m_train, m_train)
        #     theta = np.linalg.inv(self.x.T @ W @ self.x) @ self.x.T @ W @ self.y    # shape (n, )
        #     y_pred[i] = x[i] @ theta

        w = np.exp(- np.linalg.norm(np.expand_dims(x, axis=1) - self.x, axis=-1) ** 2 / (2 * self.tau ** 2)) # shape (m, m_train)
        W = np.apply_along_axis(np.diag, axis=1, arr=w)  # shape (m, m_train, m_train)  # 将 np.diag 作用到 w 的 axis=1 的每一行
        theta = np.linalg.inv(self.x.T @ W @ self.x) @ self.x.T @ W @ self.y    # shape (m, n)
        y_pred = np.einsum("ij,ij->i", x, theta)
        return y_pred
        # *** END CODE HERE ***

if __name__ == "__main__":
    figure_path = "figures/p05b_ds5.png"
    main(tau=5e-1,
         train_path='../data/ds5_train.csv',
         eval_path='../data/ds5_valid.csv')