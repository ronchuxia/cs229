import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

    # *** START CODE HERE ***
    clf = GDA()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_eval)
    np.savetxt(pred_path, y_pred, fmt="%d")

    eval_accuracy = np.sum(y_pred == y_eval) / y_eval.shape[0]
    print("Eval Accuracy: ", eval_accuracy)

    util.plot(x_train, y_train, clf.theta, figure_path_train)
    util.plot(x_eval, y_eval, clf.theta, figure_path_eval)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m = y.shape[0]
        self.phi = np.sum(y == 1) / m   # shape (1, )
        # self.mu_1 = np.sum(np.expand_dims(y, axis=1) * x, axis=0) / np.sum(y == 1)   # shape (n, )
        # self.mu_0 = np.sum(np.expand_dims(np.logical_not(y), axis=1) * x, axis=0) / np.sum(y == 0)   # shape (n, )
        self.mu_1 = np.sum(x[y == 1], axis=0) / np.sum(y == 1)  # shape (n, )
        self.mu_0 = np.sum(x[y == 0], axis=0) / np.sum(y == 0)  # shape (n, )

        mu = np.expand_dims(y, axis=1) * np.expand_dims(self.mu_1, axis=0) + np.expand_dims(np.logical_not(y), axis=1) * np.expand_dims(self.mu_0, axis=0)  # shape (m, n)

        self.sigma = (x - mu).T @ (x - mu) / m  # shape (n, n)

        theta = np.linalg.inv(self.sigma).T @ (self.mu_1 - self.mu_0)  # shape (n, )    # 协方差矩阵是对称矩阵，可以不用转置
        theta_0 = - (1/2) * self.mu_1.T @ np.linalg.inv(self.sigma) @ self.mu_1 + (1/2) * self.mu_0.T @ np.linalg.inv(self.sigma) @ self.mu_0 - np.log((1 - self.phi) / self.phi)    # shape (1, )  # 一维数组可以不用转置，直接与二维数组点乘
        self.theta = np.insert(theta, 0, theta_0)   # shape (n + 1, )

        # print(self.theta)

        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return (util.add_intercept(x) @ self.theta > 0) # NOTE: 不需要计算概率，根据决策边界进行决策即可
    
        # prob_0 = (1 - self.phi) * np.exp(- (1/2) * np.einsum("ij,ij->i", (x - self.mu_0) @ np.linalg.inv(self.sigma), x - self.mu_0))
        # prob_1 = self.phi * np.exp(- (1/2) * np.einsum("ij,ij->i", (x - self.mu_1) @ np.linalg.inv(self.sigma), x - self.mu_1))
        # return prob_1 > prob_0
        # *** END CODE HERE ***

if __name__ == "__main__":
    figure_path_train = 'figures/p01e_ds1_train.png'
    figure_path_eval = 'figures/p01e_ds1_eval.png'
    main(train_path='../data/ds1_train.csv',
         eval_path='../data/ds1_valid.csv',
         pred_path='output/p01b_pred_1.txt')
    
    figure_path_train = 'figures/p01e_ds2_train.png'
    figure_path_eval = 'figures/p01e_ds2_eval.png'
    main(train_path='../data/ds2_train.csv',
         eval_path='../data/ds2_valid.csv',
         pred_path='output/p01b_pred_2.txt')