import numpy as np
import matplotlib.pyplot as plt
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    # x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    model = PoissonRegression(step_size=lr)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred, fmt="%f")

    mean_error = np.mean(np.abs(y_eval - y_pred) / y_eval)
    print("Mean Test Error: {:.2%}".format(mean_error))
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n)    # shape (n, )
        
        v = np.ones_like(self.theta) * np.inf
        while self.step_size * np.linalg.norm(v) > 1e-5 :
            v = x.T @ (y - np.exp(x @ self.theta)) / m  # 除以 m，否则梯度过大
            self.theta += self.step_size * v
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x @ self.theta)
        # *** END CODE HERE ***

if __name__ == "__main__":
    main(lr=1e-7,
        train_path='../data/ds4_train.csv',
        eval_path='../data/ds4_valid.csv',
        pred_path='output/p03d_pred.txt')