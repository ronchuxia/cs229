import numpy as np
import util
import os

from linear_model import LinearModel

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_eval)
    np.savetxt(pred_path, y_pred, fmt="%d")

    eval_accuracy = np.sum(y_pred == y_eval) / y_eval.shape[0]
    print("Eval Accuracy: ", eval_accuracy)

    util.plot(x_train, y_train, clf.theta, figure_path_train)
    util.plot(x_eval, y_eval, clf.theta, figure_path_eval)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        def h(x):
            """
            Compute hypothesis.
            Args:
                x: Training example inputs. Shape (m, n).

            Returns:
                Outputs. Shape (m,).

            """
            return 1 / (1 + np.exp(- x @ self.theta))

        def H(x):
            """
            Compute hessian.
            Args:
                x: Training example inputs. Shape (m, n).

            Returns:
                Hessian of J(theta). Shape(n, n).
            """
            return (x.T * h(x) * (1 - h(x))) @ x / x.shape[0]   # NOTE

        m, n = x.shape
        self.theta = np.zeros(n)    # shape (n, )

        v = np.ones(self.theta.shape)
        while (np.linalg.norm(v) > 1e-5) :
            d_theta = - (1 / m) * np.sum(np.expand_dims(y - h(x), axis=1) * x, axis=0) 
            v = np.linalg.inv(H(x)) @ d_theta
            self.theta -= v

        # print(self.theta)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # prob = 1 / (1 + np.exp(- x @ self.theta))
        # pred = prob > 0.5
        # return pred

        return (x @ self.theta > 0) # NOTE: 不需要计算概率，根据决策边界进行决策即可
        # *** END CODE HERE ***

if __name__ == "__main__":
    figure_path_train = 'figures/p01b_ds1_train.png'
    figure_path_eval = 'figures/p01b_ds1_eval.png'
    main(train_path='../data/ds1_train.csv',
         eval_path='../data/ds1_valid.csv',
         pred_path='output/p01b_pred_1.txt')
    
    figure_path_train = 'figures/p01b_ds2_train.png'
    figure_path_eval = 'figures/p01b_ds2_eval.png'
    main(train_path='../data/ds2_train.csv',
         eval_path='../data/ds2_valid.csv',
         pred_path='output/p01b_pred_2.txt')
    