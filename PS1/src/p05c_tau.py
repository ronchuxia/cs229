import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    mse_list = []
    for tau in tau_values:
        x_train, y_train = util.load_dataset(train_path, add_intercept=True)

        model = LocallyWeightedLinearRegression(tau=tau)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_val)
        
        mse = np.mean((y_pred - y_val) ** 2)
        print("tau: ", tau, "mse: ", mse)

        plt.figure()
        plt.plot(x_train, y_train, "bx")
        plt.plot(x_val, y_pred, "ro")
        plt.savefig("figures/p05c_ds5_" + str(tau) + ".png")

    # *** END CODE HERE ***

if __name__ == "__main__":
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
        train_path='../data/ds5_train.csv',
        valid_path='../data/ds5_valid.csv',
        test_path='../data/ds5_test.csv',
        pred_path='output/p05c_pred.txt')
