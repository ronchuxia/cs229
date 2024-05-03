import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train, y_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    np.savetxt(pred_path_c, y_pred, fmt="%d")

    test_accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
    print("C Test Accuracy: ", test_accuracy)

    util.plot(x_train, y_train, model.theta, figure_path_train.replace(WILDCARD, 'c'))
    util.plot(x_test, y_test, model.theta, figure_path_test.replace(WILDCARD, 'c'))

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    np.savetxt(pred_path_c, y_pred, fmt="%d")

    test_accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
    print("D Test Accuracy: ", test_accuracy)

    util.plot(x_train, y_train, model.theta, figure_path_train.replace(WILDCARD, 'd'))
    util.plot(x_test, y_test, model.theta, figure_path_test.replace(WILDCARD, 'd'))

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_val, y_val = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Estimate correction factor alpha
    prob_val = 1 / (1 + np.exp(- x_val @ model.theta))
    alpha = np.mean(prob_val)

    # Apply correction factor alpha and predict labels
    prob_test = 1 / (1 + np.exp(- x_test @ model.theta)) / alpha
    y_pred = prob_test > 0.5
    np.savetxt(pred_path_c, y_pred, fmt="%d")

    test_accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
    print("E Test Accuracy: ", test_accuracy)

    util.plot(x_train, y_train, model.theta, figure_path_train.replace(WILDCARD, 'e'), correction=alpha)
    util.plot(x_test, y_test, model.theta, figure_path_test.replace(WILDCARD, 'e'), correction=alpha)
    # *** END CODER HERE

if __name__ == "__main__":
    figure_path_train = 'figures/p02X_ds3_train.png'
    figure_path_test = 'figures/p02X_ds3_test.png'
    main(train_path='../data/ds3_train.csv',
        valid_path='../data/ds3_valid.csv',
        test_path='../data/ds3_test.csv',
        pred_path='output/p02X_pred.txt')