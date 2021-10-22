# Important note: you do not have to modify this file for your homework.

import util
import numpy as np


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    count, _ = X.shape

    probs = 1. / (1 + np.exp(-X.dot(theta)))
    grad = (Y - probs).dot(X)

    preds = np.where(probs > 0.5, 1, 0)
    accuracy = np.mean([preds == Y])
    return grad, accuracy


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    theta = np.zeros(X.shape[1])
    learning_rate = 0.1
    train_curve = []
    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad, accuracy = calc_grad(X, Y, theta)
        theta = theta + learning_rate * grad
        train_curve.append((i,accuracy))

        ## To restrict the number of iterations for plotting
        if i > 100000:
            return theta, train_curve
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return theta, train_curve


def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
    theta_a, train_curve_a = logistic_regression(Xa, Ya)
    # util.plot_points_fig(Xa, Ya,'data_A_plot.png')
    util.plot(Xa, Ya, theta_a, 'log_reg_a.png')
    util.plot_accuracy_curve(train_curve_a, 'train_curve_a.png')
    print("Training accuracy at the end for dataset A:", train_curve_a[-1][1])

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)
    theta_b, train_curve_b = logistic_regression(Xb, Yb)
    # util.plot_points_fig(Xb, Yb,'data_B_plot.png')
    util.plot(Xb, Yb, theta_b, 'log_reg_b.png')
    util.plot_accuracy_curve(train_curve_b, 'train_curve_b.png')
    print("Training accuracy at the end for dataset B:", train_curve_b[-1][1])

if __name__ == '__main__':
    main()
