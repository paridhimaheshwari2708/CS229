import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    clf = GDA()
    clf.fit(x_train, y_train)

    # Plot decision boundary on validation set
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    util.plot(x_valid, y_valid, clf.theta, save_path.replace('.txt', '.png'))

    # Use np.savetxt to save predictions from validation set to save_path
    y_hat_valid = clf.predict(x_valid)
    np.savetxt(save_path, y_hat_valid)
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, dim = x.shape

        # Find phi, mu_0, mu_1, and sigma
        phi = (1/n) * np.sum(y)
        mu_0 = (1/np.sum(y == 0)) * np.dot((y == 0).transpose(), x)
        mu_1 = (1/np.sum(y == 1)) * np.dot((y == 1).transpose(), x)
        sigma = (1/n) * (np.matmul(np.matmul((x-mu_0).transpose(), np.diag(y==0)), (x-mu_0)) \
                            + np.matmul(np.matmul((x-mu_1).transpose(), np.diag(y==1)), (x-mu_1)))

        # Write theta in terms of the parameters
        theta = np.matmul(np.linalg.inv(sigma), mu_1-mu_0)
        theta_0 = -np.log((1-phi)/phi) \
                    + (1/2) * np.matmul(np.matmul(mu_0.transpose(), np.linalg.inv(sigma)), mu_0) \
                    - (1/2) * np.matmul(np.matmul(mu_1.transpose(), np.linalg.inv(sigma)), mu_1)
        self.theta = np.insert(theta, 0, theta_0)

        return
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        g = np.dot(x, self.theta)
        y_hat = self.sigmoid(g)
        return y_hat
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
