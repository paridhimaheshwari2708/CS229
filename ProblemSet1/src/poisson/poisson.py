import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    clf = PoissonRegression(step_size=lr)
    clf.fit(x_train, y_train)

    # Run on the validation set, and use np.savetxt to save outputs to save_path
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    p_eval = clf.predict(x_eval)
    np.savetxt(save_path, p_eval)

    # Creating the Scatter Plot
    plt.figure()
    plt.scatter(y_eval,p_eval,alpha=0.5,c='blue',label='True vs Predicted Counts')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.legend()
    plt.savefig('poisson_valid.png')
    

    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
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

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, d = x.shape[0],x.shape[1]
        if(self.theta == None):
            self.theta = np.zeros(d)
        
        for iter in range(0, self.max_iter):
            h_theta_x = self.predict(x)
            new_theta = np.zeros(d)
            # This code is using for loops across the examples data
            # for i in range(0, d):
            #     batch_gradient = 0
            #     for j in range(0,n):
            #         batch_gradient = batch_gradient + (y[j]-h_theta_x[j])*x[j][i] 
            #     new_theta[i] = self.theta[i] + self.step_size*batch_gradient
            
            ## This code is faster as we use matrices for calucating the gradient
            batch_gradient = np.sum((y-h_theta_x).reshape(-1,1)*x, axis=0)
            new_theta = self.theta + self.step_size*batch_gradient

            # Checking the norm codition to break the loop
            norm = np.sum(np.abs(new_theta - self.theta))
            
            self.theta = new_theta

            if self.verbose and iter % 5 == 0:
                print('[iter: {:02d}, theta: {}]'
                      .format(iter, [round(t, 5) for t in self.theta]))

            
            if (norm<self.eps):
                break
            

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        y = np.exp(x.dot(self.theta))
        return y
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
