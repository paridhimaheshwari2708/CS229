import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.linalg.solve(np.matmul(X.transpose(),X) ,np.matmul(X.transpose(),y) )
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        new_poly_x = X[:,0].reshape(-1,1) ## Starting the single column for x0
        for i in range(1, k+1):
            power_x_col = np.power(X[:,1],i).reshape(-1,1) ## For each new feature, taking the power to create a new column array
            new_poly_x = np.hstack((new_poly_x, power_x_col)) ## Appending the arrays for feature matrix
        return new_poly_x
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        new_feature_x = X[:,0].reshape(-1,1) ## Starting the single column for x0
        for i in range(1, k+1):
            power_x_col = np.power(X[:,1],i).reshape(-1,1) ## For each new feature, taking the power to create a new column array
            new_feature_x = np.hstack((new_feature_x, power_x_col)) ## Appending the arrays for feature matrix
        sin_x_col = np.sin(X[:,1]).reshape(-1,1)
        new_feature_x = np.hstack((new_feature_x, sin_x_col))
        return new_feature_x
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        h_theta_x = np.matmul(X,self.theta)
        return h_theta_x
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        LM_model = LinearModel()
        if(sine==True):
            train_x_hat = LM_model.create_sin(k,train_x)
            plot_x_hat = LM_model.create_sin(k,plot_x)
        else:
            train_x_hat = LM_model.create_poly(k,train_x)
            plot_x_hat = LM_model.create_poly(k,plot_x)
        LM_model.fit(train_x_hat, train_y)

        ## Testing the model on plot_x data
        plot_y = LM_model.predict(plot_x_hat)
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all experiments
    '''
    # *** START CODE HERE ***
    run_exp(train_path=train_path,sine=False, ks=[3],filename='plot_5b.png' )
    run_exp(train_path=train_path,sine=False, ks=[3,5,10,20],filename='plot_5c.png' )
    run_exp(train_path=train_path,sine=True, ks=[0,1,2,3,5,10,20],filename='plot_5d.png' )
    run_exp(train_path=small_path,sine=False, ks=[1,2,3,5,10,20],filename='plot_5e.png' )
    run_exp(train_path=small_path,sine=True, ks=[1,2,3,5,10,20],filename='plot_5e_2.png' )
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
