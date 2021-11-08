import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    n = x.shape[0]
    random_groups = np.random.randint(low=0, high = K, size=n ) ## Choosing random groups for each sample
    mu, sigma = [],[]
    for i in range(K):
        x_filtered = x[random_groups==i]
        mu_i = np.mean(x_filtered, axis=0)
        sigma_i = np.cov(x_filtered.T) ## Each row should be variable and columns should be observations. Hence, taking transpose
        mu.append(mu_i)
        sigma.append(sigma_i)
    # print(mu[0].shape, sigma[0].shape, x[0].shape, len(mu))

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.full(shape=(K,), fill_value= (1.0/K)) ## Equal probability

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.full(shape=(n,K), fill_value= (1.0/K))
    # print(type(w), w.dtype)

    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        w = e_step(x, w, mu, sigma, phi)

        # (2) M-step: Update the model parameters phi, mu, and sigma

        phi, mu, sigma = m_step(x, w, mu, sigma)
        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll
        ll = get_log_likelihood(x,mu,sigma,phi)
        it += 1
        print(it, ll)
        # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        w = e_step(x, w, mu, sigma, phi)

        # (2) M-step: Update the model parameters phi, mu, and sigma
        
        phi, mu, sigma = m_step_semi_supervised(x,x_tilde,z_tilde, w, mu, sigma, alpha)

        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll
        ll = get_log_likelihood_semi_supervised(x,x_tilde,z_tilde, w, mu, sigma, phi, alpha)
        it += 1
        print(it, ll)
        # *** END CODE HERE ***

    return w


# *** START CODE HERE ***
# Define any helper functions

def prob_x_given_z_gaussian(x, mu, sigma, dim):

    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    part_2 = (-1.0/2)*(x-mu).T.dot(sigma_inv).dot(x-mu)
    part_1 = np.power(2*np.pi, -dim/2.0)*np.power(sigma_det, -1.0/2)
    prob_x_given_z = part_1*np.exp(part_2)

    return prob_x_given_z

def e_step(x, w, mu, sigma, phi):
    ## Need to calculate p_z_given_x; Use Bayes theorem to calculate this from the know variables
    num_samples, dim = x.shape
    for n in range(num_samples):
        for i in range(K):
            mu_i = mu[i]
            sigma_i = sigma[i]

            p_x_given_z = prob_x_given_z_gaussian(x[n], mu_i,sigma_i, dim)
            mariginal_prob = p_x_given_z * phi[i]
            w[n][i] = mariginal_prob
    
    w_updated = w/ np.sum(w, axis =1, keepdims=True) # Sum in the denominator gives p_x

    return w_updated


def m_step(x,w, mu, sigma):
    phi_updated = np.mean(w, axis=0)

    num_samples = x.shape[0]

    mu_updated, sigma_updated = [], []
    for l in range(K):
        w_l = w[:,l:l+1]
        mu_l = np.sum(w_l*x, axis = 0)/ np.sum(w_l) ## updating mu in the same list 
        mu_updated.append(mu_l)
        # print(mu_l.shape)
        sum_term = 0
        for n in range(num_samples):
            outer_product = np.outer((x[n]-mu_l),(x[n]-mu_l))
            sum_term += w[n][l] * outer_product
        sigma_new = sum_term/np.sum(w_l)
        # print(sigma_new.shape)
        sigma_updated.append(sigma_new)

    return phi_updated, mu_updated, sigma_updated

def m_step_semi_supervised(x,x_tilde,z_tilde, w, mu, sigma, alpha):

    num_samples, dim = x.shape
    n_tilda = x_tilde.shape[0]
    phi_updated, mu_updated, sigma_updated = [],[],[]
    for l in range(K):
        w_l = w[:,l:l+1]
        num_phi = np.sum(w_l) + alpha * np.sum(z_tilde==l)
        denom_phi = num_samples + alpha * n_tilda
        phi_new = num_phi/denom_phi
        phi_updated.append(phi_new)
        
        # print("here", x_tilde[(z_tilde == l).reshape(-1)].shape, np.sum(z_tilde == l))
        x_l_filtered = x_tilde[(z_tilde == l).reshape(-1)]
        num_mu = np.sum(w_l*x, axis = 0) + alpha * np.sum(x_l_filtered, axis=0)
        denom_mu = np.sum(w_l) + alpha * np.sum(z_tilde == l)
        mu_l = num_mu/denom_mu
        mu_updated.append(mu_l)
        # print(mu_l.shape)
        sigma_num_term = 0
        for n in range(0,num_samples):
            outer_product = np.outer((x[n]-mu_l),(x[n]-mu_l))
            sigma_num_term += w[n][l] * outer_product

        for n in range(0,x_l_filtered.shape[0]):
            outer_product = np.outer((x_l_filtered[n]-mu_l),(x_l_filtered[n]-mu_l))
            sigma_num_term += alpha * outer_product

        sigma_l = sigma_num_term/denom_mu ## denominatorr same for mu and sigma update rule
        sigma_updated.append(sigma_l)

    return phi_updated, mu_updated, sigma_updated

def get_log_likelihood(x, mu,sigma,phi):

    num_samples, dim = x.shape
    log_ll = 0
    for n in range(num_samples):
        sum_z =0
        for i in range(K):
            p_x_given_z = prob_x_given_z_gaussian(x[n], mu[i],sigma[i], dim)

            mariginal_prob = p_x_given_z * phi[i]  
            sum_z += mariginal_prob
        log_prob = np.log(sum_z)
        log_ll += log_prob
    return log_ll

def get_log_likelihood_semi_supervised(x,x_tilde,z_tilde, w, mu, sigma, phi, alpha):
    
    n_tilda, dim = x_tilde.shape

    ll_sup = 0
    for n in range(0,n_tilda):
        label = int(z_tilde[n])
        # print(mu[label].shape,sigma[label].shape)
        p_x_given_z = prob_x_given_z_gaussian(x_tilde[n], mu[label],sigma[label], dim)
        probability = p_x_given_z * phi[label]  
        ll_sup += np.log(probability)
    
    ll_total = get_log_likelihood(x, mu,sigma,phi) + alpha * ll_sup

    return ll_total
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.

        main(is_semi_supervised=True, trial_num=t)

        # *** END CODE HERE ***
