import sys
import time

import numpy as np
from joblib import cpu_count
from numpy.linalg import svd
from scipy import linalg
from sklearn.decomposition import sparse_encode
from sklearn.utils import check_random_state


def _update_dict_procrustes(dictionary, Y, code, verbose=False,
                            return_r2=False,
                            random_state=None):
    """Update the dense dictionary factor in place.

    Parameters
    ----------
    dictionary : array of shape (n_features, n_components)
        Value of the dictionary at the previous iteration.

    Y : array of shape (n_features, n_samples)
        Data matrix.

    code : array of shape (n_components, n_samples)
        Sparse coding of the data against which to optimize the dictionary.

    verbose:
        Degree of output the procedure will print.

    return_r2 : bool
        Whether to compute and return the residual sum of squares corresponding
        to the computed solution.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    dictionary : array of shape (n_features, n_components)
        Updated dictionary.

    """
    n_components = len(code)
    n_samples = Y.shape[0]
    # Residuals, computed 'in-place' for efficiency
    Q = Y.dot(code.T)
    U, S, Vt = svd(Q)
    dictionary = U.dot(Vt)
    residuals = np.sum((Y - dictionary.dot(code)) ** 2)
    return dictionary, residuals


def dict_learning(X, n_components, alpha, max_iter=100, tol=1e-8,
                  method='lars', n_jobs=1, dict_init=None, code_init=None,
                  callback=None, verbose=False, random_state=None,
                  sample_weights=None,
                  return_n_iter=False):
    """Solves a dictionary learning matrix factorization problem.

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::

        (U^*, V^*) = argmin 0.5 || X - U V ||_2^2 + alpha * || U ||_1
                     (U,V)
                    with || V_k ||_2 = 1 for all  0 <= k < n_components

    where V is the dictionary and U is the sparse code.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Data matrix.

    n_components : int,
        Number of dictionary atoms to extract.

    alpha : int,
        Sparsity controlling parameter.

    max_iter : int,
        Maximum number of iterations to perform.

    tol : float,
        Tolerance for the stopping condition.

    method : {'lars', 'cd'}
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.

    n_jobs : int,
        Number of parallel jobs to run, or -1 to autodetect.

    dict_init : array of shape (n_components, n_features),
        Initial value for the dictionary for warm restart scenarios.

    code_init : array of shape (n_samples, n_components),
        Initial value for the sparse code for warm restart scenarios.

    callback : callable or None, optional (default: None)
        Callable that gets invoked every five iterations

    verbose : bool, optional (default: False)
        To control the verbosity of the procedure.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    return_n_iter : bool
        Whether or not to return the number of iterations.

    Returns
    -------
    code : array of shape (n_samples, n_components)
        The sparse code factor in the matrix factorization.

    dictionary : array of shape (n_components, n_features),
        The dictionary factor in the matrix factorization.

    errors : array
        Vector of errors at each iteration.

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to True.

    See also
    --------
    dict_learning_online
    DictionaryLearning
    MiniBatchDictionaryLearning
    SparsePCA
    MiniBatchSparsePCA
    """
    if method not in ('lars', 'cd'):
        raise ValueError('Coding method %r not supported as a fit algorithm.'
                         % method)
    method = 'lasso_' + method

    t0 = time.time()
    # Avoid integer division problems
    alpha = float(alpha)
    random_state = check_random_state(random_state)

    if n_jobs == -1:
        n_jobs = cpu_count()

    if sample_weights is None:
        sample_weights = np.ones(X.shape[0])

    # Init the code and the dictionary with SVD of Y
    if code_init is not None and dict_init is not None:
        code = np.array(code_init, order='F')
        # Don't copy V, it will happen below
        dictionary = dict_init
    else:
        code, S, dictionary = linalg.svd(X, full_matrices=False)
        dictionary = S[:, np.newaxis] * dictionary
    r = len(dictionary)
    if n_components <= r:  # True even if n_components=None
        code = code[:, :n_components]
        dictionary = dictionary[:n_components, :]
    else:
        code = np.c_[code, np.zeros((len(code), n_components - r))]
        dictionary = np.r_[dictionary,
                           np.zeros((n_components - r, dictionary.shape[1]))]

    # Fortran-order dict, as we are going to access its row vectors
    dictionary = np.array(dictionary, order='F')

    residuals = 0

    errors = []
    current_cost = np.nan

    if verbose == 1:
        print('[dict_learning]', end=' ')

    # If max_iter is 0, number of iterations returned should be zero
    ii = -1

    for ii in range(max_iter):
        dt = (time.time() - t0)
        if verbose == 1:
            sys.stdout.write(".")
            sys.stdout.flush()
        elif verbose:
            print("Iteration % 3i "
                  "(elapsed time: % 3is, % 4.1fmn, current cost % 7.3f)"
                  % (ii, dt, dt / 60, current_cost))

        # Update code
        code = sparse_encode(X, dictionary, algorithm=method, alpha=alpha,
                             init=code, n_jobs=n_jobs)
        # Update dictionary
        dictionary, residuals = _update_dict_procrustes(dictionary.T,
                                                        X.T * sample_weights[
                                                              None, :], code.T,
                                                        verbose=verbose,
                                                        return_r2=True,
                                                        random_state=random_state)
        dictionary = dictionary.T

        # Cost function
        current_cost = 0.5 * residuals + alpha * np.sum(np.abs(code))
        errors.append(current_cost)

        if ii > 0:
            dE = errors[-2] - errors[-1]
            # assert(dE >= -tol * errors[-1])
            if dE < tol * errors[-1]:
                if verbose == 1:
                    # A line return
                    print("")
                elif verbose:
                    print("--- Convergence reached after %d iterations" % ii)
                break
        if ii % 5 == 0 and callback is not None:
            callback(locals())

    if return_n_iter:
        return code, dictionary, errors, ii + 1
    else:
        return code, dictionary, errors
