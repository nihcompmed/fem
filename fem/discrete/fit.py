"""This module implements a class that can fit the Potts model to data and make predictions with the model. `FEM for discrete data`_

.. _FEM for discrete data:
   https://joepatmckenna.github.io/fem/discrete.html
"""

import time
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
import combinatorics
from .. import fortran_module


class model(object):
    """This class implements a Potts model that can be fit to data and used to make predictions.

    Attributes:
        degs (list): A list of model degrees
        n_x (int): The number of input variables
        n_y (int): The number of output variables
        m_x (ndarray): An array of length `n_x` listing the number of values that each input variable takes
        m_y (ndarray):
        m_x_cumsum (ndarray):
        m_y_cumsum (ndarray):
        x_int (ndarray): The input variable data `x` mapped to nonnegative integers
        y_int (ndarray): The output variable data `y` mapped to nonnegative integers
        cat_x (list): A list of per-row dicts that map the input x symbol data to the integer data in `x_int`
        cat_y (list):
        cat_x_inv (list): A list of dicts that are the inverses of the list of dicts in `cat_x`
        cat_y_inv (list):
        w (dict): A dictionary of length `len(degs)` keyed by `degs`. `w[deg]` are the model parameters for degree `deg`
        disc (list): A list of length `n_x` of the running discrepancies per variable computed during the model fit
        impute (bool): If true, the model will map each combination of hold-one-out input variables to the held out variable

    Examples:

        >>> import fem
        >>> n, m = 10, 3
        >>> fem.discrete.time_series(fem.discrete.model_parameters(n, m), n, m)
        >>> model = fem.discrete.model()
        >>> model.fit(x[:, :-1], x[:, 1:])
        >>> model.predict(x[:, -1])

    """

    def __init__(self, degs=[1]):
        self.degs = degs

    def fit(self,
            x,
            y=None,
            iters=100,
            overfit=True,
            impute=None,
            svd='approx'):
        """Fit the Potts model to the data

        Args:
            x (ndarray):
            y (ndarray):
            iters (int):
            overfit (bool):
            impute (bool):
            svd (str):

        Returns:
            (dict, list): The fitted model parameters and the running discrepancies
        """

        degs = self.degs

        x = np.array(x)
        x_int, cat_x = categorize(x)

        m_x = np.array([len(c) for c in cat_x])
        n_x = x_int.shape[0]

        if y is None:
            impute = True
            y = x
            y_int, cat_y = x_int, cat_x
            m_y = m_x
            n_y = n_x
        else:
            impute = False
            y = np.array(y)
            y_int, cat_y = categorize(y)
            m_y = np.array([len(c) for c in cat_y])
            n_y = y_int.shape[0]

        cat_x_inv = [{v: k for k, v in cat.iteritems()} for cat in cat_x]
        cat_y_inv = [{v: k for k, v in cat.iteritems()} for cat in cat_y]

        m_x_cumsum = np.insert(m_x.cumsum(), 0, 0)
        m_y_cumsum = np.insert(m_y.cumsum(), 0, 0)

        idx_x_by_deg = [combinatorics.multiindices(n_x, deg) for deg in degs]
        mm_x = np.array(
            [np.sum([np.prod(m_x[i]) for i in idx]) for idx in idx_x_by_deg])
        mm_x_cumsum = np.insert(mm_x.cumsum(), 0, 0)

        if (not impute) or (impute and svd == 'approx'):

            x_oh = one_hot(x_int, m_x, degs)
            x_oh_pinv = svd_pinv(x_oh)

            w, disc, it = fortran_module.fortran_module.discrete_fit(
                x_int, y_int, m_x, m_y,
                m_y.sum(), degs, x_oh_pinv[0], x_oh_pinv[1], x_oh_pinv[2],
                iters, overfit, impute)

            disc = [d[1:it[i]] for i, d in enumerate(disc)]

        elif impute and svd == 'exact':

            w, disc = np.zeros((mm_x_cumsum[-1], mm_x_cumsum[-1])), []

            for i in range(n_x):

                not_i = np.delete(range(n_x), i)
                x_oh = one_hot(x_int[not_i], m_x[not_i], degs)

                x_oh_pinv = svd_pinv(x_oh)

                wi, d, it = fortran_module.fortran_module.discrete_fit(
                    x_int[not_i], y_int[[i]], m_x[not_i], m_y[[i]],
                    m_y[i].sum(), degs, x_oh_pinv[0], x_oh_pinv[1],
                    x_oh_pinv[2], iters, overfit, False)

                end = time.time()

                w[m_x_cumsum[i]:m_x_cumsum[i + 1], :m_x_cumsum[
                    i]] = wi[:, :m_x_cumsum[i]]
                w[m_x_cumsum[i]:m_x_cumsum[i + 1], m_x_cumsum[
                    i + 1]:] = wi[:, m_x_cumsum[i]:]

                disc.append(d[0][1:it[0]])

        w = {
            deg: w[:, mm_x_cumsum[i]:mm_x_cumsum[i + 1]]
            for i, deg in enumerate(degs)
        }

        self.impute = impute

        self.x_int = x_int
        self.cat_x = cat_x
        self.m_x = m_x
        self.n_x = n_x

        self.cat_x_inv = cat_x_inv
        self.cat_y_inv = cat_y_inv
        self.m_x_cumsum = m_x_cumsum
        self.m_y_cumsum = m_y_cumsum

        self.y_int = y_int
        self.cat_y = cat_y
        self.m_y = m_y
        self.n_y = n_y

        self.disc = disc
        self.w = w

    def predict(self, x):
        """Predict the state of the output variable given input variable `x`

        Args:
            x (ndarray):

        Returns:
            (ndarray, ndarray): prediction and probability

        """
        cat_x = self.cat_x
        w = self.w
        degs = self.degs
        m_x = self.m_x
        n_y = self.n_y
        m_y_cumsum = self.m_y_cumsum
        cat_y_inv = self.cat_y_inv

        x = np.array(x)
        if x.ndim == 1:
            x = x[:, np.newaxis]

        x_int = np.empty(x.shape, dtype=int)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x_int[i, j] = cat_x[i][x[i, j]]

        x_oh = one_hot(x_int, m_x, degs)
        x_oh = x_oh.toarray()

        w = np.hstack(w.values())

        p = np.exp(w.dot(x_oh))
        p = np.split(p, m_y_cumsum[1:-1], axis=0)
        for i in range(n_y):
            p[i] /= p[i].sum(0)

        y_int = np.array([pi.argmax(axis=0) for pi in p])

        j = np.arange(y_int.shape[1])
        p = np.array([pi[yi, j] for yi, pi in zip(y_int, p)])

        y = np.empty(y_int.shape, dtype=x.dtype)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                y[i, j] = cat_y_inv[i][y_int[i, j]]

        y = y.squeeze()
        p = p.squeeze()

        return y, p

    def energy(self, x):

        cat_x = self.cat_x
        m_x = self.m_x
        w = self.w

        x_int = [cat_x[i][j] for i, j in enumerate(x)]
        x_oh = one_hot(x_int, m_x)

        return -(x_oh.T * w[1] * x_oh).squeeze()


def one_hot(x, m, degs=[1]):
    """Compute the one-hot encoding of `x`

    Args:
        x (ndarray): `n` by `l` array of nonnegative integers to be converted to one-hot encoding
        m (ndarray): length `n` array storing the number of possible integers in each row of `x`
        degs (list): list of degrees of one-hot-encodings to compute

    Returns:
        csc_matrix: sparse matrix that is the one-hot encoding of `x`. The degrees of one-hot encodings are vertically stacked.

    Examples:
        The one-hot encoding of a discrete variable records the state of the variable by a boolean vector with a 1 at the index which is the state of the variable. The one-hot encoding of a system of discrete variables is the concatenation of the one-hot encodings of each variable. For example if the system of discrete variables :math:`(x_1, x_2, x_3)` has state :math:`(x_1,x_2,x_3)=(1,1,2)` and the variables may take values from the first :math:`m=(3,3,4)` nonnegative integers, i.e. :math:`x_1,x_2\in\{0,1,2\}` and :math:`x_3\in\{0,1,2,3\}`, then the one-hot encoding of :math:`(x_1, x_2, x_3)^T` is :math:`\sigma=(0,1,0,0,1,0,0,0,1,0)^T`.

        >>> from fem.discrete import one_hot
        >>> x = [1,1,2]
        >>> m = [3,3,4]
        >>> x_oh = one_hot(x, m)
        >>> print x_oh.todense()
        matrix([[0.],
                [1.],
                [0.],
                [0.],
                [1.],
                [0.],
                [0.],
                [0.],
                [1.],
                [0.]])
    """

    x = np.array(x)
    m = np.array(m)
    degs = np.array(degs)

    if x.ndim == 1:
        x = x[:, np.newaxis]

    n, l = x.shape
    k = degs.shape[0]

    max_deg = degs.max()

    # list of indices that correspond to one hot encoding powers
    idx_len = combinatorics.binomial_coefficients(n, max_deg)[degs].sum()
    idx = []
    for deg in degs:
        for i in combinatorics.multiindices(n, deg):
            idx.append(i)

    # height of vertical vectors
    mi = np.array([np.prod(m[i]) for i in idx])
    # height of one-hot encoding
    m_sum = mi.sum()

    # convert one hot power to binary index
    s = np.vstack(
        [combinatorics.mixed_radix_to_base_10(x[i], m[i]) for i in idx])

    # row offsets
    stratifier = np.insert(mi.cumsum(), 0, 0)[:-1]

    # construct sparse matrix
    data = np.ones(idx_len * l)
    indices = (s + stratifier[:, np.newaxis]).T.flatten()
    indptr = idx_len * np.arange(l + 1)
    x_oh = csc_matrix((data, indices, indptr), shape=(m_sum, l))

    return x_oh


def categorize(x):
    """Convert symbolic x data to integer data. `x` contains a list of lists that are samples of different variables. Each list of samples is mapped to the first number of unique values detected of nonnegative integers.

    Args:
        x (list): A list where each element is a list of samples of a different variable

    Returns:
        (list, list): The integer data and a list of dictionaries that map symbol data to integer data for each row of `x`

    Examples:

        >>> from fem.discrete import categorize
        >>> x = [['a', 'b', 'a', 'a'], [10, 11, 12, 10, 11]]
        >>> x_int, cat_x = categorize(x)
        >>> x_int
        [array([0, 1, 0, 0]), array([0, 1, 2, 0, 1])]
        >>> cat_x
        [{'a': 0, 'b': 1}, {10: 0, 11: 1, 12: 2}]
    """

    n = len(x)
    l = [len(xi) for xi in x]

    x_int = [np.empty(shape=l[i], dtype=int) for i in range(n)]

    cat_x = []
    for i in range(n):
        unique_states = np.sort(np.unique(x[i]))
        m = len(unique_states)
        num = dict(zip(unique_states, np.arange(m)))
        for j in range(l[i]):
            x_int[i][j] = num[x[i][j]]
        cat_x.append(num)

    if np.allclose(l, l[0]):
        x_int = np.array(x_int)

    return x_int, cat_x


def svd_pinv(x):
    """Compute the SVD-based pseudoinverse matrices

    Args:
        x (csc_matrix): The matrix for which to compute the pseudoinverse

    Returns:
        (csc_matrix, ndarray, csc_matrix): If the SVD of `x` is :math:`USV^T`, this function returns :math:`V`, :math:`S^+`, and :math:`U^T`
    """

    x_rank = np.linalg.matrix_rank(x.todense())
    x_svd = svds(x, k=min(x_rank, min(x.shape) - 1))
    # x_oh_svd = svds(x_oh, k=x_oh_rank)

    sv_pinv = x_svd[1]
    zero_sv = np.isclose(sv_pinv, 0)
    sv_pinv[~zero_sv] = 1.0 / sv_pinv[~zero_sv]
    sv_pinv[zero_sv] = 0.0
    x_pinv = [x_svd[2].T, sv_pinv, x_svd[0].T]

    return x_pinv
