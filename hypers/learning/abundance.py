import numpy as np
import hypers as hp
from cvxopt import matrix, solvers
import scipy.optimize as opt


class abundance:
    """ Provides instance of abundance classes """
    def __init__(self, X: 'hp.hparray'):
        self.ucls = ucls(X)
        self.nnls = nnls(X)
        self.fcls = fcls(X)


class ucls:
    def __init__(self, X: 'hp.hparray'):
        self.X = X
        self.map = None

    def calculate(self, x_fit: np.ndarray) -> np.ndarray:
        if x_fit.ndim == 1:
            x_fit = x_fit.reshape(x_fit.shape[0], 1)
        x_inverse = np.linalg.pinv(x_fit)
        self.map = np.dot(x_inverse, self.X.collapse().T).T.reshape(self.X.shape[:-1] + (x_fit.shape[-1],))

        return self.map


class nnls:
    def __init__(self, X: 'hp.hparray'):
        self.X = X
        self.map = None

    def calculate(self, x_fit: np.ndarray) -> np.ndarray:
        if x_fit.ndim == 1:
            x_fit = x_fit.reshape(x_fit.shape[0], 1)
        M = self.X.collapse()

        N, p1 = M.shape
        q, p2 = x_fit.T.shape

        self.map = np.zeros((N, q), dtype=np.float32)
        MtM = np.dot(x_fit.T, x_fit)
        for n1 in range(N):
            self.map[n1] = opt.nnls(MtM, np.dot(x_fit.T, M[n1]))[0]
        self.map = self.map.reshape(self.X.shape[:-1] + (x_fit.shape[-1],))

        return self.map


class fcls:
    def __init__(self, X: 'hp.hparray'):
        self.X = X
        self.map = None

    def calculate(self, x_fit: np.ndarray) -> np.ndarray:
        if x_fit.ndim == 1:
            x_fit = x_fit.reshape(x_fit.shape[0], 1)
        solvers.options['show_progress'] = False

        M = self.X.collapse()

        N, p1 = M.shape
        nvars, p2 = x_fit.T.shape
        C = _numpy_to_cvxopt_matrix(x_fit)
        Q = C.T * C

        lb_A = -np.eye(nvars)
        lb = np.repeat(0, nvars)
        A = _numpy_None_vstack(None, lb_A)
        b = _numpy_None_concatenate(None, -lb)
        A = _numpy_to_cvxopt_matrix(A)
        b = _numpy_to_cvxopt_matrix(b)

        Aeq = _numpy_to_cvxopt_matrix(np.ones((1, nvars)))
        beq = _numpy_to_cvxopt_matrix(np.ones(1))

        M = np.array(M, dtype=np.float64)
        self.map = np.zeros((N, nvars), dtype=np.float32)
        for n1 in range(N):
            d = matrix(M[n1], (p1, 1), 'd')
            q = - d.T * C
            sol = solvers.qp(Q, q.T, A, b, Aeq, beq, None, None)['x']
            self.map[n1] = np.array(sol).squeeze()
        self.map = self.map.reshape(self.X.shape[:-1] + (x_fit.shape[-1],))

        return self.map


def _numpy_None_vstack(A1, A2):
    if A1 is None:
        return A2
    else:
        return np.vstack([A1, A2])


def _numpy_None_concatenate(A1, A2):
    if A1 is None:
        return A2
    else:
        return np.concatenate([A1, A2])


def _numpy_to_cvxopt_matrix(A):
    A = np.array(A, dtype=np.float64)
    if A.ndim == 1:
        return matrix(A, (A.shape[0], 1), 'd')
    else:
        return matrix(A, A.shape, 'd')
