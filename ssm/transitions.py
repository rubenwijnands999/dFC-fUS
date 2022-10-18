from warnings import warn
import autograd.numpy as np
from autograd.scipy.special import logsumexp
from autograd import hessian
from ssm.util import ensure_args_are_lists

class Transitions(object):
    def __init__(self, K, D, M=0):
        self.K, self.D, self.M = K, D, M

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def permute(self, perm):
        pass

    def log_prior(self):
        return 0

    def log_transition_matrices(self, data, input, mask, tag):
        raise NotImplementedError

    def transition_matrices(self, data, input, mask, tag):
        return np.exp(self.log_transition_matrices(data, input, mask, tag))

    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        warn("Analytical Hessian is not implemented for this transition class. \
              Optimization via Laplace-EM may be slow. Consider using an \
              alternative posterior and inference method.")
        obj = lambda x, E_zzp1: np.sum(E_zzp1 * self.log_transition_matrices(x, input, mask, tag))
        hess = hessian(obj)
        terms = np.array([-1 * hess(x[None,:], Ezzp1) for x, Ezzp1 in zip(data, expected_joints)])
        return terms

class StationaryTransitions(Transitions):
    """
    Standard Hidden Markov Model with fixed initial distribution and transition matrix.
    """
    def __init__(self, K, D, M=0):
        super(StationaryTransitions, self).__init__(K, D, M=M)
        #Ps = .95 * np.eye(K) + .05 * npr.rand(K, K)
        # Uniform
        Ps = np.ones((K,K))
        Ps /= Ps.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(Ps)

    @property
    def params(self):
        return (self.log_Ps,)

    @params.setter
    def params(self, value):
        self.log_Ps = value[0]

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]

    @property
    def transition_matrix(self):
        return np.exp(self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True))

    def log_transition_matrices(self, data, input, mask, tag):
        log_Ps = self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True)
        return log_Ps[None, :, :]

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        K = self.K
        P = sum([np.sum(Ezzp1, axis=0) for _, Ezzp1, _ in expectations]) + 1e-32
        P = np.nan_to_num(P / P.sum(axis=-1, keepdims=True))

        # Set rows that are all zero to uniform
        P = np.where(P.sum(axis=-1, keepdims=True) == 0, 1.0 / K, P)
        log_P = np.log(P)
        self.log_Ps = log_P - logsumexp(log_P, axis=-1, keepdims=True)

    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        return np.zeros((T-1, D, D))

