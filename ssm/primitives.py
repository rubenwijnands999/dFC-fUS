# Define an autograd extension for HMM normalizer
import autograd.numpy as np
from autograd.scipy.special import logsumexp
from autograd.extend import primitive
from autograd.tracer import getval
from ssm.messages import forward_pass

to_c = lambda arr: np.copy(getval(arr), 'C') if not arr.flags['C_CONTIGUOUS'] else getval(arr)

@primitive
def hmm_normalizer(pi0, Ps, ll):
    T, K = ll.shape
    alphas = np.zeros((T, K))

    # Make sure everything is C contiguous
    pi0 = to_c(pi0)
    Ps = to_c(Ps)
    ll = to_c(ll)

    forward_pass(pi0, Ps, ll, alphas)
    return logsumexp(alphas[-1])
