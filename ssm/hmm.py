import autograd.numpy as np
import autograd.numpy.random as npr
import ssm.observations as obs
import ssm.transitions as trans
import ssm.init_state_distns as isd
import ssm.hierarchical as hier

from ssm.primitives import hmm_normalizer
from ssm.messages import hmm_expected_states, hmm_filter, hmm_sample, viterbi
from ssm.util import ensure_args_are_lists, ensure_args_not_none, ssm_pbar

__all__ = ['HMM']

class HMM(object):
    """
    Base class for hidden Markov models.

    Notation:
    K: number of discrete latent states
    D: dimensionality of observations
    M: dimensionality of inputs

    In the code we will sometimes refer to the discrete
    latent state sequence as z and the data as x.
    """
    def __init__(self, K, D, M=0, init_state_distn=None,
                 transitions='standard',
                 transition_kwargs=None,
                 hierarchical_transition_tags=None,
                 observations="gaussian", observation_kwargs=None,
                 hierarchical_observation_tags=None, **kwargs):

        # Make the initial state distribution
        if init_state_distn is None:
            init_state_distn = isd.InitialStateDistribution(K, D, M=M)
        if not isinstance(init_state_distn, isd.InitialStateDistribution):
            raise TypeError("'init_state_distn' must be a subclass of"
                            " ssm.init_state_distns.InitialStateDistribution.")

        # Make the transition model
        transition_classes = dict(
            standard=trans.StationaryTransitions,
            stationary=trans.StationaryTransitions
            )

        if isinstance(transitions, str):
            if transitions not in transition_classes:
                raise Exception("Invalid transition model: {}. Must be one of {}".
                    format(transitions, list(transition_classes.keys())))

            transition_kwargs = transition_kwargs or {}
            transitions = \
                hier.HierarchicalTransitions(transition_classes[transitions], K, D, M=M,
                                        tags=hierarchical_transition_tags,
                                        **transition_kwargs) \
                if hierarchical_transition_tags is not None \
                else transition_classes[transitions](K, D, M=M, **transition_kwargs)
        if not isinstance(transitions, trans.Transitions):
            raise TypeError("'transitions' must be a subclass of"
                            " ssm.transitions.Transitions")

        # This is the master list of observation classes.
        # When you create a new observation class, add it here.
        observation_classes = dict(gaussian=obs.GaussianObservations)

        if isinstance(observations, str):
            observations = observations.lower()
            if observations not in observation_classes:
                raise Exception("Invalid observation model: {}. Must be one of {}".
                    format(observations, list(observation_classes.keys())))

            observation_kwargs = observation_kwargs or {}
            observations = \
                hier.HierarchicalObservations(observation_classes[observations], K, D, M=M,
                                        tags=hierarchical_observation_tags,
                                        **observation_kwargs) \
                if hierarchical_observation_tags is not None \
                else observation_classes[observations](K, D, M=M, **observation_kwargs)
        if not isinstance(observations, obs.Observations):
            raise TypeError("'observations' must be a subclass of"
                            " ssm.observations.Observations")

        self.K, self.D, self.M = K, D, M
        self.init_state_distn = init_state_distn
        self.transitions = transitions
        self.observations = observations

    @property
    def params(self):
        return self.init_state_distn.params, \
               self.transitions.params, \
               self.observations.params

    @params.setter
    def params(self, value):
        self.init_state_distn.params = value[0]
        self.transitions.params = value[1]
        self.observations.params = value[2]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, init_method="random"):
        """
        Initialize parameters given data.
        """
        self.init_state_distn.initialize(datas, inputs=inputs, masks=masks, tags=tags)
        self.transitions.initialize(datas, inputs=inputs, masks=masks, tags=tags)
        self.observations.initialize(datas, inputs=inputs, masks=masks, tags=tags, init_method=init_method)

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        assert np.all(np.sort(perm) == np.arange(self.K))
        self.init_state_distn.permute(perm)
        self.transitions.permute(perm)
        self.observations.permute(perm)

    def sample(self, T, prefix=None, input=None, tag=None, with_noise=True):
        """
        Sample synthetic data from the model. Optionally, condition on a given
        prefix (preceding discrete states and data).

        Parameters
        ----------
        T : int
            number of time steps to sample

        prefix : (zpre, xpre)
            Optional prefix of discrete states (zpre) and continuous states (xpre)
            zpre must be an array of integers taking values 0...num_states-1.
            xpre must be an array of the same length that has preceding observations.

        input : (T, input_dim) array_like
            Optional inputs to specify for sampling

        tag : object
            Optional tag indicating which "type" of sampled data

        with_noise : bool
            Whether or not to sample data with noise.

        Returns
        -------
        z_sample : array_like of type int
            Sequence of sampled discrete states

        x_sample : (T x observation_dim) array_like
            Array of sampled data
        """
        K = self.K
        D = (self.D,) if isinstance(self.D, int) else self.D
        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(D, tuple)
        assert isinstance(M, tuple)
        assert T > 0

        # Check the inputs
        if input is not None:
            assert input.shape == (T,) + M

        # Get the type of the observations
        if isinstance(self.observations, obs.InputDrivenObservations):
            dtype = int
        else:
            dummy_data = self.observations.sample_x(0, np.empty(0, ) + D)
            dtype = dummy_data.dtype

        # fit the data array
        if prefix is None:
            # No prefix is given.  Sample the initial state as the prefix.
            pad = 1
            z = np.zeros(T, dtype=int)
            data = np.zeros((T,) + D, dtype=dtype)
            input = np.zeros((T,) + M) if input is None else input
            mask = np.ones((T,) + D, dtype=bool)

            # Sample the first state from the initial distribution
            pi0 = self.init_state_distn.initial_state_distn
            z[0] = npr.choice(self.K, p=pi0)
            data[0] = self.observations.sample_x(z[0], data[:0], input=input[0], with_noise=with_noise)

            # We only need to sample T-1 datapoints now
            T = T - 1

        else:
            # Check that the prefix is of the right type
            zpre, xpre = prefix
            pad = len(zpre)
            assert zpre.dtype == int and zpre.min() >= 0 and zpre.max() < K
            assert xpre.shape == (pad,) + D

            # Construct the states, data, inputs, and mask arrays
            z = np.concatenate((zpre, np.zeros(T, dtype=int)))
            data = np.concatenate((xpre, np.zeros((T,) + D, dtype)))
            input = np.zeros((T+pad,) + M) if input is None else np.concatenate((np.zeros((pad,) + M), input))
            mask = np.ones((T+pad,) + D, dtype=bool)

        # Fill in the rest of the data
        for t in range(pad, pad+T):
            Pt = self.transitions.transition_matrices(data[t-1:t+1], input[t-1:t+1], mask=mask[t-1:t+1], tag=tag)[0]
            z[t] = npr.choice(self.K, p=Pt[z[t-1]])
            data[t] = self.observations.sample_x(z[t], data[:t], input=input[t], tag=tag,
                                                 with_noise=with_noise)
        print(Pt)
        # Return the whole data if no prefix is given.
        # Otherwise, just return the simulated part.
        if prefix is None:
            return z, data
        else:
            return z[pad:], data[pad:]

    @ensure_args_not_none
    def expected_states(self, data, input=None, mask=None, tag=None):
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transitions.transition_matrices(data, input, mask, tag)
        log_likes = self.observations.log_likelihoods(data, input, mask, tag)
        return hmm_expected_states(pi0, Ps, log_likes) # gives expected_states, expected_joints, normalizer

    @ensure_args_not_none
    def most_likely_states(self, data, input=None, mask=None, tag=None):
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transitions.transition_matrices(data, input, mask, tag)
        log_likes = self.observations.log_likelihoods(data, input, mask, tag)
        return viterbi(pi0, Ps, log_likes)

    @ensure_args_not_none
    def filter(self, data, input=None, mask=None, tag=None):
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transitions.transition_matrices(data, input, mask, tag)
        log_likes = self.observations.log_likelihoods(data, input, mask, tag)
        return hmm_filter(pi0, Ps, log_likes)

    @ensure_args_not_none
    def smooth(self, data, input=None, mask=None, tag=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        Ez, _, _ = self.expected_states(data, input, mask)
        return self.observations.smooth(Ez, data, input, tag)

    def log_prior(self):
        """
        Compute the log prior probability of the model parameters
        """
        return self.init_state_distn.log_prior() + \
               self.transitions.log_prior() + \
               self.observations.log_prior()

    @ensure_args_are_lists
    def log_likelihood(self, datas, inputs=None, masks=None, tags=None):
        """
        Compute the log probability of the data under the current
        model parameters.

        :param datas: single array or list of arrays of data.
        :return total log probability of the data.
        """
        ll = 0
        for data, input, mask, tag in zip(datas, inputs, masks, tags):
            pi0 = self.init_state_distn.initial_state_distn
            Ps = self.transitions.transition_matrices(data, input, mask, tag)
            log_likes = self.observations.log_likelihoods(data, input, mask, tag)
            ll += hmm_normalizer(pi0, Ps, log_likes)
            assert np.isfinite(ll)
        return ll

    @ensure_args_are_lists
    def log_probability(self, datas, inputs=None, masks=None, tags=None):
        return self.log_likelihood(datas, inputs, masks, tags) + self.log_prior()

    def expected_log_likelihood(
            self, expectations, datas, inputs=None, masks=None, tags=None):
        """
        Compute log-likelihood given current model parameters.

        :param datas: single array or list of arrays of data.
        :return total log probability of the data.
        """
        ell = 0.0
        for (Ez, Ezzp1, _), data, input, mask, tag in \
                zip(expectations, datas, inputs, masks, tags):

            pi0 = self.init_state_distn.initial_state_distn
            log_Ps = self.transitions.log_transition_matrices(data, input, mask, tag)
            log_likes = self.observations.log_likelihoods(data, input, mask, tag)

            ell += np.sum(Ez[0] * np.log(pi0))
            ell += np.sum(Ezzp1 * log_Ps)
            ell += np.sum(Ez * log_likes)
            assert np.isfinite(ell)

        return ell

    def expected_log_probability(
            self, expectations, datas, inputs=None, masks=None, tags=None):
        """
        Compute the log-probability of the data given current
        model parameters.
        """
        ell = self.expected_log_likelihood(
            expectations, datas, inputs=inputs, masks=masks, tags=tags)
        return ell + self.log_prior()


    def _fit_em(self, datas, inputs, masks, tags, verbose = 2, num_iters=100, tolerance=0,
                init_state_mstep_kwargs={},
                transitions_mstep_kwargs={},
                observations_mstep_kwargs={},
                **kwargs):
        """
        Fit the parameters with expectation maximization.

        E step: compute E[z_t] and E[z_t, z_{t+1}] with message passing;
        M-step: analytical maximization of E_{p(z | x)} [log p(x, z; theta)].
        """
        lls = [self.log_probability(datas, inputs, masks, tags)]

        pbar = ssm_pbar(num_iters, verbose, "LP: {:.1f}", [lls[-1]])

        for itr in pbar:
            # E step: compute expected latent states with current parameters
            expectations = [self.expected_states(data, input, mask, tag)
                            for data, input, mask, tag,
                            in zip(datas, inputs, masks, tags)] #expectations consist of expected_states, expected_joints, normalizer

            # M step: maximize expected log joint wrt parameters
            self.init_state_distn.m_step(expectations, datas, inputs, masks, tags, **init_state_mstep_kwargs)
            self.transitions.m_step(expectations, datas, inputs, masks, tags, **transitions_mstep_kwargs)
            self.observations.m_step(expectations, datas, inputs, masks, tags, **observations_mstep_kwargs)

            # Store progress
            lls.append(self.log_prior() + sum([ll for (_, _, ll) in expectations]))

        return lls

    @ensure_args_are_lists
    def fit(self, datas, inputs=None, masks=None, tags=None,
            verbose=2, method="em",
            initialize=True,
            init_method="random",
            **kwargs):

        _fitting_methods = \
            dict(em=self._fit_em)

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".
                            format(method, _fitting_methods.keys()))

        if initialize:
            self.initialize(datas,
                            inputs=inputs,
                            masks=masks,
                            tags=tags,
                            init_method=init_method)

        return _fitting_methods[method](datas,
                                        inputs=inputs,
                                        masks=masks,
                                        tags=tags,
                                        verbose=verbose,
                                        **kwargs)


