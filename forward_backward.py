# coding: utf8

"""
the forward-backward algorithm for Hidden Markov Model
"""


states = ("s1", "s2")
observations = ("o1", "o1", "o2", "o1", "o1")
start_probabilities = {"s1": 0.5, "s2": 0.5}
transition_matrix = {
    "s1": {"s1": 0.7, "s2": 0.3},
    "s2": {"s1": 0.3, "s2": 0.7},
}
emission_matrix = {
    "s1": {"o1": 0.9, "o2": 0.1},
    "s2": {"o1": 0.2, "o2": 0.8},
}


def forward_backward(obs, states, start_p, trans_m, emit_m):
    """
    introduce some param in the formula we use below:
    in general, "t" means a centain point in the timeline, "T" means the length of the timeline.
    alpha is a normalization constant to make the sum of all the probabilities to 1, usually 1 / sum(all the probabilities), 
    we can omit it in forward/backward algorithm because they are not ultimately what we want, but the smoothed value is
    """

    # run forward algorithm, using markov property, we get:
    # P(State_t | Observation_0:t) = alpha * P(Observation_t | State_t) * sum(P(State_t | State_t-1) * P(State_t-1 | Observation_0:t-1) for all State in timeline t-1)
    forward_probs = []
    f_0_to_pre_t = {}
    for i in xrange(len(obs)):
        f_0_to_t = {}
        for s in states:
            if i == 0:
                # when it's in timeline 0, P(State_0 | Observation_0:0) can be the prior probability of the state
                f_0_to_t[s] = emit_m[s][obs[i]] * start_p[s]
            else:
                # using the recursive formula f_1:t = P(State_t | Observation_0:t), compute f_0:t with f_0:t-1
                # here, s0 means a state occur before s
                f_0_to_t[s] = emit_m[s][obs[i]] * sum(trans_m[s0][s] * f_0_to_pre_t[s0] for s0 in states)

        forward_probs.append(f_0_to_t)
        f_0_to_pre_t = f_0_to_t

    forward_probs = probs_normalize(forward_probs)
    forward_probs.insert(0, start_p)

    # run backward, k required (1 <= k < t)
    # P(Observation_k+1:t | State_k) = sum(P(Observation_k+1 | State_k+1) * P(Observation_k+2:t | State_k+1) * P(State_k+1 | State_k) for all state in timeline k+1)
    backward_probs = []
    b_post_k_to_t = {}
    for i in xrange(len(obs), -1, -1):
        b_k_to_t = {}
        for s in states:
            if i == len(obs):
                # here, i = T(the length of the sequence and exceed the max index t-1), 
                # and then Observation_t:t-1 becomes a null sequence, its probability is 1
                b_k_to_t[s] = 1.0
            else:
                # using the recursive formula b_k+1:t = P(Observation_k+1:t | State_k), compute b_k+1:t with b_k+2:t
                # here, s1 means a state occur after s
                b_k_to_t[s] = sum(emit_m[s1][obs[i]] * b_post_k_to_t[s1] * trans_m[s][s1] for s1 in states)
            
        backward_probs.insert(0, b_k_to_t)
        b_post_k_to_t = b_k_to_t

    backward_probs = probs_normalize(backward_probs)
    backward_probs.append(dict((s, 1.0) for s in states))
        
    # compute the smoothed probability values
    smoothed_probs = []
    for i in xrange(len(obs)+1):
        v = dict((s, forward_probs[i][s] * backward_probs[i][s]) for s in states)
        smoothed_probs.append(v)

    smoothed_probs = probs_normalize(smoothed_probs)

    return forward_probs, backward_probs, smoothed_probs


def probs_normalize(probs):
    return map(lambda x: dict((k, float(v)/sum(x.itervalues())) for k, v in x.iteritems()), probs)


if __name__ == "__main__":
    f, b, s = forward_backward(observations, states, start_probabilities, transition_matrix, emission_matrix)
    print "\n".join(map(str, (f, b, s)))
