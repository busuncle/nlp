# coding: utf8

"""
the forward-backward algorithm for Hidden Markov Model
"""


states = ("s1", "s2")
end_state = "s2"
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


def forward_backward(obs, states, start_p, trans_m, emit_m, end_stat):
    """
    in general, "t" means a centain point in the timeline, "T" means the length of the timeline
    """

    # run forward
    forward_probs = []
    f_0_to_pre_t = {}
    for i in xrange(len(obs)):
        f_0_to_t = {}
        for s in states:
            p_sums = dict((s0, emit_m[s][obs[i]] * trans_m[s0][s]) for s0 in states)
            if i == 0:
                f_0_to_t[s] = sum(p_sums[k] * start_p[k] for k in p_sums)
            else:
                f_0_to_t[s] = sum(p_sums[k] * f_0_to_pre_t[k] for k in p_sums)

        forward_probs.append(f_0_to_t)
        f_0_to_pre_t = f_0_to_t

    # run backward
    backward_probs = []
    b_post_t_to_T = {}
    for i in xrange(len(obs), 0, -1):
        b_t_to_T = {}
        for s in states:
            if i == len(obs):
                b_t_to_T[s] = trans_m[s][end_state]
            else:
                b_t_to_T[s] = sum(trans_m[s][s0] * emit_m[s0][obs[i]] * b_post_t_to_T[s0] for s0 in states)
            
        backward_probs.insert(0, b_t_to_T)
        b_post_t_to_T = b_t_to_T
        
    # compute the smoothed probability values
    scale_num = sum(forward_probs[-1][k] * trans_m[k][end_state] for k in states)
    posterior = {}
    for s in states:
        posterior[s] = [forward_probs[i][s] * backward_probs[i][s] / scale_num for i in xrange(len(obs))]

    return forward_probs, backward_probs, posterior



if __name__ == "__main__":
    print "\n".join(map(str, forward_backward(observations, states, start_probabilities, transition_matrix, emission_matrix, end_state)))

