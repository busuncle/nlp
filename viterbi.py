# coding: utf8

"""
a demo for learning Hidden Markov Model(HMM) and Viterbi algorithm
"""



states = ("s1", "s2")
observations = ("o1", "o2", "o3")
start_probabilities = {"s1": 0.6, "s2": 0.4}
transition_matrix = {
    "s1": {"s1": 0.7, "s2": 0.3},
    "s2": {"s1": 0.4, "s2": 0.6},
}
emission_matrix = {
    "s1": {"o1": 0.5, "o2": 0.4, "o3": 0.1},
    "s2": {"o1": 0.1, "o2": 0.3, "o3": 0.6},
}


def viterbi(obs, stat, start_p, trans_m, emit_m):
    """
    the two to-be-fill matrix, one contains optimal probability, 
    the other contains optimal state(a backward pointer for constructing the viterbi path)

            S1      S2      ...     Sn
    O(t=0)  P01     P02     ...     P0n
    O(t=1)  P11     P12     ...     P1n
                            ...
    O(t=m)  Pm1     Pm2     ...     Pmn

            S1      S2      ...     Sn
    O(t=0)  S01     S02     ...     S0n
    O(t=1)  S11     S12     ...     S1n
                            ...
    O(t=m)  Sm1     Sm2     ...     Smn
    """
    prob_matrix = [{}]
    stat_matrix = [{}]

    # init start status
    for s in stat:
        prob_matrix[0][s] = start_p[s] * emit_m[s][obs[0]]
        stat_matrix[0][s] = s

    # using dynamic programming method to fill the two matrix in time sequence t1, t2, ..., tn
    for t in xrange(1, len(obs)):
        prob_matrix.append({})
        stat_matrix.append({})

        for s in stat:
            optimal_prob, optimal_stat = max(
                [(prob_matrix[t - 1][si] * trans_m[si][s] * emit_m[s][obs[t]], si) for si in stat])
            # optimal probability for state s in time t
            prob_matrix[t][s] = optimal_prob
            # backward pointer for optimal_stat
            stat_matrix[t][s] = optimal_stat

    # find the optimal path with backward pointer, state at t=0 is an initial value and will be ignored
    path = [max(prob_matrix[-1].iteritems(), key=lambda x:x[1])[0]]
    for i in xrange(len(obs) - 1, 0, -1):
        pre_state = stat_matrix[i][path[-1]]
        path.append(pre_state)

    return path[::-1]



if __name__ == "__main__":
    res = viterbi(observations, states, start_probabilities, transition_matrix, emission_matrix)
    print res

