# This script contains functions for intermediate experimentation

import numpy as np
from Algs.optimize_weights_intermittent import opt_alphas_intermittent, evaluate_S
from topology import clustered_topology_intermittent_vs_perfect_compare


def compare_clustered_topologies_intermittent_vs_perfect(num_clients: int = 10, cluster1_ang: float = np.pi / 4,
                                                         cluster2_ang: float = np.pi / 2):
    """
    Function to experiment and find a clustered topology with desirable angular orientation of the clusters so that
    intermittent connectivity gives better performance than perfect connectivity
    :param num_clients -- Number of clients
    :param cluster1_ang -- Angular orientation of first cluster
    :param cluster2_ang -- Angular orientation of second cluster
    Returns the S value for perfectly and intermittently connected decentralized topologies
    """

    t_prob, prob_ngbrs, conn_mat = clustered_topology_intermittent_vs_perfect_compare(cluster1_ang=cluster1_ang,
                                                                                      cluster2_ang=cluster2_ang)
    A_intm = opt_alphas_intermittent(transmit_probs=t_prob, ctr_max = 50 * num_clients, prob_ngbrs=prob_ngbrs)
    A_perf = opt_alphas_intermittent(transmit_probs=t_prob, ctr_max=50 * num_clients, prob_ngbrs=conn_mat)

    S_intm = evaluate_S(transmit_probs=t_prob, alphas=A_intm, prob_ngbrs=prob_ngbrs)
    S_perf = evaluate_S(transmit_probs=t_prob, alphas=A_perf, prob_ngbrs=conn_mat)

    print("S-value with perfect connectivity: {}".format(S_perf))
    print("S-value with intermittent connectivity: {}".format(S_intm))


if __name__ == "__main__":

    compare_clustered_topologies_intermittent_vs_perfect(cluster1_ang=2 * np.pi / 3, cluster2_ang=4 * np.pi / 3)

