# This script comprises of modules to implement the weight optimization procedure using Gauss-Seidel iterations
# for minimizing the suboptimality gap in the presence of intermittent connectivity between clients

import numpy as np
import sys
import os

from topology import client_locations_mmWave_clusters, client_locations_mmWave_clusters_perfect_conn, \
    client_locations_mmWave_clusters_intermittent


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def evaluate_S(transmit_probs: np.ndarray = None, alphas: np.ndarray = None, prob_ngbrs: np.ndarray = None):
    """
    Evaluate the value of S.
    :param transmit_probs: Array of transmission probabilities from each of the clients to the PS.
    :param alphas: Matrix of weights
    :param prob_ngbrs: Matrix of probabilities for intermittent connectivity amongst clients
    """

    # The links are assumed to be bidirectional and the transmissions between two nodes are correlated.
    # If the channel is blocked, neither can transmit to the other, and hence \tau_{i \to j} = \tau_{j \to i}
    # This implies that P is symmetric

    num_clients = len(transmit_probs)
    alphas_dim = alphas.shape
    neighbors_dim = prob_ngbrs.shape

    # Rename variables
    p = transmit_probs
    P = prob_ngbrs                  # P[i][j] is the probability of successfully transmitting from node i to node j
    A = alphas

    # Validate inputs
    assert num_clients == alphas_dim[0] == alphas_dim[1]
    assert num_clients == neighbors_dim[0] == neighbors_dim[1]

    # Evaluate S(p, P, A)
    S = 0

    # First term
    for i in range(num_clients):
        for l in range(num_clients):
            for j in range(num_clients):
                S += p[j] * (1 - p[j]) * P[i][j] * P[l][j] * A[j][i] * A[j][l]

    # Second term
    for i in range(num_clients):
        for j in range(num_clients):
            S += P[i][j] * p[j] * (1 - P[i][j]) * A[j][i] * A[j][i]

    # Third term
    for i in range(num_clients):
        for l in range(num_clients):
            assert P[i][l] == P[l][i], "Matrix prob_nbgr must be symmetric."
            E = P[i][l]
            S += p[i] * p[l] * (E - P[i][l] * P[l][i]) * A[i][l] * A[l][i]

    return S


def evaluate_Sbar(transmit_probs: np.ndarray = None, alphas: np.ndarray = None, prob_ngbrs: np.ndarray = None):
    """
    Evaluate the value of S_bar -- convex upper bound on S.
    :param transmit_probs: Array of transmission probabilities from each of the clients to the PS.
    :param alphas: Matrix of weights
    :param prob_ngbrs: Matrix of probabilities for intermittent connectivity amongst clients
    """

    # The links are assumed to be bidirectional and the transmissions between two nodes are correlated.
    # If the channel is blocked, neither can transmit to the other, and hence \tau_{i \to j} = \tau_{j \to i}
    # This implies that P is symmetric

    num_clients = len(transmit_probs)
    alphas_dim = alphas.shape
    neighbors_dim = prob_ngbrs.shape

    # Rename variables
    p = transmit_probs
    P = prob_ngbrs                  # P[i][j] is the probability of successfully transmitting from node i to node j
    A = alphas

    # Validate inputs
    assert num_clients == alphas_dim[0] == alphas_dim[1]
    assert num_clients == neighbors_dim[0] == neighbors_dim[1]

    # Evaluate S(p, P, A)
    Sb = 0

    # First term
    for i in range(num_clients):
        for l in range(num_clients):
            for j in range(num_clients):
                Sb += p[j] * (1 - p[j]) * P[i][j] * P[l][j] * A[j][i] * A[j][l]

    # Second term
    for i in range(num_clients):
        for j in range(num_clients):
            Sb += P[i][j] * p[j] * (1 - P[i][j]) * A[j][i] * A[j][i]

    # Third term
    for i in range(num_clients):
        for l in range(num_clients):
            assert P[i][l] == P[l][i], "Matrix prob_nbgr must be symmetric."
            E = P[i][l]
            Sb += p[i] * p[l] * (E - P[i][l] * P[l][i]) * A[l][i] * A[l][i]

    return Sb


def opt_alphas_global(transmit_probs: np.ndarray = None, ctr_max: int = 200, prob_ngbrs: np.ndarray = None):
    """
    Optimize the value of alphas by minimizing the convex relaxation S_bar
    :param transmit_probs: Array of transmission probabilities from each of the clients.
    :param ctr_max: Maximum number of transmissions.
    :param prob_ngbrs: Matrix of probabilities for intermittent connectivity amongst clients
    """

    num_clients = len(transmit_probs)
    neighbors_dim = prob_ngbrs.shape

    # Rename variables
    p = transmit_probs
    P = prob_ngbrs              # P[i][j] is the probability of successfully transmitting from node i to node j

    # Validate inputs
    assert num_clients == neighbors_dim[0] == neighbors_dim[1]

    # Initialize the weights for Gauss-Seidel method
    A = np.zeros([num_clients, num_clients])
    for i in range(num_clients):

        # Count the number of neighbors of node-i that can successfully relay
        num_ngbrs = 0
        for k in range(num_clients):
            if p[k] > 0 and P[i][k] > 0:
                num_ngbrs += 1

        # Initialize the weight in j-th row and i-th column
        for j in range(num_clients):
            if p[j] > 0 and P[i][j] > 0:
                A[j][i] = 1 / (num_ngbrs * p[j] * P[i][j])

    # Initialize counter
    ctr = 0

    # Print values before optimization
    print("Weights before optimizing:\n {}".format(A))
    S = evaluate_S(transmit_probs=transmit_probs, alphas=A, prob_ngbrs=prob_ngbrs)
    print("Value of S before optimizing: {}".format(S))
    S_bar = evaluate_Sbar(transmit_probs=transmit_probs, alphas=A, prob_ngbrs=prob_ngbrs)
    print("Value of S_bar (convex upper bound) before optimizing: {}".format(S_bar))

    # Gauss-Seidel method
    while ctr < ctr_max:

        # Increment counter
        ctr += 1
        print("ctr = {}".format(ctr))

        # Find index of column to be updated in this iteration
        i = ctr % num_clients
        if i == 0:
            i = num_clients
        i -= 1                      # For zero indexing

        # Compute Lagrange multiplier using bisection search
        # First, define the interval over which you need to do bisection search
        r_min = 0                                       # Initialize lower limit of bisection search
        r_max = 0                                       # Initialize upper limit of bisection search
        for j in range(num_clients):
            if 0 < p[j] * P[i][j] < 1:
                E = P[i][j]                             # Blockage model

                aux_sum = 0
                for l in range(num_clients):
                    if l != i:
                        aux_sum += P[l][j] * A[j][l]

                t = 2 * ((1 - p[j] * P[i][j]) + p[i] * (E / P[i][j] - P[j][i])) / (p[j] * P[i][j]) \
                    + 2 * (1 - p[j]) * aux_sum

                if t > r_max:
                    r_max = t

        # Bisection search over [r_min, r_max]
        tolerance = 1e-5                            # Tolerance to which the constraint value is satisfied
        lambda_i = (r_min + r_max) / 2

        # Initialize weight values in a temporary variable prior to bisection search
        # Do this because you need to iteratively check to ensure constraint value is satisfied
        A_i_temp = np.zeros(num_clients)
        max_prob = 0
        for k in range(num_clients):
            if p[k] * P[i][k] > max_prob:
                max_prob = p[k] * P[i][k]

        for j in range(num_clients):
            if 0 < p[j] * P[i][j] * max_prob < 1:
                aux_sum = 0
                for l in range(num_clients):
                    if l != i:
                        aux_sum += P[l][j] * A[j][l]
                E = P[i][j]                                                 # Blockage model
                A_i_temp[j] = max((-2 * (1 - p[j]) * aux_sum + lambda_i)
                                  / (2 * ((1 - p[j] * P[i][j]) + p[i] * (E / P[i][j] - P[j][i]))), 0)

            elif p[j] * P[i][j] == 1:
                num = 0
                for k in range(num_clients):
                    if p[k] * P[i][k] == 1:
                        num += 1
                A_i_temp[j] = 1 / num

            else:
                A_i_temp[j] = 0

        # Evaluate initial constraint value
        constraint_value = 0
        for j in range(num_clients):
            constraint_value += p[j] * P[i][j] * A_i_temp[j]

        # Do a bisection search
        lagrange_opt_ctr = 0
        while abs(constraint_value - 1) > tolerance:
            lagrange_opt_ctr += 1
            if constraint_value > 1:
                r_max = lambda_i
            elif constraint_value < 1:
                r_min = lambda_i
            lambda_i = (r_min + r_max) / 2

            # Compute updated temporary weights with this new lambda_i values
            A_i_temp = np.zeros(num_clients)
            max_prob = 0
            for k in range(num_clients):
                if p[k] * P[i][k] > max_prob:
                    max_prob = p[k] * P[i][k]

            for j in range(num_clients):
                if 0 < p[j] * P[i][j] * max_prob < 1:
                    aux_sum = 0
                    for l in range(num_clients):
                        if l != i:
                            aux_sum += P[l][j] * A[j][l]
                    E = P[i][j]
                    A_i_temp[j] = max((-2 * (1 - p[j]) * aux_sum + lambda_i)
                                      / (2 * ((1 - p[j] * P[i][j]) + p[i] * (E / P[i][j] - P[j][i]))), 0)

                elif p[j] * P[i][j] == 1:
                    num = 0
                    for k in range(num_clients):
                        if p[k] * P[i][k] == 1:
                            num += 1
                    A_i_temp[j] = 1 / num

                else:
                    A_i_temp[j] = 0

            # Evaluate constraint value with this new set of weights
            constraint_value = 0
            for j in range(num_clients):
                constraint_value += p[j] * P[i][j] * A_i_temp[j]
            print("Lagrange counter: {}, Constraint value: {}".format(lagrange_opt_ctr, constraint_value))

        # Update actual weights using the temporary weights
        A[:, i] = A_i_temp

        # Re-evaluate objective function values
        S_new = evaluate_S(transmit_probs=transmit_probs, alphas=A, prob_ngbrs=P)
        delta_S = S - S_new
        S = S_new

        S_bar_new = evaluate_Sbar(transmit_probs=transmit_probs, alphas=A, prob_ngbrs=P)
        delta_Sbar = S_bar - S_bar_new
        S_bar = S_bar_new

        print("Iteration: " + str(ctr) + ", S = " + str(S) + ", delta_S = " + str(delta_S) +
              ", S_bar = " + str(S_bar) + ", delta_Sbar = " + str(delta_Sbar) + ", Column:" + str(i))

    return A


def opt_alphas_local(transmit_probs: np.ndarray = None, ctr_max: int = 200, alphas_init: np.ndarray = None,
                     prob_ngbrs: np.ndarray = None):
    """
    Locally optimize the value of alphas by implementing Gauss-Seidel on S directly with warm-start
    initialization from the solution of Gauss-Seidel on the convex relaxation
    :param transmit_probs: Array of transmission probabilities from each of the clients.
    :param ctr_max: Maximum number of transmissions.
    :param alphas_init: Warm-start initialization
    :param prob_ngbrs: Matrix of probabilities for intermittent connectivity amongst clients
    """

    num_clients = len(transmit_probs)
    neighbors_dim = prob_ngbrs.shape

    # Rename variables
    p = transmit_probs
    P = prob_ngbrs  # P[i][j] is the probability of successfully transmitting from node i to node j

    # Validate inputs
    assert num_clients == neighbors_dim[0] == neighbors_dim[1]

    # Warm-start initialize the weights for Gauss-Seidel method
    A = np.copy(alphas_init)

    # Initialize counter
    ctr = 0

    # Print value before optimization
    print("Weights before optimizing:\n {}".format(A))
    S = evaluate_S(transmit_probs=transmit_probs, alphas=A, prob_ngbrs=prob_ngbrs)
    print("Value of S before optimizing: {}".format(S))

    # Gauss-Seidel method
    while ctr < ctr_max:

        # Increment counter
        ctr += 1
        print("ctr = {}".format(ctr))

        # Find index of column to be updated in this iteration
        i = ctr % num_clients
        if i == 0:
            i = num_clients
        i -= 1  # For zero indexing

        # Compute Lagrange multiplier using bisection search
        # First, define the interval over which you need to do bisection search
        r_min = 0  # Initialize lower limit of bisection search
        r_max = 0  # Initialize upper limit of bisection search
        for j in range(num_clients):
            if 0 < p[j] * P[i][j] < 1:
                E = P[i][j]                             # Blockage model

                aux_sum = 0
                for l in range(num_clients):
                    if l != i:
                        aux_sum += P[l][j] * A[j][l]

                t = 2 * (1 - p[j] * P[i][j]) / (p[j] * P[i][j]) + 2 * (1 - p[j]) * aux_sum \
                    + 2 * p[i] * (E / P[i][j] - P[j][i]) * A[i][j]

                if t > r_max:
                    r_max = t

        # Bisection search over [r_min, r_max]
        tolerance = 1e-5                                # Tolerance to which the constraint value is satisfied
        lambda_i = (r_min + r_max) / 2

        # Initialize weight values in a temporary variable prior to bisection search
        # Do this because you need to iteratively check to ensure constraint value is satisfied
        A_i_temp = np.zeros(num_clients)
        max_prob = 0
        for k in range(num_clients):
            if p[k] * P[i][k] > max_prob:
                max_prob = p[k] * P[i][k]

        for j in range(num_clients):
            if 0 < p[j] * P[i][j] * max_prob < 1:
                aux_sum = 0
                for l in range(num_clients):
                    if l != i:
                        aux_sum += P[l][j] * A[j][l]
                E = P[i][j]                                         # Blockage model
                A_i_temp[j] = max((-2 * (1 - p[j]) * aux_sum - 2 * p[i] *
                                   (E / P[i][j] - P[j][i]) * A[i][j] + lambda_i) / (2 * (1 - p[j] * P[i][j])), 0)

            elif p[j] * P[i][j] == 1:
                num = 0
                for k in range(num_clients):
                    if p[k] * P[i][k] == 1:
                        num += 1
                A_i_temp[j] = 1 / num

            else:
                A_i_temp[j] = 0

        # Evaluate initial constraint value
        constraint_value = 0
        for j in range(num_clients):
            constraint_value += p[j] * P[i][j] * A_i_temp[j]

        # Do a bisection search
        lagrange_opt_ctr = 0
        while abs(constraint_value - 1) > tolerance:
            lagrange_opt_ctr += 1
            if constraint_value > 1:
                r_max = lambda_i
            elif constraint_value < 1:
                r_min = lambda_i
            lambda_i = (r_min + r_max) / 2

            # Compute updated temporary weights with this new lambda_i values
            A_i_temp = np.zeros(num_clients)
            max_prob = 0
            for k in range(num_clients):
                if p[k] * P[i][k] > max_prob:
                    max_prob = p[k] * P[i][k]

            for j in range(num_clients):
                if 0 < p[j] * P[i][j] * max_prob < 1:
                    aux_sum = 0
                    for l in range(num_clients):
                        if l != i:
                            aux_sum += P[l][j] * A[j][l]
                    E = P[i][j]  # Blockage model
                    A_i_temp[j] = max((-2 * (1 - p[j]) * aux_sum - 2 * p[i] *
                                       (E / P[i][j] - P[j][i]) * A[i][j] + lambda_i) / (2 * (1 - p[j] * P[i][j])), 0)

                elif p[j] * P[i][j] == 1:
                    num = 0
                    for k in range(num_clients):
                        if p[k] * P[i][k] == 1:
                            num += 1
                    A_i_temp[j] = 1 / num

                else:
                    A_i_temp[j] = 0

            # Evaluate constraint value with this new set of weights
            constraint_value = 0
            for j in range(num_clients):
                constraint_value += p[j] * P[i][j] * A_i_temp[j]
            print("Lagrange counter: {}, Constraint value: {}".format(lagrange_opt_ctr, constraint_value))

        # Update actual weights using the temporary weights
        A[:, i] = A_i_temp

        # Re-evaluate objective function values
        S_new = evaluate_S(transmit_probs=transmit_probs, alphas=A, prob_ngbrs=P)
        delta_S = S - S_new
        S = S_new

        # S_bar_new = evaluate_Sbar(transmit_probs=transmit_probs, alphas=A, prob_ngbrs=P)
        # delta_Sbar = S_bar - S_bar_new
        # S_bar = S_bar_new

        # print("Iteration: " + str(ctr) + ", S = " + str(S) + ", delta_S = " + str(delta_S) +
        #       ", S_bar = " + str(S_bar) + ", delta_Sbar = " + str(delta_Sbar) + ", Column:" + str(i))
        print("Iteration: " + str(ctr) + ", S = " + str(S) + ", delta_S = " + str(delta_S) + ", Column:" + str(i))

    return A


def opt_alphas_intermittent(transmit_probs: np.ndarray = None, ctr_max: int = 200, prob_ngbrs: np.ndarray = None):
    """
    Concatenation of opt_alphas_global and opt_alphas_local
    :param transmit_probs: Array of transmission probabilities from each of the clients.
    :param ctr_max: Maximum number of transmissions.
    :param prob_ngbrs: Matrix of probabilities for intermittent connectivity amongst clients
    """

    # Get the solution of the global optimization

    blockPrint()
    A_ws = opt_alphas_global(transmit_probs=transmit_probs, ctr_max=ctr_max, prob_ngbrs=prob_ngbrs)
    A_opt = opt_alphas_local(transmit_probs=transmit_probs, ctr_max=ctr_max, prob_ngbrs=prob_ngbrs, alphas_init=A_ws)
    enablePrint()

    return A_opt


if __name__ == "__main__":

    # t_prob, P = client_locations_mmWave_clusters()
    # A = np.diag(1 / t_prob)
    # num_clients = len(t_prob)
    #
    # print(t_prob)
    # print(P)
    # print(A)
    # P = np.random.uniform(0, 1, 10) * P
    # P = (P + np.transpose(P)) / 2
    #
    # S_val = evaluate_S(transmit_probs=t_prob, alphas=A, prob_ngbrs=P)
    # print("S_val = {}".format(S_val))
    # S_bar_val = evaluate_Sbar(transmit_probs=t_prob, alphas=A, prob_ngbrs=P)
    # print("S_bar_val = {}".format(S_bar_val))
    #
    # A_opt = opt_alphas_global(transmit_probs=t_prob, prob_ngbrs=P)
    # S_val_opt = evaluate_S(transmit_probs=t_prob, alphas=A_opt, prob_ngbrs=P)
    # print("S_val_opt = {}".format(S_val_opt))
    # S_bar_val_opt = evaluate_Sbar(transmit_probs=t_prob, alphas=A_opt, prob_ngbrs=P)
    # print("S_bar_val_opt = {}".format(S_bar_val_opt))
    #
    # A_opt_local = opt_alphas_local(transmit_probs=t_prob, prob_ngbrs=P, alphas_init=A_opt)
    # print("Before local optimization, S_val_opt = {}".format(S_val_opt))
    # S_val_opt_local = evaluate_S(transmit_probs=t_prob, alphas=A_opt_local, prob_ngbrs=P)
    # print("After local optimization, S_val_opt_local = {}".format(S_val_opt_local))
    #
    # # Directly applying Gauss-Seidel procedure to the non-convex problem
    # # Initialize
    # A_init = np.zeros([num_clients, num_clients])
    # for i in range(num_clients):
    #
    #     # Count the number of neighbors of node-i that can successfully relay
    #     num_ngbrs = 0
    #     for k in range(num_clients):
    #         if t_prob[k] > 0 and P[i][k] > 0:
    #             num_ngbrs += 1
    #
    #     # Initialize the weight in j-th row and i-th column
    #     for j in range(num_clients):
    #         if t_prob[j] > 0 and P[i][j] > 0:
    #             A_init[j][i] = 1 / (num_ngbrs * t_prob[j] * P[i][j])
    #
    # A_opt_local_direct = opt_alphas_local(transmit_probs=t_prob, prob_ngbrs=P, alphas_init=A_init)
    # A_opt_local = opt_alphas_local(transmit_probs=t_prob, prob_ngbrs=P, alphas_init=A_opt)
    # A_concat = opt_alphas_intermittent(transmit_probs=t_prob, prob_ngbrs=P)
    #
    # S_val_opt = evaluate_S(transmit_probs=t_prob, alphas=A_opt, prob_ngbrs=P)
    # print("After global, before local optimization, S_val_opt = {}".format(S_val_opt))
    # S_val_opt_local = evaluate_S(transmit_probs=t_prob, alphas=A_opt_local, prob_ngbrs=P)
    # print("After global and local optimization, S_val_opt_local = {}".format(S_val_opt_local))
    #
    # S_val_init = evaluate_S(transmit_probs=t_prob, alphas=A_init, prob_ngbrs=P)
    # print("Before any optimization, S_val_init = {}".format(S_val_init))
    # S_val_ncvx_direct = evaluate_S(transmit_probs=t_prob, alphas=A_opt_local_direct, prob_ngbrs=P)
    # print("After direct GS on ncvx optimization, S_val_ncvx_direct = {}".format(S_val_ncvx_direct))
    # S_val_concat = evaluate_S(transmit_probs=t_prob, alphas=A_concat, prob_ngbrs=P)
    # print("After concatenated application of global + local optimization, "
    #       "S_val_concat = {}".format(S_val_concat))

    # transmit_p, ngbr_matrix = client_locations_mmWave_clusters_perfect_conn(num_clients=10)
    # print("transmit_p = {}".format(transmit_p))
    # print("neighbor_matrix = {}".format(ngbr_matrix))
    #
    # A = np.diag(1 / transmit_p)
    # S_val_init = evaluate_S(transmit_probs=transmit_p, alphas=A, prob_ngbrs=ngbr_matrix)
    #
    # A_opt = opt_alphas_intermittent(transmit_probs=transmit_p, prob_ngbrs=ngbr_matrix, ctr_max=50 * 10)
    # S_val_opt = evaluate_S(transmit_probs=transmit_p, alphas=A_opt, prob_ngbrs=ngbr_matrix)
    #
    # print("S_val_init = {}".format(S_val_init))
    # print("S_val_opt = {}".format(S_val_opt))
    #
    # print("A_opt = \n{}".format(A_opt))

    # Observing weight matrices for perfect vs. intermittent connectivity in topology 1,
    # when ColRel - perfect doesn't converge, but FedAvg (blind and non-blind do)

    print("FOR PERFECT CONNECTIVITY:")
    transmit_p, ngbr_matrix = client_locations_mmWave_clusters_perfect_conn(num_clients=10)
    print("transmit_p = {}".format(transmit_p))
    print("neighbor_matrix = \n{}".format(ngbr_matrix))

    A = np.diag(1 / transmit_p)
    S_val_init = evaluate_S(transmit_probs=transmit_p, alphas=A, prob_ngbrs=ngbr_matrix)

    A_opt = opt_alphas_intermittent(transmit_probs=transmit_p, prob_ngbrs=ngbr_matrix, ctr_max=50 * 10)
    S_val_opt = evaluate_S(transmit_probs=transmit_p, alphas=A_opt, prob_ngbrs=ngbr_matrix)

    print("S_val_init = {}".format(S_val_init))
    print("S_val_opt = {}".format(S_val_opt))

    print("A_opt = \n{}".format(A_opt))

    print("Constraint values for perfect connectivity:\n")
    for i in range(10):
        final_constraint_val = 0
        for j in range(10):
            final_constraint_val += transmit_p[j] * ngbr_matrix[i][j] * A_opt[j][i]
        print("Checking unbiasedness for client-{}. Value:{}".format(i, final_constraint_val))

    print("FOR INTERMITTENT CONNECTIVITY:")
    transmit_p, P, ngbr_matrix = client_locations_mmWave_clusters_intermittent(num_clients=10)
    print("transmit_p = {}".format(transmit_p))
    print("P = \n{}".format(P))
    print("neighbor_matrix = \n{}".format(ngbr_matrix))

    A = np.diag(1 / transmit_p)
    S_val_init = evaluate_S(transmit_probs=transmit_p, alphas=A, prob_ngbrs=P)

    A_opt = opt_alphas_intermittent(transmit_probs=transmit_p, prob_ngbrs=P, ctr_max=50 * 10)
    S_val_opt = evaluate_S(transmit_probs=transmit_p, alphas=A_opt, prob_ngbrs=P)

    print("S_val_init = {}".format(S_val_init))
    print("S_val_opt = {}".format(S_val_opt))

    print("A_opt = \n{}".format(A_opt))

    print("Constraint values for intermittent connectivity:\n")
    for i in range(10):
        final_constraint_val = 0
        for j in range(10):
            final_constraint_val += transmit_p[j] * P[i][j] * A_opt[j][i]
        print("Checking unbiasedness for client-{}. Value:{}".format(i, final_constraint_val))





