# This script simulates distributed mean estimation with intermittent connectivity

import os
import numpy as np
import random
import matplotlib.pyplot as plt
from Algs.optimize_weights_intermittent import opt_alphas_intermittent
from topology import client_locations_mmWave_clusters, fully_connected_topology, \
    clustered_topology_intermittent_vs_perfect_compare


def dme_intermittent_naive(transmit_probs: np.ndarray = None, client_data: np.ndarray = None):
    """
        Simulates distributed mean estimation without collaboration amongst clients
        :param transmit_probs: Array of transmission probabilities from each of the clients.
        :param client_data: matrix containing the vector for each of the clients
        -- each row contains data for a specific client
        Returns the mean estimated at the parameter server
        """

    # Renaming variables
    p = transmit_probs  # Transmission probability to the PS
    X = client_data
    num_clients = len(transmit_probs)  # Number of clients
    dim = client_data.shape[1]  # Dimension of each vector

    assert num_clients == client_data.shape[0], \
        "prob_ngbrs and client_data should have dimensions consistent with the number of clients!"

    transmits = [transmit_probs[i] >= random.uniform(0, 1) for i in range(num_clients)]

    mean_est = np.zeros(dim)
    for i in range(num_clients):
        if transmits[i]:
            mean_est += client_data[i]
    mean_est /= num_clients

    return mean_est


def dme_intermittent_colab(transmit_probs: np.ndarray = None, prob_ngbrs: np.ndarray = None,
                           client_data: np.ndarray = None):
    """
    Simulates distributed mean estimation with collaboration amongst clients
    :param transmit_probs: Array of transmission probabilities from each of the clients.
    :param prob_ngbrs: Matrix of probabilities for intermittent connectivity amongst clients
    :param client_data: matrix containing the vector for each of the clients
    -- each row contains data for a specific client
    Returns the mean estimated at the parameter server
    """

    # Renaming variables
    p = transmit_probs                            # Transmission probability to the PS
    P = prob_ngbrs                                # (Intermittent) topology between clients
    X = client_data
    num_clients = len(transmit_probs)             # Number of clients
    dim = client_data.shape[1]                    # Dimension of each vector

    assert num_clients == prob_ngbrs.shape[0] == prob_ngbrs.shape[1] == client_data.shape[0], \
        "prob_ngbrs and client_data should have dimensions consistent with the number of clients!"

    # Generate a random realization of connectivity amongst clients according to blockage model
    transmit_clients_colab = np.zeros([num_clients, num_clients])
    for i in range(num_clients):
        transmit_clients_colab[i][i] = 1
        for j in range(i+1, num_clients):
            transmit_clients_colab[i][j] = prob_ngbrs[i][j] >= random.uniform(0, 1)
            transmit_clients_colab[j][i] = transmit_clients_colab[i][j]

    # Assert that transmit_clients_colab is symmetric
    assert np.array_equal(transmit_clients_colab, np.transpose(transmit_clients_colab)), \
        "Some error in intermittent connectivity generation!"

    # Optimize weights
    A = opt_alphas_intermittent(transmit_probs=p, ctr_max=10 * num_clients, prob_ngbrs=P)

    # Compute vector to transmit at each client
    Y = np.zeros_like(X)
    for i in range(num_clients):
        for j in range(num_clients):
            if transmit_clients_colab[j][i]:
                Y[i] += A[i][j] * X[j]

    # Compute the mean of locally averaged variables
    # Get a random realization of the connectivity of clients to the PS
    transmits = [p[i] >= random.uniform(0, 1) for i in range(num_clients)]

    mean_est = np.zeros(dim)
    for i in range(num_clients):
        if transmits[i]:
            mean_est += Y[i]
    mean_est /= num_clients

    return mean_est


def dme_simulate(dim: int = 100, num_clients: int = 10, num_trials: int = 50):
    """
    Simulates distributed mean estimation
    :param dim: Dimension of the problem
    :param num_clients: Number of clients
    :param num_trials: Number of independent trials for which the experiment is repeated
    Returns simulation plots
    """

    p_arr_means = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    error_arr_naive_avg = np.zeros(len(p_arr_means))
    error_arr_colab_avg = np.zeros(len(p_arr_means))
    _, connectivity_matrix = client_locations_mmWave_clusters()

    for i in range(num_trials):

        # # Generate a random realization of the transmission probabilities of clients to PS
        # p_arr = [np.random.normal(loc=p_arr_means[ii], scale=1e-3) for ii in range(len(p_arr_means))]

        # Generate a random realization of clients to each other
        # P = np.random.uniform(0, 1, 10) * connectivity_matrix
        # P = (P + np.transpose(P)) / 2
        P = 0.5 * connectivity_matrix
        for j in range(num_clients):
            P[j][j] = 1

        data = np.zeros([num_clients, dim])
        for k in range(num_clients):
            data[k] = np.random.randn(dim) + k ** 5
        mean_true = np.mean(data, axis=0)

        error_arr_naive = np.zeros(len(p_arr_means))
        error_arr_colab = np.zeros(len(p_arr_means))
        for (ctr, p) in enumerate(p_arr_means):

            print("Trial no.: {}, ctr: {}/{}".format(i, ctr, len(p_arr_means)))
            t_probs = np.zeros(num_clients)
            for jj in range(num_clients):
                t_probs[jj] = np.random.normal(loc=p, scale=1e-3)
            t_probs = np.clip(a=t_probs, a_min=0, a_max=1)
            # print("ctr={}, t_probs={}".format(ctr, t_probs))

            # Do naive mean estimation
            mean_est_naive = dme_intermittent_naive(transmit_probs=t_probs, client_data=data)
            error_arr_naive[ctr] = np.linalg.norm(mean_est_naive - mean_true, 2)

            # Do mean estimation with collaboration
            mean_est_colab = dme_intermittent_colab(transmit_probs=t_probs, prob_ngbrs=P, client_data=data)
            error_arr_colab[ctr] = np.linalg.norm(mean_est_colab - mean_true, 2)

        error_arr_naive_avg += error_arr_naive
        error_arr_colab_avg += error_arr_colab

    error_arr_naive_avg /= num_trials
    error_arr_colab_avg /= num_trials

    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(p_arr_means, error_arr_naive_avg, marker='P', label="Naive", markersize=8, linewidth=3.0, color='black')
    ax.plot(p_arr_means, error_arr_colab_avg, marker='^', label="Collaboration", markersize=8, linewidth=3.0,
            color='blue')
    plt.xticks(weight='bold', fontsize="20")
    plt.yticks(weight='bold', fontsize="20")
    ax.set_xlabel('Transmit probability to PS (p)', weight="bold", size=40)
    ax.set_ylabel("Mean squared error", weight="bold", size=35)
    ax.set_title("DME with Intermittent Connectivity", weight="bold", size=40)
    legend_properties = {'weight': 'bold', 'size': 30}
    plt.grid(which="both", axis="both")
    plt.legend(loc="best", prop=legend_properties)
    plt.show()


def dme_simulate_varying_good_conn_clients(dim: int = 100, num_clients: int = 10, num_trials: int = 50):
    """
    Obtains the plot of MSE vs. the number of good connectivity clients
    :param dim: Dimension of the problem
    :param num_clients: Number of clients
    :param num_trials: Number of independent trials for which the experiment is repeated
    Returns simulation plots
    """

    p_good = 0.9                # Transmit probability of good connectivity clients
    p_bad = 0.2                 # Transmit probability of bad connectivity clients

    # Get the intermittent connectivity matrix amongst clients
    P = 0.5 * fully_connected_topology(num_clients)
    for ii in range(num_clients):
        P[ii][ii] = 1

    # gc -- good clients
    gc_arr = np.array([k for k in range(num_clients-1)]) + 1
    err_arr_naive = np.zeros(len(gc_arr))
    err_arr_colab = np.zeros(len(gc_arr))

    for gc in range(num_clients - 1):
        t_probs = p_bad * np.ones(num_clients)
        t_probs[0:gc+1] = p_good

        err_naive_avg = 0
        err_colab_avg = 0

        for i in range(num_trials):

            print("gc: {}/{}. Trial no.: {}/{}.".format(gc+1, num_clients-1, i+1, num_trials))

            # Generate data across clients
            data = np.random.randn(num_clients, dim)
            # data = np.zeros([num_clients, dim])
            # for k in range(num_clients):
            #     data[k] = k ** 5 * np.random.randn(dim)
            mean_true = np.mean(data, axis=0)

            # Do naive mean estimation
            mean_est_naive = dme_intermittent_naive(transmit_probs=t_probs, client_data=data)
            err_naive = np.linalg.norm(mean_est_naive - mean_true, 2)
            err_naive_avg += err_naive

            # Do mean estimation with collaboration
            mean_est_colab = dme_intermittent_colab(transmit_probs=t_probs, prob_ngbrs=P, client_data=data)
            err_colab = np.linalg.norm(mean_est_colab - mean_true, 2)
            err_colab_avg += err_colab

        err_arr_naive[gc] = err_naive_avg / num_trials
        err_arr_colab[gc] = err_colab_avg / num_trials

    # Save the results
    cwd = os.getcwd()
    err_arr_naive_fname = os.path.join(cwd, 'err_arr_naive.npy')
    err_arr_colab_fname = os.path.join(cwd, 'err_arr_colab.npy')
    np.save(err_arr_naive_fname, err_arr_naive)
    np.save(err_arr_colab_fname, err_arr_colab)

    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(gc_arr, err_arr_naive, marker='P', label="Naive", markersize=8, linewidth=3.0, color='black')
    ax.plot(gc_arr, err_arr_colab, marker='^', label="Collaboration", markersize=8, linewidth=3.0, color='blue')
    plt.xticks(weight='bold', fontsize="20")
    plt.yticks(weight='bold', fontsize="20")
    ax.set_xlabel('No. of good connectiivty clients', weight="bold", size=40)
    ax.set_ylabel("Mean squared error", weight="bold", size=35)
    ax.set_title("DME with Intermittent Connectivity", weight="bold", size=40)
    legend_properties = {'weight': 'bold', 'size': 30}
    plt.grid(which="both", axis="both")
    plt.legend(loc="best", prop=legend_properties)
    plt.show()


def plot_dme_varying_good_conn_clients(num_clients: int = 10):

    gc_arr = np.array([k for k in range(num_clients - 1)]) + 1
    cwd = os.getcwd()
    err_arr_naive = np.load(os.path.join(cwd, 'err_arr_naive.npy'))
    err_arr_colab = np.load(os.path.join(cwd, 'err_arr_colab.npy'))
    fig, ax = plt.subplots()
    ax.plot(gc_arr[1:], err_arr_naive[1:], marker='P', label="Naive", markersize=8, linewidth=3.0, color='black')
    ax.plot(gc_arr[1:], err_arr_colab[1:], marker='^', label="Collaboration", markersize=8, linewidth=3.0,
            color='blue')
    plt.xticks(weight='bold', fontsize="20")
    plt.yticks(weight='bold', fontsize="20")
    ax.set_xlabel('No. of good connectiivty clients', weight="bold", size=40)
    ax.set_ylabel("Mean squared error", weight="bold", size=35)
    ax.set_title("DME with Intermittent Connectivity", weight="bold", size=40)
    legend_properties = {'weight': 'bold', 'size': 30}
    plt.grid(which="both", axis="both")
    plt.legend(loc="best", prop=legend_properties)
    plt.show()


def dme_compare_intermittent_vs_perfect_topology(num_clients: int = 10, dim: int = 100, num_trials: int = 100):
    """
    Comparison of perfect vs. intermittent decentralized connectivity amongst clients for a specific mmWave clustered
    topology
    """

    t_prob, prob_ngbrs, conn_mat = clustered_topology_intermittent_vs_perfect_compare(cluster1_ang=2 * np.pi / 3,
                                                                                      cluster2_ang=4 * np.pi / 3)

    err_perf_conn = 0
    err_intr_conn = 0

    for r in range(num_trials):

        print("Trial no: {}".format(r))

        data = np.zeros([num_clients, dim])
        for k in range(num_clients):
            data[k] = np.random.randn(dim) * k ** 5 * 1e-4
        mean_true = np.mean(data, axis=0)

        mean_perf_conn = dme_intermittent_colab(transmit_probs=t_prob, prob_ngbrs=conn_mat, client_data=data)
        mean_intr_conn = dme_intermittent_colab(transmit_probs=t_prob, prob_ngbrs=prob_ngbrs, client_data=data)

        err_perf_conn += np.linalg.norm(mean_perf_conn - mean_true, 2)
        err_intr_conn += np.linalg.norm(mean_intr_conn - mean_true, 2)

    err_perf_conn /= num_trials
    err_intr_conn /= num_trials

    print("MSE with perfect connectivity: {}".format(err_perf_conn))
    print("MSE with intermittent connectivity: {}".format(err_intr_conn))


if __name__ == "__main__":

    # dme_simulate(num_trials=10)
    # dme_simulate_varying_good_conn_clients(num_trials=1)
    # plot_dme_varying_good_conn_clients()

    dme_compare_intermittent_vs_perfect_topology()
