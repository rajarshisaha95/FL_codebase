import torch
from torch.utils.data import DataLoader
# custom modules
import data_loader as dl
from nn_classes import *
from utils import *
import numpy as np
from loss import loss_pick
import random

from Algs.optimize_weights_positive import opt_alphas
from Algs.optimize_weights_intermittent import opt_alphas_intermittent
from topology import client_locations_mmWave_clusters_intermittent, client_locations_mmWave_clusters_perfect_conn, \
    client_locations_mmWave_clusters_perfect_conn2, client_locations_mmWave_clusters_intermittent_conn2


def train_colrel(args, device):
    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds, data_map = dl.get_indices(trainset, args)

    # PS model
    net_ps = get_net(args)
    net_ps.eval()
    params = count_parameters(net_ps)
    global_momentum = torch.zeros(params)

    # Fully connected decentralized topology
    # neighbor_matrix = np.ones((num_client, num_client))
    # np.fill_diagonal(neighbor_matrix, 0)
    # weight_matrix = np.ones_like(neighbor_matrix) / num_client

    # Transmission probabilities
    # transmit_p_arr = np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.5, 0.8, 0.1, 0.2, 0.9])

    # Load connectivity information from topology for mmWave clusters
    # Generates a clustered topology of clients with perfect connectivity
    # transmit_p_arr, neighbor_matrix = client_locations_mmWave_clusters_perfect_conn(num_clients=num_client)

    # Trying out a new (slightly more connected topology for mmWave clusters)
    transmit_p_arr, neighbor_matrix = client_locations_mmWave_clusters_perfect_conn2(num_clients=num_client)
    print("transmit_p_arr = {}".format(transmit_p_arr))
    print("neighbor_matrix = {}".format(neighbor_matrix))

    # Optimize weights
    weight_matrix = opt_alphas_intermittent(transmit_probs=transmit_p_arr, ctr_max=50 * num_client,
                                            prob_ngbrs=neighbor_matrix)

    # Old tryouts for naive weight choices
    # weight_matrix = np.diag(1/transmit_p_arr)

    # Equal collaboration on fully connected topology
    # weight_matrix = np.ones_like(neighbor_matrix) / num_client

    net_users = [get_net(args) for u in range(num_client)]
    opts = [torch.optim.SGD(net.parameters(), args.lr, weight_decay=args.wd) for net in net_users]

    criterions = [loss_pick(args) for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, pin_memory=True)

    [pull_model(net_users[cl], net_ps) for cl in range(num_client)]
    # synch all clients models models with PS

    accuracys = []
    trainloaders = [DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                               shuffle=True) for cl in range(num_client)]

    # Collaborative Forwarding
    for rnd in range(args.comm_rounds):
        transmits = [transmit_p_arr[i] >= random.uniform(0, 1) for i in range(num_client)]
        difference_vec = []
        ps_model_flat = get_model_flattened(net_ps, 'cpu')
        received_diffs = torch.zeros(params)

        for cl, net in enumerate(net_users):
            epoch(net, opts[cl], trainloaders[cl], criterions[cl], device, args)
            difference_vec.append(ps_model_flat.sub(get_model_flattened(net, 'cpu')))

        for cl, neighbors in enumerate(neighbor_matrix):
            clients_neighbors = np.arange(num_client)[neighbors>0]
            avg_diff = difference_vec[cl].mul(weight_matrix[cl][cl])
            for neighbor in clients_neighbors:
                avg_diff.add_(difference_vec[neighbor], alpha=weight_matrix[cl][neighbor])
            if transmits[cl]:
                received_diffs.add_(avg_diff, alpha=1/num_client)

        # Server side
        # global momentum may need to scale if lr will change during the training.
        global_momentum = global_momentum.mul(args.beta).add(received_diffs)
        unflat_model(net_ps, ps_model_flat.sub(global_momentum))
        [pull_model(net, net_ps) for net in net_users]                              # Broadcast

        if (rnd+1) % 5 == 0:
            net_ps.to(device)
            acc = evaluate_accuracy(net_ps.to(device), testloader, device)
            net_ps.to('cpu')
            accuracys.append(acc * 100)
            print('Current round {}, Test accuracy {}'.format(rnd+1, accuracys[-1]))

    return accuracys


def train_colrel_intermittent(args, device):
    """
    Trains a model with collaborative relaying when the decentralized connectivity amongst clients is intermittent.
    """
    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds, data_map = dl.get_indices(trainset, args)

    # PS model
    net_ps = get_net(args)
    net_ps.eval()
    params = count_parameters(net_ps)
    global_momentum = torch.zeros(params)

    # # Fully connected (but intermittent) decentralized connectivity
    # neighbor_matrix = np.ones((num_client, num_client))
    # np.fill_diagonal(neighbor_matrix, 0)
    # P = 0.5 * np.ones((num_client, num_client))
    # np.fill_diagonal(P, 1)

    # Transmission probabilities
    # transmit_p_arr = np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.5, 0.8, 0.1, 0.2, 0.9])

    # Load connectivity information from topology for mmWave clusters
    # Generates a clustered topology of clients with intermittent connectivity
    transmit_p_arr, P, neighbor_matrix = client_locations_mmWave_clusters_intermittent(num_clients=num_client)

    # Trying out a new (slightly more connected topology for mmWave clusters)
    transmit_p_arr, P, neighbor_matrix = client_locations_mmWave_clusters_intermittent_conn2(num_clients=num_client)

    print("transmit_p_arr = {}".format(transmit_p_arr))
    print("P = {}".format(P))
    print("neighbor_matrix = {}".format(neighbor_matrix))

    # Optimized collaboration
    weight_matrix = opt_alphas_intermittent(transmit_probs=transmit_p_arr, ctr_max=50 * num_client, prob_ngbrs=P)

    net_users = [get_net(args) for u in range(num_client)]
    opts = [torch.optim.SGD(net.parameters(), args.lr, weight_decay=args.wd) for net in net_users]

    criterions = [loss_pick(args) for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, pin_memory=True)

    [pull_model(net_users[cl], net_ps) for cl in range(num_client)]
    # synch all clients models models with PS

    accuracys = []
    trainloaders = [DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                               shuffle=True) for cl in range(num_client)]

    # Collaborative Forwarding
    for rnd in range(args.comm_rounds):

        # Generate a random instance of the connectivity of clients to the PS
        transmits = [transmit_p_arr[i] >= random.uniform(0, 1) for i in range(num_client)]

        # Generate a random instance of the decentralized connectivity of clients to each other
        transmit_clients_colab = np.zeros([num_client, num_client])
        for i in range(num_client):
            transmit_clients_colab[i][i] = 1                                            # Always connected to itself
            for j in range(i+1, num_client):
                transmit_clients_colab[i][j] = P[i][j] >= random.uniform(0, 1)
                transmit_clients_colab[j][i] = transmit_clients_colab[i][j]             # Blockage model

        difference_vec = []
        ps_model_flat = get_model_flattened(net_ps, 'cpu')
        received_diffs = torch.zeros(params)

        for cl, net in enumerate(net_users):
            epoch(net, opts[cl], trainloaders[cl], criterions[cl], device, args)
            difference_vec.append(ps_model_flat.sub(get_model_flattened(net, 'cpu')))

        for cl, neighbors in enumerate(neighbor_matrix):
            clients_neighbors = np.arange(num_client)[neighbors > 0]
            avg_diff = difference_vec[cl].mul(weight_matrix[cl][cl])
            for neighbor in clients_neighbors:
                if transmit_clients_colab[neighbor][cl]:
                    avg_diff.add_(difference_vec[neighbor], alpha=weight_matrix[cl][neighbor])
            if transmits[cl]:
                received_diffs.add_(avg_diff, alpha=1/num_client)

        # Server side
        # global momentum may need to scale if lr will change during the training.
        global_momentum = global_momentum.mul(args.beta).add(received_diffs)
        unflat_model(net_ps, ps_model_flat.sub(global_momentum))
        [pull_model(net, net_ps) for net in net_users]                              # Broadcast

        if (rnd+1) % 5 == 0:
            net_ps.to(device)
            acc = evaluate_accuracy(net_ps.to(device), testloader, device)
            net_ps.to('cpu')
            accuracys.append(acc * 100)
            print('Current round {}, Test accuracy {}'.format(rnd+1, accuracys[-1]))

    return accuracys


def epoch(net, opt, loader, criterion, device, args):
    net.to(device)
    l_iter = 0
    # can be change to epoch
    while l_iter < args.LocalIter:
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            predicts = net(inputs)
            loss = criterion(predicts, labels)
            loss.backward()
            opt.step()
            l_iter += 1
            if l_iter == args.LocalIter:
                break
    net.to('cpu')
    return None
