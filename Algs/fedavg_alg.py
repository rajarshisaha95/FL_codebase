from threading import local
import torch
from torch.utils.data import DataLoader
# custom modules
import data_loader as dl
from nn_classes import *
from utils import *
import numpy as np
from loss import loss_pick
import random

from Algs.vector_quantizers import naive_quantizer, hadamard_quantizer


def train_fedavg(args, device):

    print("Federated Averaging with perfect connectivity.")

    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds, data_map = dl.get_indices(trainset, args)

    # PS model
    net_ps = get_net(args)
    net_ps.eval()
    params = count_parameters(net_ps)
    global_momentum = torch.zeros(params)

    net_users = [get_net(args) for u in range(num_client)]
    opts = [torch.optim.SGD(net.parameters(), args.lr, weight_decay=args.wd) for net in net_users]

    criterions = [loss_pick(args) for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, pin_memory=True)

    [pull_model(net_users[cl], net_ps) for cl in range(num_client)]
    # synch all clients models models with PS

    accuracys = []
    trainloaders = [DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                               shuffle=True) for cl in range(num_client)]

    # Federated Averaging
    for rnd in range(args.comm_rounds):
        difference_vec = []
        ps_model_flat = get_model_flattened(net_ps, 'cpu')
        received_diffs = torch.zeros(params)

        for cl, net in enumerate(net_users):
            epoch(net, opts[cl], trainloaders[cl], criterions[cl], device, args)
            difference_vec.append(ps_model_flat.sub(get_model_flattened(net, 'cpu')))

        for cl in range(num_client):
            received_diffs.add_(difference_vec[cl], alpha=1/num_client)

        # print("Received diffs: {}", format(received_diffs))
        # print("Data type: {}", format(type(received_diffs)))
        # print("Data type: {}", format(received_diffs.shape))

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


def train_fedavg_blind(args, device):

    print("Blind FedAvg with intermittent connectivity of clients to PS and no collaboration.")

    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds, data_map = dl.get_indices(trainset, args)

    # PS model
    net_ps = get_net(args)
    net_ps.eval()
    params = count_parameters(net_ps)
    global_momentum = torch.zeros(params)

    net_users = [get_net(args) for u in range(num_client)]
    opts = [torch.optim.SGD(net.parameters(), args.lr, weight_decay=args.wd) for net in net_users]

    criterions = [loss_pick(args) for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, pin_memory=True)

    [pull_model(net_users[cl], net_ps) for cl in range(num_client)]
    # synch all clients models models with PS

    accuracys = []
    trainloaders = [DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                               shuffle=True) for cl in range(num_client)]

    # Transmission probabilities
    transmit_p_arr = np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.5, 0.8, 0.1, 0.2, 0.9])

    # Blind federated Averaging
    for rnd in range(args.comm_rounds):
        transmits = [transmit_p_arr[i] >= random.uniform(0, 1) for i in range(num_client)]
        difference_vec = []
        ps_model_flat = get_model_flattened(net_ps, 'cpu')
        received_diffs = torch.zeros(params)

        for cl, net in enumerate(net_users):
            epoch(net, opts[cl], trainloaders[cl], criterions[cl], device, args)
            difference_vec.append(ps_model_flat.sub(get_model_flattened(net, 'cpu')))

        for cl in range(num_client):
            if transmits[cl]:
                received_diffs.add_(difference_vec[cl], alpha=1/num_client)

        # print("Received diffs: {}", format(received_diffs))
        # print("Data type: {}", format(type(received_diffs)))
        # print("Data type: {}", format(received_diffs.shape))

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


def train_fedavg_nonblind(args, device):

    print("Non-blind FedAvg with intermittent connectivity of clients to PS and no collaboration.")

    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds, data_map = dl.get_indices(trainset, args)

    # PS model
    net_ps = get_net(args)
    net_ps.eval()
    params = count_parameters(net_ps)
    global_momentum = torch.zeros(params)

    net_users = [get_net(args) for u in range(num_client)]
    opts = [torch.optim.SGD(net.parameters(), args.lr, weight_decay=args.wd) for net in net_users]

    criterions = [loss_pick(args) for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, pin_memory=True)

    [pull_model(net_users[cl], net_ps) for cl in range(num_client)]
    # synch all clients models models with PS

    accuracys = []
    trainloaders = [DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                               shuffle=True) for cl in range(num_client)]

    # Transmission probabilities
    # transmit_p_arr = np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.5, 0.8, 0.1, 0.2, 0.9])

    # Transmission probabilities for lesser connectivity (initial expt. --
    # iid local datasets with heterogeneous connectivity)
    # transmit_p_arr = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9])

    # Transmission probabilities for lesser connectivity (second expt --
    # iid local datasets with heterogeneous connectivity with poorer connectivity of the good client)
    transmit_p_arr = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
    print("transmit_p_array: {}".format(transmit_p_arr))

    # Non-blind federated Averaging
    for rnd in range(args.comm_rounds):
        transmits = [transmit_p_arr[i] >= random.uniform(0, 1) for i in range(num_client)]
        num_clients_transmitting = np.count_nonzero(transmits)
        difference_vec = []
        ps_model_flat = get_model_flattened(net_ps, 'cpu')
        received_diffs = torch.zeros(params)

        for cl, net in enumerate(net_users):
            epoch(net, opts[cl], trainloaders[cl], criterions[cl], device, args)
            difference_vec.append(ps_model_flat.sub(get_model_flattened(net, 'cpu')))

        for cl in range(num_client):
            if transmits[cl]:
                received_diffs.add_(difference_vec[cl], alpha=1/num_clients_transmitting)

        # print("Received diffs: {}", format(received_diffs))
        # print("Data type: {}", format(type(received_diffs)))
        # print("Data type: {}", format(received_diffs.shape))

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


def train_fedavg_naive_quant(args, device):
    B = args.bits
    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds, data_map = dl.get_indices(trainset, args)

    # PS model
    net_ps = get_net(args)
    net_ps.eval()
    params = count_parameters(net_ps)
    global_momentum = torch.zeros(params)

    net_users = [get_net(args) for u in range(num_client)]
    opts = [torch.optim.SGD(net.parameters(), args.lr, weight_decay=args.wd) for net in net_users]

    criterions = [loss_pick(args) for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, pin_memory=True)

    [pull_model(net_users[cl], net_ps) for cl in range(num_client)]
    # synch all clients models models with PS

    accuracys = []
    trainloaders = [DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                               shuffle=True) for cl in range(num_client)]

    # Federated Averaging
    for rnd in range(args.comm_rounds):
        difference_vec = []
        ps_model_flat = get_model_flattened(net_ps, 'cpu')
        received_diffs = torch.zeros(params)

        for cl, net in enumerate(net_users):
            epoch(net, opts[cl], trainloaders[cl], criterions[cl], device, args)
            difference_vec.append(ps_model_flat.sub(get_model_flattened(net, 'cpu')))

        for cl in range(num_client):
            local_update_quant = naive_quantizer(x=difference_vec[cl].numpy(), bits=B)
            # received_diffs.add_(difference_vec[cl], alpha=1/num_client)
            received_diffs.add_(torch.from_numpy(local_update_quant), alpha=1/num_client)
            # print("Local update at client {}: {}".format(cl, difference_vec[cl]))
            # print("Quantized local update at client {}: {}".format(cl, local_update_quant))

        # print("Received diffs: {}", format(received_diffs))
        # print("Data type: {}", format(type(received_diffs)))
        # print("Data type: {}", format(received_diffs.shape))

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


def train_fedavg_hadamard_quant(args, device):
    B = args.bits
    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds, data_map = dl.get_indices(trainset, args)

    # PS model
    net_ps = get_net(args)
    net_ps.eval()
    params = count_parameters(net_ps)
    global_momentum = torch.zeros(params)

    net_users = [get_net(args) for u in range(num_client)]
    opts = [torch.optim.SGD(net.parameters(), args.lr, weight_decay=args.wd) for net in net_users]

    criterions = [loss_pick(args) for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, pin_memory=True)

    [pull_model(net_users[cl], net_ps) for cl in range(num_client)]
    # synch all clients models models with PS

    accuracys = []
    trainloaders = [DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                               shuffle=True) for cl in range(num_client)]

    # Federated Averaging
    for rnd in range(args.comm_rounds):
        difference_vec = []
        ps_model_flat = get_model_flattened(net_ps, 'cpu')
        received_diffs = torch.zeros(params)

        for cl, net in enumerate(net_users):
            epoch(net, opts[cl], trainloaders[cl], criterions[cl], device, args)
            difference_vec.append(ps_model_flat.sub(get_model_flattened(net, 'cpu')))

        for cl in range(num_client):
            local_update_quant = hadamard_quantizer(x=difference_vec[cl].numpy(), bits=B)
            # received_diffs.add_(difference_vec[cl], alpha=1/num_client)
            received_diffs.add_(torch.from_numpy(local_update_quant), alpha=1/num_client)
            # print("Local update at client {}: {}".format(cl, difference_vec[cl]))
            # print("Quantized local update at client {}: {}".format(cl, local_update_quant))

        # print("Received diffs: {}", format(received_diffs))
        # print("Data type: {}", format(type(received_diffs)))
        # print("Data type: {}", format(received_diffs.shape))

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
