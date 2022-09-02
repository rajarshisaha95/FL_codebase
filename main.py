import random
import datetime
import os
import argparse
import itertools

import numpy as np
import torch
from parameters import args_parser
from Algs.colrel_alg import train_colrel, train_colrel_intermittent
from Algs.fedavg_alg import train_fedavg, train_fedavg_blind, train_fedavg_nonblind,\
    train_fedavg_naive_quant, train_fedavg_hadamard_quant
from pynvml import *


if __name__ == '__main__':
    args = args_parser()
    if args.cuda:
        if torch.cuda.device_count() < 1:
            device ='cpu'
            print('No Nvidia gpu found to use cuda, overriding "cpu" as device')
        else:
            device = torch.device(f"cuda:{args.gpu_id}")
    simulation_ID = int(random.uniform(1, 999))
    x = datetime.datetime.now()
    date = '{}, {}, {}.{}'.format(x.strftime('%B'), x.strftime('%d'), x.strftime('%H'), x.strftime('%M'))

    # new algorithms must be added in mapper to work
    alg_mapper = {'colrel': train_colrel, 'fed_avg': train_fedavg, 'fed_avg_naive_quant': train_fedavg_naive_quant,
                  'fed_avg_hadamard_quant': train_fedavg_hadamard_quant, 'colrel_int': train_colrel_intermittent,
                  'fed_avg_blind': train_fedavg_blind, 'fed_avg_nonblind': train_fedavg_nonblind}

    dist = args.dataset_dist
    if dist == 'dirichlet':
        dist += '_Alpha_{}'.format(args.alpha)
    elif dist == 'sort_part':
        dist += '_cls_{}'.format(args.numb_cls_usr)

    newFile = '{}-{}'.format(args.mode, dist)
    if not os.path.exists(os.getcwd() + '/Results'):
        os.mkdir(os.getcwd() + '/Results')
    n_path = os.path.join(os.getcwd(), 'Results')
    nn = args.nn_name
    if not os.path.exists(os.path.join(n_path, nn)):
        os.mkdir(os.path.join(n_path, nn))
    n_path = os.path.join(n_path, nn, newFile)
    for i in range(args.trials):
        accs = alg_mapper[args.mode](args, device)
        if i == 0:
            while os.path.exists(n_path):
                tmp_path = n_path.split(' (')
                if len(tmp_path) > 1:
                    temp_value = eval(tmp_path[-1][0])
                    n_path = '{} ({})'.format(tmp_path[0], temp_value + 1)
                else:
                    n_path = '{} ({})'.format(tmp_path[0], 1)
            os.mkdir(n_path)
            f = open(n_path + '/simulation_Details.txt', 'w+')
            f.write('simID = ' + str(simulation_ID) + '\n')
            f.write('Started at {}'.format(date) + '\n')
            f.write('############## Args ###############' + '\n')
            for arg in vars(args):
                line = str(arg) + ' : ' + str(getattr(args, arg))
                f.write(line + '\n')
            f.write('############ Results ###############' + '\n')
            f.close()
        s_loc = f'Result_{args.mode}' + '-' + str(i)
        s_loc = os.path.join(n_path, s_loc)
        np.save(s_loc, accs)
        f = open(n_path + '/simulation_Details.txt', 'a+')
        f_acc = accs[-1] if accs[-1] > 0 else accs[-2]
        f.write('Trial {} final accuracy {}'.format(i, f_acc) + '\n')
        f.close()
        if accs[-1] < 0:
            f = open(n_path + '/simulation_Details.txt', 'a+')
            f.write('Automatic early stop (diverge)'.format(i, accs[-1]) + '\n')
            f.close()
            print('Simulation, {} ended early since it did not converge'.format(newFile))
            np.save(s_loc, accs[:-1])
            break

