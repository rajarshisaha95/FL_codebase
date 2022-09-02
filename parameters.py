import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # technical params
    parser.add_argument('--cuda', type=bool, default=True, help='Use cuda as device')
    parser.add_argument('--gpu_id', type=int, default=0, help='if cuda true, select gpu to work')
    parser.add_argument('--worker_per_device', type=int, default=1, help='parallel processes per device')
    parser.add_argument('--excluded_gpus', type=list, default=[], help='bypassed gpus')
    parser.add_argument('--trials', type=int, default=7, help='number of trials')

    # Federated params
    parser.add_argument('--mode', type=str, default='fed_tmp', help='slowmo,fed_avg,ADC,doubleM,fed_gkd,fed_prox')
    parser.add_argument('--comm_rounds', type=int, default=500, help='Total communication rounds')
    parser.add_argument('--LocalIter', type=int, default=8, help='Local steps for workers')
    parser.add_argument('--num_client', type=int, default=10, help='number of clients')
    parser.add_argument('--transmit_p', type=float, default=.5, help='number of clients')
    parser.add_argument('--bits', type=float, default=8, help='bits per dimension per client')

    # dataset related
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='cifar10,cifar100,svhn,tiny_imagenet,glue')
    parser.add_argument('--dataset_dist', type=str, default='sort_part',
                        help='distribution of dataset; iid or sort_part, dirichlet')
    parser.add_argument('--numb_cls_usr', type=int, default=2,
                        help='number of label type per client if sort_part selected')
    parser.add_argument('--alpha', type=list, default=[1,.5],
                        help='alpha constant for dirichlet dataset_dist,lower for more skewness')
    parser.add_argument('--bs', type=int, default=64, help='batchsize')

    # nn related
    parser.add_argument('--nn_name', type=str, default='simplecnn',
                        help='simplecnn,simplecifar,VGGs resnet(8-9-18-20), gru')
    parser.add_argument('--weight_init', type=str, default='-',
                        help='nn weight init, kn (Kaiming normal) or - (None)')
    parser.add_argument('--norm_type', type=str, default='-',
                        help='gn (GroupNorm), bn (BatchNorm), - (None)')
    parser.add_argument('--num_groups', type=int, default=32,
                        help='number of groups if GroupNorm selected as norm_type, 1 for LayerNorm')

    # optimiser related
    parser.add_argument('--loss', type=str, default='CE', help='see loss.py')
    parser.add_argument('--lr', type=float, default=0.05, help='learning_rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='L2 regularizer strength')
    parser.add_argument('--beta', type=float, default=0.9, help='drift control constant (Global momentum)')
    parser.add_argument('--alpha_distill', type=float, default=0.1, help='KLD alpha')
    parser.add_argument('--T', type=float, default=5, help='KLD temperature')

    # misc params

    args = parser.parse_args()
    return args
