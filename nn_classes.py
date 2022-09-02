from Models.CNN import *
from Models.ResNet import *
from Models.RNN import RecurrentModel
from Models.VGG import VGG


def get_net(args):
    class GroupNorm(nn.GroupNorm):
        def __init__(self, num_channels):
            super().__init__(num_groups=args.num_groups, num_channels=num_channels)

    norms = {'bn': nn.BatchNorm2d,
             'gn': GroupNorm,
             '-': NoneNorm, None: NoneNorm}
    labels = {'cifar10': 10,
              'svhn': 10,
              'cifar100': 100,
              'tiny_imagenet': 200}
    norm, num_cls = norms.get(args.norm_type), labels.get(args.dataset_name)
    # First Network Architecture, than its parameters in order
    neural_networks = {'simplecifar': [SimpleCifarNet, norm, num_cls],
                       'simplecnn': [SimpleCNN, norm, num_cls],
                       'simplecifarmoon': [SimpleCifarNetMoon, norm, num_cls],
                       'moon_net':[moon_net, norm, num_cls],
                       'resnet8': [ResNet_3Layer, BasicBlock, [1, 1, 1], norm, num_cls],
                       'resnet20': [ResNet_3Layer, BasicBlock, [2, 2, 2], norm, num_cls],
                       'resnet9': [ResNet, BasicBlock, [1, 1, 1, 1], norm, num_cls],
                       'resnet18': [ResNet, BasicBlock, [2, 2, 2, 2], norm, num_cls],
                       'vgg11': [VGG, '11', norm, num_cls],
                       'vgg13': [VGG, '13', norm, num_cls],
                       'vgg16': [VGG, '16', norm, num_cls],
                       'vgg19': [VGG, '19', norm, num_cls],
                       'gru': [RecurrentModel, args.nn_name],
                       'lstm': [RecurrentModel, args.nn_name],
                       'rnn': [RecurrentModel, args.nn_name]
                       }
    try:
        network = neural_networks[args.nn_name.lower()]
    except:
        print('Available Neural Networks')
        print(neural_networks.keys())
        raise ValueError
    net = network[0](*network[1:])
    if args.mode =='moon':
        net = ModelFedCon(net,num_cls,args.nn_name.lower())
    init_weights(net, args)
    return net


class NoneNorm(nn.Module):
    def __init__(self, num_features):
        super(NoneNorm, self).__init__()

    def forward(self, x):
        return x


def kaiming_normal_init(m):  ## improves convergence at <200 comm rounds
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        # nn.init.xavier_normal(m.weight)
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0)


def init_weights(net, args):
    if args.weight_init == 'kn':
        for m in net.modules():
            kaiming_normal_init(m)


class ModelFedCon(nn.Module):

    def __init__(self, model, n_classes,nn_name):
        super(ModelFedCon, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])
        out_dim = 256

        if nn_name[:6] =='resnet':
            self.num_ftrs = model.linear.in_features
        elif nn_name == 'Simplecifar':
            self.num_ftrs = 512
        else:
            raise NotImplementedError

        # projection MLP
        self.l1 = nn.Linear(self.num_ftrs, self.num_ftrs)
        self.l2 = nn.Linear(self.num_ftrs, out_dim)

        # last layer
        self.l3 = nn.Linear(out_dim, n_classes)

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        x = F.relu(self.l1(h))
        x = self.l2(x)
        y = self.l3(x)
        return x, y