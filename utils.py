import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def pull_model(model_user, model_server):
    with torch.no_grad():
        for param_user, param_server in zip(model_user.parameters(), model_server.parameters()):
            param_user.data = param_server.data[:] + 0
    return None


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def zero_grad(model):
    for param in model.parameters():
        param.grad = torch.zeros_like(param.data)
    return None


def initialize_zero(model):
    for param in model.parameters():
        param.data.mul_(0)
    return None




def get_model_flattened(model, device):
    model_flattened = torch.empty(0).to(device)
    for p in model.parameters():
        a = p.data.flatten().to(device)
        model_flattened = torch.cat((model_flattened, a), 0)
    return model_flattened



def unflat_model(model, model_flattened):
    # unflattens the grad_flattened into the model.grad
    i = 0
    for p in model.parameters():
        temp = model_flattened[i:i+p.data.numel()]
        p.data = temp.reshape(p.data.size())
        i += p.data.numel()
    return None





def step_sgd(model, momentum, lr,args):
    last_ind = 0
    for p in model.parameters():
        if p.requires_grad:
            d_p = p.grad.data.detach().add(p.data.detach(),alpha=args.wd)
            if momentum is None:
                buf = d_p
            else:
                length, dims = d_p.numel(), d_p.size()
                buf = momentum[last_ind:last_ind + length].view(dims)
                buf.mul_(args.Lmomentum).add_(d_p)
                momentum[last_ind:last_ind + length] = buf.flatten()  # update buffer
                last_ind += length
            if args.nesterov:
                d_p = d_p.add(buf, alpha=args.Lmomentum)
            else:
                d_p = buf
            p.data.add_(d_p, alpha=-lr)
    if momentum is not None:
        return momentum
    else:
        return None

def evaluate_accuracy(model, testloader, device):
    """Calculates the accuracy of the model"""

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


class PadSequence:
    def __call__(self, batch):
        class MockTextInput:

            def __init__(self, x, x_lengths):
                self.x = x
                self.x_lengths = x_lengths

            def to(self, device):
                self.x = self.x.to(device)

                return self

        xx, yy = zip(*batch)

        xx_lens = [xx[idx][0].shape[0] for idx in range(len(xx))]

        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)

        yy = torch.LongTensor(yy)

        res = list(zip(xx_pad, xx_lens, yy))

        x, x_lens, labels = map(list, zip(*res))
        labels = torch.LongTensor(labels)
        x = torch.stack(x)

        return MockTextInput(x, x_lens), labels


def pad_sequence_if_text(dataset_name):
    if dataset_name != "glue":
        return None

    return PadSequence()
