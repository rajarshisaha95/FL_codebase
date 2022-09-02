import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_pick(args):
    loss_funcs = {'global': distill,
                'conf_global': confidence_distill,
                'ntd': FedLS_NTD, 'NTD': FedLS_NTD,
                'LS':NLLsmooth(),'CE':CrossEntropy(),
                }
    return loss_funcs[args.loss]


class NLLsmooth(nn.Module):
    """NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.05):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(NLLsmooth, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target,*params):
        logprobs = F.log_softmax(x, dim=-1)  # along last direction
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))  # increase the dimension
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class CrossEntropy(nn.CrossEntropyLoss):
    def forward(self, input, target, *params):
        return F.cross_entropy(input, target)


def distill(outputs, labels, teacher_outputs, args,*params):
    alpha = args.alpha_distill
    T = args.T
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
                             F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + F.cross_entropy(outputs,
                                                                                                        labels) * (
                          1. - alpha)
    return KD_loss


def confidence_distill(outputs, labels, teacher_outputs, args, data_dist):
    alpha = args.alpha_distill
    dist = torch.from_numpy(data_dist)
    max_ = torch.max(dist, dim=0)[0]
    dist = torch.ones_like(dist).sub(dist.mul(1 / max_))
    dist = dist.view(1, dist.size(0)).expand(labels.size(0), dist.size(0))
    dist = dist.to(labels)
    for label, d in zip(labels, dist):
        d[label] = 0
    teacher_softs = F.softmax(teacher_outputs.detach(), dim=1)
    teacher_softs.mul_(dist)
    for label, prediction in zip(labels, teacher_softs):
        true_pred = 1 - prediction.sum(dim=0)
        pred_vec = torch.zeros_like(prediction).to(label)
        pred_vec[label] = true_pred
        prediction.add_(pred_vec)
    teacher_softs = teacher_softs.mul(0.95).add(torch.ones_like(teacher_softs).to(labels),
                                                alpha=0.05 / teacher_softs.size(1))
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs, dim=1), teacher_softs) * alpha + F.cross_entropy(outputs,
                                                                                                     labels) * (
                          1 - alpha)
    return KD_loss


def FedLS_NTD(outputs, labels, teacher_outputs, args,*params):
    local_ntp = torch.zeros(outputs.size(0), outputs.size(1) - 1).to(labels)
    global_ntp = torch.zeros(outputs.size(0), outputs.size(1) - 1).to(labels)
    lam = args.lamb
    alpha = args.alpha_distill
    T = args.T
    for i, label in enumerate(labels):
        local_ntp[i] = torch.cat([outputs[i][:label], outputs[i][label + 1:]])
        global_ntp[i] = torch.cat([teacher_outputs[i][:label], teacher_outputs[i][label + 1:]])
    local_ntp = local_ntp.float()
    global_ntp = global_ntp.float()
    NTD_loss = nn.KLDivLoss()(F.log_softmax(local_ntp, dim=1),
                              F.softmax(global_ntp, dim=1))
    KLD_loss = nn.KLDivLoss()(F.log_softmax(outputs, dim=1),
                              F.softmax(teacher_outputs, dim=1))
    distill_loss = (1 - lam) * KLD_loss + lam * NTD_loss
    Loss = (1 - alpha) * F.cross_entropy(outputs, labels) + alpha * distill_loss
    return Loss


