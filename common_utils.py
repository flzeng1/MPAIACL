import random

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
import torch.nn as nn
from datetime import datetime
import os

import torch.nn.functional as F

marginloss = nn.MarginRankingLoss(0.5)


def parameter_visualization(args):
    args_dict = vars(args)
    max_key_len = max(len(key) for key in args_dict.keys())
    lines = []
    for key, value in args_dict.items():
        print(f"{key.ljust(max_key_len)} : {value}")
        lines.append(f"{key.ljust(max_key_len)} : {value}")
    return "\n".join(lines)

def print_log(args, final_test_acc, total_time, model, args_list):
    info_log = 'Dataset: {}, Epochs: {}, Batch Size: {}, Repetition: {}, Domain: {}, Total Time: {}, Ave Time: {}, Fine-model: {}' \
        .format(args.dataset, args.epochs, args.batch_size, args.repetition, args.domain, total_time,
                total_time / args.repetition, model)

    avg_acc = np.mean(final_test_acc)
    acc_log = 'finall: Test ACC OOD: [{:.4f}±{:.4f}]'.format(avg_acc,
                                                             np.std(final_test_acc))
    final_log = "ALL OOD:{}".format(final_test_acc)

    print(info_log)
    print(acc_log)
    print(final_log)


    log = f"{args_list}\n\n{info_log}\n\n{acc_log}\n\n{final_log}"
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # file_name = f"{args.dataset}_{args.net}_{args.repetitions}_acc_{avg_acc:.4f}_{current_time}.txt"
    file_name = f"{args.shift}_{args.dataset}_{args.domain}_{args.repetition}_acc_{avg_acc:.4f}_{current_time}.txt"

    file_path = os.path.join('./result', file_name)
    with open(file_path, 'w') as f:
        f.write(log)

    print(log)


def set_network_train_eval(causaler, attacker, train="causaler"):
    if train == 'causaler':
        for net in causaler:
            net.train()
            net.zero_grad()
        for net in attacker:
            net.eval()
    else:
        for net in attacker:
            net.train()
            net.zero_grad()
        for net in causaler:
            net.eval()


# all in the same shape
def info_nce(ori, pos, neg, temp=1):
    ori = F.normalize(ori, dim=1)
    pos = F.normalize(pos, dim=1)
    neg = F.normalize(neg, dim=1)

    pos_sim = torch.exp(torch.sum(ori * pos, dim=-1) / temp)
    neg_sim = torch.exp(neg / temp)

    loss = -torch.log((pos_sim / neg_sim.sum(dim=-1))).mean()
    return loss


def info_nce_DA(z1, z2, temp=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat((z1, z2), dim=0)

    sim_matrix = torch.exp(torch.mm(z, z.t().contiguous()) / temp)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * z1.size(0), device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * z1.size(0), -1)

    pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / temp)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    loss = -torch.log(pos_sim / sim_matrix.sum(dim=-1)).mean()
    return loss


def contrast_loss(h_graph_cau, h_graph_cau_aug, h_graph_adv, temp=0.5):

    h_graph_cau = F.normalize(h_graph_cau, dim=1)
    h_graph_cau_aug = F.normalize(h_graph_cau_aug, dim=1)
    h_graph_adv = F.normalize(h_graph_adv, dim=1)

    pos_sim = torch.sum(h_graph_cau * h_graph_cau_aug, dim=1) / temp
    neg_sim = torch.sum(h_graph_cau * h_graph_adv, dim=1) / temp

    logits = torch.stack([pos_sim, neg_sim], dim=1)

    labels = torch.zeros(h_graph_cau.size(0), dtype=torch.long, device=h_graph_cau.device)

    loss = F.cross_entropy(logits, labels)

    return loss

def triplet_contrast_loss(h_anchor, h_negative, pos_aug_fn=None, margin=1.0):

    h_anchor = F.normalize(h_anchor, dim=1)
    h_negative = F.normalize(h_negative, dim=1)

    if pos_aug_fn is not None:
        h_positive = pos_aug_fn(h_anchor)
    else:
        h_positive = F.dropout(h_anchor, p=0.1, training=True)

    # pairwise distances
    d_ap = F.pairwise_distance(h_anchor, h_positive, p=2)
    d_an = F.pairwise_distance(h_anchor, h_negative, p=2)

    loss = F.relu(d_ap - d_an + margin).mean()
    return loss


def set_random_seed(seed):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)





def get_OPT(model, args, l2reg):
    opt_attacker = optim.Adam(model.attacker.parameters(), lr=args.attacker_lr, weight_decay=l2reg)
    opt_causaler = optim.Adam(list(model.graph_front.parameters())
                              + list(model.graph_backs.parameters())
                              + list(model.causaler.parameters())
                              + list(model.predictor.parameters()),
                              lr=args.causaler_lr, weight_decay=l2reg)

    return opt_attacker, opt_causaler


def get_schedule(opt_attacker, opt_causaler, args):
    scheduler_attacker = None
    scheduler_causaler = None

    if args.lr_scheduler == 'step':
        scheduler_attacker = StepLR(opt_attacker, step_size=args.lr_decay, gamma=args.lr_gamma)
        scheduler_causaler = StepLR(opt_causaler, step_size=args.lr_decay, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'muti':
        scheduler_attacker = MultiStepLR(opt_attacker, milestones=args.milestones, gamma=args.lr_gamma)
        scheduler_causaler = MultiStepLR(opt_causaler, milestones=args.milestones, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cos':
        scheduler_attacker = CosineAnnealingLR(opt_attacker, T_max=args.epochs, eta_min=args.lr_min)
        scheduler_causaler = CosineAnnealingLR(opt_causaler, T_max=args.epochs, eta_min=args.lr_min)
    else:
        pass

    return scheduler_attacker, scheduler_causaler


# cite from 'AIA'



def init_weights(net, init_type='orthogonal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'default':
                pass
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>



