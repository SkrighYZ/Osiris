import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ddp_utils import GatherLayer
from models.backbone import resnet18


def _make_projector(sizes):
    """
    make a simple MLP with linear layers followed by ReLU, as in SimCLR
    """
    layers = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=True))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))

    return nn.Sequential(*layers)


def _mask_correlated_samples(batch_size):
    """
    Generate a boolean mask which masks out the similarity between views of the same example in the similarity matrix
    e.g., a mask for batch size = 2 is a 4x4 matrix (due to two augmented views)
        0  1  0  1
        1  0  1  0
        0  1  0  1  
        1  0  1  0 
    """
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=bool)
    mask.fill_diagonal_(0)
    mask[:, batch_size:].fill_diagonal_(0)
    mask[batch_size:, :].fill_diagonal_(0)
    return mask


class NT_Xent(nn.Module):
    """
    https://arxiv.org/abs/2002.05709
    Modified from https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/nt_xent.py
    """
    def __init__(self, batch_size, temperature=0.1):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = _mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")


    def forward(self, z_i, z_j):
        """
        Standard contrastive loss on [z_i, z_j]

        param z_i (bsz, d): the stacked g(f(x)) for one augmented view x
        param z_j (bsz, d): the stacked g(f(x')) for the other view x'
        
        returns loss
        """

        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        # https://github.com/Spijkervet/SimCLR/issues/37
        z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
        z_j = torch.cat(GatherLayer.apply(z_j), dim=0)

        batch_size = z_i.size(0)
        N = 2 * batch_size
        
        z = torch.cat((z_i, z_j), dim=0)
        sim = z @ z.t()

        # positives are the similarity between different views of the same example 
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # negatives are the similarity between different examples
        mask = _mask_correlated_samples(batch_size) if batch_size != self.batch_size else self.mask     # accounts for the last batch of the epoch
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1) / self.temperature
        loss = self.criterion(logits, labels)
        loss /= N

        return loss
    

class Cross_NT_Xent(nn.Module):
    """
    Cross-task loss in Osiris
    """
    def __init__(self, batch_size, temperature=0.1):
        super(Cross_NT_Xent, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j, u_i, u_j):
        """
        Contrastive loss for discriminating z and u
        No comparison between examples within z or u

        param z_i (bsz, d): the stacked h(f(x)) for one augmented view x from the current task
        param z_j (bsz, d): the stacked h(f(x')) for the other view x' from the current task
        param u_i (p*bsz, d): the stacked h(f(y)) for one augmented view y from the memory
        param u_j (p*bsz, d): the stacked h(f(y')) for the other view y' from the memory
        
        returns loss
        """

        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)
        u_i = F.normalize(u_i, p=2, dim=1)
        u_j = F.normalize(u_j, p=2, dim=1)

        z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
        z_j = torch.cat(GatherLayer.apply(z_j), dim=0)
        u_i = torch.cat(GatherLayer.apply(u_i), dim=0)
        u_j = torch.cat(GatherLayer.apply(u_j), dim=0)

        batch_size = z_i.size(0)
        N = batch_size * 2

        # positives are the similarity between different views of the same example within z
        positive_samples = torch.sum(z_i*z_j, dim=-1).repeat(2).reshape(N, 1)

        # negatives are comparisons between the examples in z and the ones in u
        z = torch.cat([z_i, z_j], dim=0)
        u = torch.cat([u_i, u_j], dim=0)
        negative_samples = z @ u.t()

        # loss
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat([positive_samples, negative_samples], dim=1) / self.temperature
        loss_zu = self.criterion(logits, labels)
        loss_zu /= N
        
        # for a symmetric loss, switch z and u
        # we do not need to recompute the similarity matrix between z and u
        # simply use the columns rather than the rows of the matrix as negatives
        batch_size = u_i.size(0)
        N = batch_size * 2
        positive_samples = torch.sum(u_i*u_j, dim=-1).repeat(2).reshape(N, 1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat([positive_samples, negative_samples.t()], dim=1) / self.temperature
        loss_uz =  self.criterion(logits, labels)
        loss_uz /= N

        # final cross-task loss
        loss = 0.5 * (loss_zu + loss_uz)

        return loss
    
class Distill_NT_Xent(nn.Module):
    """
    Distillation loss in Osiris-D
    """
    def __init__(self, batch_size, temperature=0.1):
        super(Distill_NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = _mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        """
        Contrastive loss for using z_i to predict data from z_j, and the other way around
        No comparison between examples within z_i or z_j

        param z_i (2*bsz, d): the stacked h(f([x, x'])) for both views of examples in the current task encoded by f
        param z_j (2*bsz, d): the stacked h(f'([x, x'])) for both views of examples in the current task encoded by an alternative encoder f'

        returns loss
        """

        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
        z_j = torch.cat(GatherLayer.apply(z_j), dim=0)

        N = z_i.size(0)
        batch_size = N // 2

        sim = z_i @ z_j.t()

        # positives are the similarity between the same example encoded by different encoders
        positive_samples = torch.diag(sim).reshape(N, 1)

        # negatives are the similarity with other examples 
        # we still need to mask because there are two views of each example in each of z_i and z_j
        mask = _mask_correlated_samples(batch_size) if batch_size != self.batch_size else self.mask     # accounts for the last batch of the epoch
        negative_samples = sim[mask].reshape(N, -1)

        # loss
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1) / self.temperature
        loss_ij = self.criterion(logits, labels)
        loss_ij /= N

        # for a symmetric loss, switch z_i and z_j
        # we do not need to recompute the similarity matrix between z_i and z_j
        # simply use the columns rather than the rows of the matrix as negatives
        negative_samples = sim.t()[mask].reshape(N, -1)
        logits = torch.cat((positive_samples, negative_samples), dim=1) / self.temperature
        loss_ji = self.criterion(logits, labels)
        loss_ji /= N

        # final distillation loss
        loss = 0.5 * (loss_ij + loss_ji)

        return loss
    

class SimCLR(nn.Module):
    """
    https://arxiv.org/abs/2002.05709
    This is the FT/Offline model in https://arxiv.org/abs/2404.19132
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = resnet18(zero_init_residual=True, norm=args.norm, act=args.act)
        self.backbone.fc = nn.Identity()        # we do not need a linear classifier
        self.criterion = NT_Xent(args.batch_size)

        sizes = [512] + list(map(int, args.projector.split('-')))
        self.projector = _make_projector(sizes)

    def forward(self, x1, x2):
        """
        param x1 (bsz, ...): augmented views of examples
        param x2 (bsz, ...): another set of augmented views of the same examples
        
        returns loss
        """

        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))

        loss = self.criterion(z1, z2)
        return loss, [loss.item()]
    

class Osiris(nn.Module):
    """
    https://arxiv.org/abs/2404.19132
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = resnet18(zero_init_residual=True, norm=args.norm, act=args.act)
        self.backbone.fc = nn.Identity()    # we do not need a linear classifier

        batch_size_Y = int(args.batch_size // args.world_size * args.p) * args.world_size
        batch_size_X = args.batch_size

        self.criterion_curr = NT_Xent(batch_size_X)
        self.criterion_cross = Cross_NT_Xent(batch_size_X)
        self.criterion_replay = NT_Xent(batch_size_Y)
        self.criterion_distill = Distill_NT_Xent(batch_size_X)

        # projector g
        sizes = [512] + list(map(int, args.projector.split('-')))
        self.projector = _make_projector(sizes)

        # predictor h
        self.predictor = _make_projector(sizes)
        for param in self.predictor.parameters():
            param.requires_grad = False


    def forward(self, xy1, xy2, z1_prev, z2_prev, mem_idx, task):
        """
        param xy1 (bsz+p*bsz, ...): stacked augmented images from the current task (X) and the memory (Y)
        param xy2 (bsz+p*bsz, ...): same images with another sample of augmentations, i.e., X' U Y'
        param z1_prev (bsz, d): X encoded by some previous checkpoint, or None
        param z2_prev (bsz, d): X' encoded by some previous checkpoint, or None
        param mem_idx (bsz+p*bsz): boolean index mask for xy1 and xy2 which gives the index of memory examples, for convenience only
        param task: current task index; this is only used for deciding whether to use vanilla SimCLR (in the first task) or Osiris (subsequent tasks)
        
        returns loss
        """
        
        zu1 = self.backbone(xy1)
        zu2 = self.backbone(xy2)

        if task == 0:
            # the first task
            z1 = self.projector(zu1)
            z2 = self.projector(zu2)
            loss = self.criterion_curr(z1, z2)
            loss1, loss2, loss3 = loss.item(), 0, 0

        else:
            z1, z2 = zu1[~mem_idx], zu2[~mem_idx]
            u1, u2 = zu1[mem_idx], zu2[mem_idx]

            # current task loss
            # on space 1 (i.e., with g o f)
            z1_s1 = self.projector(z1)
            z2_s1 = self.projector(z2)
            loss1 = self.criterion_curr(z1_s1, z2_s1)

            # cross-task loss
            # on space 2 (i.e., with h o f)
            z1_s2 = self.predictor(z1)
            z2_s2 = self.predictor(z2)
            u1_s2 = self.predictor(u1)
            u2_s2 = self.predictor(u2)
            loss2 = self.criterion_cross(z1_s2, z2_s2, u1_s2, u2_s2)

            # past-task loss
            # also on space 2 (i.e., with h o f)
            if self.args.model == 'osiris-d':
                z1_prev = self.predictor(z1_prev)
                z2_prev = self.predictor(z2_prev)
                z_prev = torch.cat([z1_prev, z2_prev], dim=0)
                z = torch.cat([z1_s2, z2_s2], dim=0)
                loss3 = self.criterion_distill(z, z_prev)
            
            elif self.args.model == 'osiris-r':
                loss3 = self.criterion_replay(u1_s2, u2_s2)

            else:
                raise NotImplementedError

            loss = loss1 + 0.5 * (loss2 + loss3)    # overall loss
            loss1, loss2, loss3 = loss1.item(), loss2.item(), loss3.item()      # for logging

        return loss, [loss1, loss2, loss3]







