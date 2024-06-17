# Skeleton borrowed from https://github.com/divyam3897/UCL/blob/main/models/utils/continual_model.py 

import torch.nn as nn
import torch
import copy

from models.buffer import Buffer
from utils.ddp_utils import concat_all_gather

class ContinualModel(nn.Module):
    """
    Continual learning model wrapper.
    """
    def __init__(self, args, net):
        super(ContinualModel, self).__init__()

        self.net = net
        self.net_prev = None
        self.args = args
        self.buffer = Buffer(args)


    def forward(self, aug_x1, aug_x2, img, task):
        x1, x2, mem_idx = self.recall(aug_x1, aug_x2, task)

        if task > 0 and self.args.model == 'osiris-d':
            z1_prev = self.net_prev.backbone(aug_x1).detach()
            z2_prev = self.net_prev.backbone(aug_x2).detach()

        else:
            z1_prev, z2_prev = None, None

        loss = self.net(x1, x2, z1_prev, z2_prev, mem_idx, task)
        
        img = concat_all_gather(img)
        self.buffer.add_data(img)

        return loss


    @torch.no_grad()
    def recall(self, x1, x2, task):
        # Sample a batch from the memory (buf_x1, buf_x2) and merge it with the incoming batch (x1, x2)

        if task > 0:
            assert self.buffer.num_seen_examples >= self.args.batch_size
            
            per_gpu_k = int(x1.size(0) * self.args.p)   # |Y| <= |X| 
            k = self.args.world_size * per_gpu_k
            start = self.args.rank * per_gpu_k

            buf_x1, buf_x2 = self.buffer.get_data(k, segment=[start, start+per_gpu_k])
            mixed_x1, mixed_x2, mem_idx = self.select(x1, x2, buf_x1, buf_x2)

        else:
            mixed_x1, mixed_x2 = x1, x2
            mem_idx = None

        return mixed_x1, mixed_x2, mem_idx


    @torch.no_grad()
    def select(self, x1, x2, buf_x1, buf_x2):
        # Simply concatenate x and buf_x for each view
        # mem_idx is an index mask that identifies the memory examples
        mixed_x1 = torch.cat([x1, buf_x1], dim=0)
        mixed_x2 = torch.cat([x2, buf_x2], dim=0)
        mem_idx = torch.zeros(x1.size(0)+buf_x1.size(0), dtype=torch.bool)
        mem_idx[x1.size(0):] = True
        return mixed_x1, mixed_x2, mem_idx


    def update_model_states(self, task):
        if task == 0: 
            return

        if task == 1:
            # initialize h from g
            # delete this line if using different architectures for h and g
            self.net.predictor.load_state_dict(self.net.projector.state_dict())
            
            for param in self.net.predictor.parameters():
                param.requires_grad = True

        if task > 0 and self.args.model == 'osiris-d':
            # save checkpoint for f and freeze it
            self.net_prev = copy.deepcopy(self.net)
            for param in self.net_prev.parameters():
                param.requires_grad = False

        return
