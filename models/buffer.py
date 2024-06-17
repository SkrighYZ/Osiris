# Modified from https://github.com/divyam3897/UCL/blob/main/utils/buffer.py

import torch
import numpy as np
from utils.loading_utils import BufferSSLTransform


def reservoir(num_seen_examples, buffer_size):
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples+1)
    return (rand if rand < buffer_size else -1)


class Buffer:
    """
    The memory buffer of replay-based methods.
    """
    def __init__(self, args):
        self.args = args
        self.buffer_size = args.buffer_size
        self.num_seen_examples = 0

        if args.dataset == 'cifar100':
            self.transform = BufferSSLTransform(in_size=32, mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        elif args.dataset == 'tinyimagenet':
            self.transform = BufferSSLTransform(in_size=64, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    @torch.no_grad()
    def add_data(self, examples):
        # Add a batch of examples into memory

        if not hasattr(self, 'examples'):
            self.examples = torch.zeros((self.buffer_size, *examples.shape[1:]), dtype=torch.float32, requires_grad=False).cpu()
            
        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            if index >= 0:
                self.examples[index] = examples[i].cpu()
            self.num_seen_examples += 1


    @torch.no_grad()
    def get_data(self, size, segment):
        # Sample a set of augmented images from memory
        # For DDP, all processes sample the same set of data from the memory 
        # But each process only augments a subset (segment) of the selected memory and encodes it
        # This prevents any two processes to replay the same example in DDP and distributes data augmentation to different processes.

        assert self.num_seen_examples > self.buffer_size
        assert size <= self.examples.shape[0]   
        choice = np.random.choice(self.examples.shape[0], size=size, replace=False)

        selected_x1, selected_x2 = [], []
        for ex in self.examples[choice][segment[0]:segment[1]]:
            ex1, ex2 = self.transform(ex)
            selected_x1.append(ex1)
            selected_x2.append(ex2)
            
        selected_x1 = torch.stack(selected_x1).cuda(non_blocking=True)
        selected_x2 = torch.stack(selected_x2).cuda(non_blocking=True)

        return selected_x1, selected_x2
