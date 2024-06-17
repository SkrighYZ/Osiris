
from PIL import Image
import torch.utils.data as data
import copy, pickle
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


class VerboseSubset(data.Subset):
    """
    Creates a subset of the dataset for each task
    The classes labels (self.targets) are mapped to {0, 1, ..., len(classes)} rather than using the original class labels
    """
    def __init__(self, dataset, indices, classes):
        self.dataset = dataset
        self.indices = indices

        self.classes = classes
        class_map = {c: i for i, c in enumerate(classes)}
        self.targets = [class_map[self.dataset.targets[idx]] for idx in indices]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]][0], self.targets[idx]

    def __len__(self):
        return len(self.indices)



class CIFAR(data.Dataset):
    """
    Split-CIFAR-100 task sequence 
    Classes are ordered based on class_order
    By calling self.update_order(t), switch self.samples to the examples from task t
    """
    def __init__(self, root, transform, num_tasks, class_order):

        file = pickle.load(open(root / 'cifar-100-python/train', 'rb'), encoding='latin1')
        meta = pickle.load(open(root / 'cifar-100-python/meta', 'rb'), encoding='latin1')
        x = file['data'].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1)) # convert to NHWC

        self.classes = meta['fine_label_names']
        self.class_to_idx = {c: _idx for _idx, c in enumerate(self.classes)}
        self.base_samples = [(x[i], file['fine_labels'][i]) for i in range(file['data'].shape[0])]
        
        self.num_tasks = num_tasks
        self.class_order = class_order
        self.classes_per_task = len(class_order) // num_tasks

        self.samples = None        # instantiated by calling update_order()
        self.root = root

        self.transform = transform
        self.to_tensor = transforms.Compose([
                transforms.Resize(32, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img, target = self.samples[index]       # we do not use target in UCL
        img = Image.fromarray(img) 
        aug_img = self.transform(img)
        original_img = self.to_tensor(img)

        return original_img, aug_img, target

    def __len__(self):
        return len(self.samples)

    def update_order(self, task):
        # change self.samples to data in a specific task
        curr_classes = self.class_order[task*self.classes_per_task:task*self.classes_per_task+self.classes_per_task]
        curr_indices = [self.class_to_idx[c] for c in curr_classes]
        self.samples = [sample for sample in copy.deepcopy(self.base_samples) if sample[1] in curr_indices]




class TinyImageNet(data.Dataset):
    """
    Tiny-ImageNet-200 task sequence 
    Classes are ordered based on class_order
    By calling self.update_order(t), switch self.samples to the examples from task t
    """
    def __init__(self, root, transform, num_tasks, class_order, pathfile):

        with open(pathfile, 'r') as f:
            self.sample_paths = f.read().split('\n')

        self.classes = sorted(list(set([s.split('/')[-2] for s in self.sample_paths])))
        self.class_to_idx = {nid: _idx for _idx, nid in enumerate(self.classes)}
        self.base_samples = [(root / p, self.class_to_idx[p.split('/')[-2]]) for p in self.sample_paths]
        
        self.num_tasks = num_tasks
        self.class_order = class_order
        self.classes_per_task = len(class_order) // num_tasks

        self.samples = None
        self.root = root
        self.loader = pil_loader

        self.transform = transform
        self.to_tensor = transforms.Compose([
                transforms.Resize(64, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor()
        ])

    def __getitem__(self, index):
        fpath, target = self.samples[index]         # we do not use target in UCL
        img = self.loader(fpath)
        aug_img = self.transform(img)
        original_img = self.to_tensor(img)

        return original_img, aug_img, target

    def __len__(self):
        return len(self.samples)

    def update_order(self, task):
        # change self.samples to data in a specific task
        curr_classes = self.class_order[task*self.classes_per_task:task*self.classes_per_task+self.classes_per_task]
        curr_indices = [self.class_to_idx[c] for c in curr_classes]
        self.samples = [sample for sample in copy.deepcopy(self.base_samples) if sample[1] in curr_indices]


def pil_loader(path):
    # https://github.com/python-pillow/Pillow/issues/835
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')