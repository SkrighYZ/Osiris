import numpy as np
import random
import pickle

from PIL import Image, ImageOps, ImageFilter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import InterpolationMode

from utils.data_utils import TinyImageNet, VerboseSubset, CIFAR

C100_MEAN = (0.5071, 0.4867, 0.4408)
C100_STD = (0.2675, 0.2565, 0.2761)
T200_MEAN = (0.485, 0.456, 0.406)
T200_STD = (0.229, 0.224, 0.225)

def get_stream_datasets(args):
	"""
	Dataset for UCL training with SSL augmentations
	"""
	class_order = pickle.load(open(args.order_fp, 'rb'))

	if args.dataset == 'cifar100':
		dataset = CIFAR(args.data_dir, 
			transform=SSLTransform(in_size=32, mean=C100_MEAN, std=C100_STD), 
			num_tasks=args.num_tasks, class_order=class_order)

	elif args.dataset == 'tinyimagenet':
		dataset = TinyImageNet(args.data_dir,
			transform=SSLTransform(in_size=64, mean=T200_MEAN, std=T200_STD), 
			num_tasks=args.num_tasks, class_order=class_order, pathfile=args.t200_paths)
	
	return dataset


def get_knn_data_loaders(args, task_label=False):
	"""
	Loaders for KNN evaluation
	No augmentation
	No shuffling
	"""
	if args.dataset == 'cifar100':
		test_transform = transforms.Compose([
				transforms.Resize(32, interpolation=InterpolationMode.BICUBIC),
				transforms.ToTensor(),
				transforms.Normalize(mean=C100_MEAN, std=C100_STD)
		])
		trainset = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=test_transform)
		testset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=test_transform)
	
	elif args.dataset == 'tinyimagenet':
		test_transform = transforms.Compose([
				transforms.Resize(64, interpolation=InterpolationMode.BICUBIC),
				transforms.ToTensor(),
				transforms.Normalize(mean=T200_MEAN, std=T200_STD)
		])

		trainset = datasets.ImageFolder(root=args.data_dir / 'tiny-imagenet-200/train', transform=test_transform)
		testset = datasets.ImageFolder(root=args.data_dir / 'tiny-imagenet-200/val', transform=test_transform)

	else:
		raise NotImplementedError

	if task_label:
		# get separate loaders for each task
		train_loaders = []
		test_loaders = []
		class_order = pickle.load(open(args.order_fp, 'rb'))
		classes_per_task = len(class_order) // args.num_tasks
		
		for t_i in range(args.num_tasks):
			class_names = class_order[t_i*classes_per_task:t_i*classes_per_task+classes_per_task]
			classes = [trainset.class_to_idx[c] for c in class_names]
			
			# obtain index of examples that belong to task t_i
			targets = np.array(trainset.targets)
			idx = (targets == classes[0])
			for class_idx in classes[1:]:
				idx |= (targets == class_idx)
			idx = [i for i, x in enumerate(idx) if x]

			# create a sub-dataset for task t_i
			task_trainset = VerboseSubset(trainset, idx, classes=classes)
			train_loaders.append(
				DataLoader(task_trainset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True))

			# do the same thing for the test set
			targets = np.array(testset.targets)
			idx = (targets == classes[0])
			for class_idx in classes[1:]:
				idx |= (targets == class_idx)
			idx = [i for i, x in enumerate(idx) if x]
			task_testset = VerboseSubset(testset, idx, classes=classes)
			test_loaders.append(
				DataLoader(task_testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True))

		return train_loaders, test_loaders

	else:
		# one single pair of loaders for all tasks
		train_loader = DataLoader(trainset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
		test_loader = DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

		return train_loader, test_loader



class GaussianBlur(object):
	def __init__(self, p):
		self.p = p

	def __call__(self, img):
		if random.random() < self.p:
			sigma = random.random() * 1.9 + 0.1
			return img.filter(ImageFilter.GaussianBlur(sigma))
		else:
			return img


class Solarization(object):
	def __init__(self, p):
		self.p = p

	def __call__(self, img):
		if random.random() < self.p:
			return ImageOps.solarize(img)
		else:
			return img


class SSLTransform:
	"""
    https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
	def __init__(self, in_size, mean, std, min_crop_ratio=0.08):
		self.transform1 = transforms.Compose([
			transforms.RandomResizedCrop(in_size, scale=(min_crop_ratio, 1), interpolation=InterpolationMode.BICUBIC),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomApply(
				[transforms.ColorJitter(brightness=0.4, contrast=0.4,
										saturation=0.2, hue=0.1)],
				p=0.8
			),
			transforms.RandomGrayscale(p=0.2),
			GaussianBlur(p=1.0), 
			Solarization(p=0.0),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std)
		])
		self.transform2 = transforms.Compose([
			transforms.RandomResizedCrop(in_size, scale=(min_crop_ratio, 1), interpolation=InterpolationMode.BICUBIC),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomApply(
				[transforms.ColorJitter(brightness=0.4, contrast=0.4,
										saturation=0.2, hue=0.1)],
				p=0.8
			),
			transforms.RandomGrayscale(p=0.2),
			GaussianBlur(p=0.1),
			Solarization(p=0.2),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std)
		])

	def __call__(self, x):
		y1 = self.transform1(x)
		y2 = self.transform2(x)
		return y1, y2


class BufferSSLTransform:
    """
    Adds ToPILImage() on top of SSLTransform
    """
    def __init__(self, in_size, mean, std, min_crop_ratio=0.08):
        self.transform1 = transforms.Compose([
        	transforms.ToPILImage(),
            transforms.RandomResizedCrop(in_size, scale=(min_crop_ratio, 1), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0), 
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.transform2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(in_size, scale=(min_crop_ratio, 1), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        y1 = self.transform1(x)
        y2 = self.transform2(x)
        return y1, y2
