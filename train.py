import os
import argparse
import json
import time
import numpy as np
import random
from pathlib import Path
import pickle
from copy import deepcopy
from torch import optim, nn
import torch

from models.contrastive import SimCLR, Osiris
from models.continual_model import ContinualModel

from utils.loading_utils import get_stream_datasets, get_knn_data_loaders
from utils.knn_utils import knn_test

import wandb


def main_worker(gpu, args):
	args.rank = gpu
	torch.distributed.init_process_group(
		backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)

	set_seed(args)

	if args.rank == 0:
		# logging
		wandb.init(
			project="ucl",
			config={
				"model": args.model,
				"dataset": args.dataset,
				"data_order": args.order_fp,
				"epochs": args.epochs,
				"lr": args.lr
			},
			job_type=f'train_{args.dataset}'
		)

		# loaders for accuracy monitoring
		knn_train_loader, knn_test_loader = get_knn_data_loaders(args, task_label=False)
		class_to_idx = knn_train_loader.dataset.class_to_idx
		class_order = pickle.load(open(args.order_fp, 'rb'))
		class_order = [class_to_idx[c] for c in class_order]

	dataset = get_stream_datasets(args)
	dataset.update_order(task=0) 	# call this here so that length of dataloader is correct
	sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
	assert args.batch_size % args.world_size == 0
	per_gpu_batch_size = args.batch_size // args.world_size
	train_loader = torch.utils.data.DataLoader(dataset, 
		batch_size=per_gpu_batch_size, num_workers=args.num_workers, pin_memory=True, sampler=sampler, 
		persistent_workers=True)		# we don't want workers to create copies of the dataset
	
	
	steps_per_task = len(train_loader) * args.epochs


	if 'simclr' in args.model:
		model = SimCLR(args)
	elif 'osiris' in args.model:
		model = Osiris(args)
	else:
		raise NotImplementedError
	
	use_cl_wrapper = bool('osiris' in args.model)
	if use_cl_wrapper:
		model = ContinualModel(args, model)

	# https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
	torch.cuda.set_device(args.rank)
	torch.cuda.empty_cache()

	model = model.cuda(args.rank)
	model = nn.SyncBatchNorm.convert_sync_batchnorm(model)


	param_weights = []
	param_biases = []
	named_parameters = model.net.named_parameters() if use_cl_wrapper else model.named_parameters()
	for n, param in named_parameters:
		if param.ndim == 1: 
			param_biases.append(param)
		else: param_weights.append(param)
	parameters = [{'params': param_weights, 'lr': args.lr, 'weight_decay': args.weight_decay}, 
				{'params': param_biases, 'lr': args.lr, 'weight_decay': 0.0}]
	optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


	model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank])
	backbone = model.module.net.backbone if use_cl_wrapper else model.module.backbone
	projector = model.module.net.projector if use_cl_wrapper else model.module.projector


	###################
	# Task outer loop #
	###################
	start_time = time.time()
	loss_running = 0

	for task in range(args.num_tasks):
			
		if args.rank == 0:
			print('Task', task)

		if task != 0:
			dataset.update_order(task)
			train_loader = torch.utils.data.DataLoader(dataset, 
				batch_size=per_gpu_batch_size, num_workers=args.num_workers, pin_memory=True, sampler=sampler, 
				persistent_workers=True)	# we don't want workers to create copies of the dataset

		if 'osiris' in args.model:
			model.module.update_model_states(task)

		model.train()

		###################
		# Epoch inner loop #
		###################
		for epoch in range(args.epochs):

			sampler.set_epoch(epoch)

			for step, (img, x, _) in enumerate(train_loader, start=epoch*len(train_loader)):

				global_step = step + task * steps_per_task + 1

				x1, x2 = x[0].cuda(non_blocking=True), x[1].cuda(non_blocking=True)
				img = img.cuda(non_blocking=True)

				optimizer.zero_grad()

				# different ranks should share the same seed for distributed replay to work
				# print(step, args.rank, random.randint(1, 1000))

				loss, loss_vals = model(x1, x2, img, task) if use_cl_wrapper else model(x1, x2)

				loss.backward()
				optimizer.step()

				loss_running += np.array(loss_vals)

				if args.rank == 0:
					if global_step % args.log_freq == 0:
						stats = dict(task=task,
									lr=optimizer.param_groups[0]['lr'],
									time=int(time.time() - start_time))
						for li, l in enumerate(loss_running):
							stats[f'loss_{li}'] = l / args.log_freq
						wandb.log(stats)
						loss_running = 0

					# eval for the offline model
					if args.num_tasks == 1 and (global_step+1) % args.eval_freq == 0:
						model.eval()
						task_accs = knn_test(args, backbone, knn_train_loader, knn_test_loader, args.num_tasks, class_order)
						model.train()
						wandb.log(dict(acc=sum(task_accs)/len(task_accs)))	# assumes uniform p(t) 
						
					torch.distributed.barrier()
				
				else:
					torch.distributed.barrier()
						
		# end of task eval for the continual models
		if args.rank == 0 and args.num_tasks > 1:
			model.eval()
			torch.save(backbone.state_dict(), args.save_dir / f'backbone_{task}.pt')
			task_accs = knn_test(args, backbone, knn_train_loader, knn_test_loader, args.num_tasks, class_order)
			model.train()
			wandb.log(dict(acc=sum(task_accs)/len(task_accs)))	# assumes uniform p(t) 

			torch.distributed.barrier()
		
		else:
			torch.distributed.barrier()

	# save the final checkpoint
	if args.rank == 0:
		torch.save(backbone.state_dict(), args.save_dir / 'resnet18.pt')


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset', type=str, default='tinyimagenet', choices=['cifar100', 'tinyimagenet'])
	parser.add_argument('--num_tasks', type=int, default=20)
	parser.add_argument('--model', type=str, choices=['simclr', 'osiris-r', 'osiris-d'])

	parser.add_argument('--epochs', type=int, default=200)
	parser.add_argument('--eval_freq', type=int, default=1955, help='number of steps between eval; only used for the offline model')
	parser.add_argument('--log_freq', default=200, type=int, metavar='N', help='number of steps between logging')
	parser.add_argument('--eval_batch_size', type=int, default=64)
	parser.add_argument('--knn_k', type=int, default=200)
	parser.add_argument('--knn_t', type=float, default=0.1)

	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--lr', type=float, default=0.03)
	parser.add_argument('--weight_decay', type=float, default=5e-4)
	parser.add_argument('--momentum', default=0.9, type=float)

	parser.add_argument('--norm', type=str, default='gn', choices=['bn', 'gn'])
	parser.add_argument('--act', type=str, default='mish', choices=['relu', 'mish'])
	parser.add_argument('--projector', default='2048-128', type=str, metavar='MLP', help='projector MLP')
	parser.add_argument('--buffer_size', type=int, default=500)
	parser.add_argument('--p', type=float, default=0.75, help='number of memory samples wrt batch size')

	parser.add_argument('--data_dir', type=Path, metavar='DIR')
	parser.add_argument('--save_dir', type=Path, metavar='DIR')
	parser.add_argument('--order_fp', type=str, help='path to the class ordering file')
	parser.add_argument('--t200_paths', type=str, default=None, help='path to the data path file for tiny imagenet')

	parser.add_argument('--seed', default=8, type=int)
	parser.add_argument('--num_workers', type=int, default=2)

	parser.add_argument("--distributed", action='store_true', help="ddp; always true")
	parser.add_argument('--rank', default=0, type=int)
	parser.add_argument('--local_rank', type=int)
	parser.add_argument('--world_size', default=2, type=int, help='number of gpus')

	args = parser.parse_args()

	print(args)

	set_seed(args)

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	with open(args.save_dir / 'args.txt', 'w') as f:
		args_copy = deepcopy(args.__dict__)
		args_copy['save_dir'] = str(args.save_dir)
		args_copy['data_dir'] = str(args.data_dir)
		json.dump(args_copy, f, indent=2)

	if args.distributed:
		assert torch.distributed.is_available()
		print("PyTorch Distributed available.")
		print("  Backends:")
		print(f"    Gloo: {torch.distributed.is_gloo_available()}")
		print(f"    NCCL: {torch.distributed.is_nccl_available()}")
		print(f"    MPI:  {torch.distributed.is_mpi_available()}")

		torch.multiprocessing.spawn(main_worker, nprocs=args.world_size, args=(args,))

	else:
		raise NotImplementedError


def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)


if __name__ == '__main__':
	main()