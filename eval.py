import os
import argparse
from pathlib import Path
import pickle, json
from tqdm import tqdm
import numpy as np

from torch import nn
import torch

from models.backbone import resnet18

from utils.loading_utils import get_knn_data_loaders
from utils.knn_utils import knn_test, knn_test_wtl, knn_test_task
from utils.metrics import accuracy, forgetting, knowledge_gain, consolidation, forward_transfer


def load_ckpt(args, ckpt=None):
    model = resnet18(norm=args.norm, act=args.act)
    model.fc = nn.Identity()
    if not ckpt is None:
        model.load_state_dict(torch.load(ckpt))
    model.to('cuda:0')
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model

def eval(args):
    per_task_accs = eval_wotl(args) if args.mode == 'wotl' else eval_wtl(args)
    per_task_accs = np.array(per_task_accs, dtype=float)
    tknn_acc = eval_task(args)

    if args.offline:
        results = dict(
            model_dir=str(args.ckpt_dir),
            eval_mode=args.mode,
            A=per_task_accs.mean()*100,
            C=tknn_acc*100
        )
        print(results)

    else:
        random_accs = get_random_accs(args)
        results = dict(
            model_dir=str(args.ckpt_dir),
            eval_mode=args.mode,
            A=accuracy(per_task_accs)*100,
            F=forgetting(per_task_accs)*100,
            K=knowledge_gain(per_task_accs)*100,
            C=tknn_acc*100,
            T=forward_transfer(per_task_accs, random_accs)*100
        )
        print(results)

    with open(args.ckpt_dir / f'results_{args.mode}.txt', 'w') as f:
        json.dump(results, f, indent=2)


"""
KNN evaluation without knowing task labels 
i.e., predicting argmax_y p(y | x)
"""
def eval_wotl(args):
    train_loader, test_loader = get_knn_data_loaders(args, task_label=False)
    class_to_idx = train_loader.dataset.class_to_idx
    class_order = pickle.load(open(args.order_fp, 'rb'))
    class_order = [class_to_idx[c] for c in class_order]

    if args.offline:
        model = load_ckpt(args, args.ckpt_dir / f'resnet18.pt')
        task_accs = knn_test(args, model, train_loader, test_loader, num_tasks=args.num_tasks, class_order=class_order)
        
        return task_accs

    accs = []
    for i in tqdm(range(args.num_tasks)):
        model = load_ckpt(args, args.ckpt_dir / f'backbone_{i}.pt')
        task_accs = knn_test(args, model, train_loader, test_loader, num_tasks=args.num_tasks, class_order=class_order)
        accs.append(task_accs)

    return accs


"""
KNN evaluation with task label given
i.e., predicting argmax_y p(y | x, t)
"""
def eval_wtl(args):
    train_loaders, test_loaders = get_knn_data_loaders(args, task_label=True)

    if args.offline:
        task_accs = []
        model = load_ckpt(args, args.ckpt_dir / f'resnet18.pt')
        for train_loader, test_loader in zip(train_loaders, test_loaders):
            acc = knn_test_wtl(args, model, train_loader, test_loader)
            task_accs.append(acc)
        
        return task_accs

    accs = []
    for i in tqdm(range(args.num_tasks)):
        task_accs = []
        model = load_ckpt(args, args.ckpt_dir / f'backbone_{i}.pt')
        for train_loader, test_loader in zip(train_loaders, test_loaders):
            acc = knn_test_wtl(args, model, train_loader, test_loader)
            task_accs.append(acc)
        accs.append(task_accs)

    return accs


"""
Task-level KNN accuracy
i.e., predicting argmax_t p(t | x)
"""
def eval_task(args):
    train_loader, test_loader = get_knn_data_loaders(args, task_label=False)

    class_to_idx = train_loader.dataset.class_to_idx
    class_order = pickle.load(open(args.order_fp, 'rb'))
    class_order = [class_to_idx[c] for c in class_order]

    model = load_ckpt(args, args.ckpt_dir / f'resnet18.pt')
    acc = knn_test_task(args, model, train_loader, test_loader, num_tasks=args.num_tasks, class_order=class_order)
    return acc


"""
Accuracy of a randomly initialized model
This is used for calculating forward transfer
"""
def get_random_accs(args, num_trials=5):
    accs = 0

    if args.mode == 'wotl':
        train_loader, test_loader = get_knn_data_loaders(args, task_label=False)
        class_to_idx = train_loader.dataset.class_to_idx
        class_order = pickle.load(open(args.order_fp, 'rb'))
        class_order = [class_to_idx[c] for c in class_order]
        
        for i in range(num_trials):
            accs += np.array(knn_test(args, load_ckpt(args), train_loader, test_loader, num_tasks=args.num_tasks, class_order=class_order))
    
    else:
        train_loaders, test_loaders = get_knn_data_loaders(args, task_label=True)

        for i in range(num_trials):
            task_accs = []
            model = load_ckpt(args)
            for train_loader, test_loader in zip(train_loaders, test_loaders):
                acc = knn_test_wtl(args, model, train_loader, test_loader)
                task_accs.append(acc)
            accs += np.array(task_accs)

    return accs / num_trials


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, choices=['cifar100', 'tinyimagenet'])
    parser.add_argument('--order_fp', type=str)
    parser.add_argument('--data_dir', type=Path, metavar='DIR')
    parser.add_argument('--ckpt_dir', type=Path, metavar='DIR')

    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--knn_k', type=int, default=200)
    parser.add_argument('--knn_t', type=float, default=0.1)

    parser.add_argument('--num_tasks', type=int)
    parser.add_argument('--mode', type=str, default='wotl')
    parser.add_argument("--offline", action='store_true', help="whether the evaluated model is the offline model")

    parser.add_argument('--norm', type=str, choices=['bn', 'gn'], default='gn')
    parser.add_argument('--act', type=str, choices=['mish', 'relu'], default='mish')

    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.ckpt_dir):
        raise Exception('Incorrect directory.')

    eval(args)


if __name__ == '__main__':
    main()