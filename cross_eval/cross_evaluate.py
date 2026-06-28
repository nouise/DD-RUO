"""Cross-architecture evaluation for distilled datasets.

Loads pre-saved synthetic images/labels (.pt) and evaluates classification
accuracy across multiple network architectures. Shared across TM/DM/DC.

Usage:
    python cross_evaluate.py --images_path images.pt --labels_path labels.pt --subset imagenette
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch

from core.utils import (get_dataset, get_network, evaluate_synset,
                        get_time, ParamDiffAug, set_seed, save_and_print)

EVAL_LRS = {
    "ResNet18ImageNet": 0.02,
    "VGG11": 0.02,
    "AlexNet": 0.02,
    "ViT": 0.02,
}


def main(args):
    args.dsa = True
    args.zca = False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    channel, im_size, num_classes, class_names, mean, std, \
        dst_train, dst_test, testloader, loader_train_dict, \
        class_map, class_map_inv = get_dataset(
            args.dataset, args.data_path, args.batch_real, args.subset, args=args)

    args.channel, args.im_size, args.num_classes = channel, im_size, num_classes
    args.mean, args.std = mean, std
    args.dc_aug_param = None
    args.dsa_param = ParamDiffAug()
    args.zca_trans = None

    model_eval_pool = args.models.split(',')

    save_and_print(args.log_path, f"Cross-architecture evaluation")
    save_and_print(args.log_path, f"Models: {model_eval_pool}")
    save_and_print(args.log_path, f"Dataset: {args.dataset}/{args.subset}, num_classes={num_classes}")

    image_syn = torch.load(args.images_path, map_location=args.device)
    label_syn = torch.load(args.labels_path, map_location=args.device)
    save_and_print(args.log_path, f"Loaded: images={image_syn.shape}, labels={label_syn.shape}")

    results = {}
    for model_name in model_eval_pool:
        args.lr_net = EVAL_LRS.get(model_name, 0.01)
        save_and_print(args.log_path, f"\n{'='*60}")
        save_and_print(args.log_path, f"Evaluating {model_name} (lr={args.lr_net})")

        accs = []
        for i in range(args.num_eval):
            net = get_network(model_name, channel, num_classes, im_size).to(args.device)
            _, _, acc_test = evaluate_synset(
                i, net, image_syn, label_syn, testloader, args)
            accs.append(acc_test)
            del net

        accs = np.array(accs)
        acc_mean, acc_std = np.mean(accs), np.std(accs)
        results[model_name] = (acc_mean, acc_std)
        save_and_print(args.log_path,
            f"{model_name}: {acc_mean:.4f} +/- {acc_std:.4f} (n={args.num_eval})")

    save_and_print(args.log_path, f"\n{'='*60}")
    save_and_print(args.log_path, "Summary:")
    for model_name, (mean, std) in results.items():
        save_and_print(args.log_path, f"  {model_name:15s}: {mean:.4f} +/- {std:.4f}")
    avg_acc = np.mean([v[0] for v in results.values()])
    save_and_print(args.log_path, f"  {'Average':15s}: {avg_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cross-architecture evaluation for distilled datasets')

    parser.add_argument('--images_path', type=str, required=True,
                        help='path to synthetic images .pt file')
    parser.add_argument('--labels_path', type=str, required=True,
                        help='path to synthetic labels .pt file')

    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--subset', type=str, default='imagenette')
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--batch_real', type=int, default=256)
    parser.add_argument('--batch_train', type=int, default=256)
    parser.add_argument('--epoch_eval_train', type=int, default=1000)

    parser.add_argument('--models', type=str,
                        default='ResNet18ImageNet,VGG11,AlexNet,ViT',
                        help='comma-separated list of architectures to evaluate')
    parser.add_argument('--num_eval', type=int, default=5,
                        help='number of evaluation runs per architecture')
    parser.add_argument('--dsa_strategy', type=str,
                        default='color_crop_cutout_flip_scale_rotate')

    parser.add_argument('--save_dir', type=str, default='./results/cross_eval')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    args.log_path = os.path.join(args.save_dir, "log.txt")
    save_and_print(args.log_path, f"Begin at: {get_time()}")
    main(args)
