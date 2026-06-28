"""Post-quantization evaluation for dataset distillation with neural image compression.

Pipeline (single pass per MSE threshold):
  Load pool checkpoint -> quantize_net(mse_err) -> evaluate accuracy -> save
  images_{mse}.pt / labels_{mse}.pt / pool.pt.

Shared across TM / DM / DC: only consumes a pool checkpoint, method-agnostic.
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torchvision.utils
import matplotlib.pyplot as plt

from core.ts.tensor_pool import TensorPool
from core.utils import (get_dataset, get_network, get_eval_pool, evaluate_synset,
                        get_time, ParamDiffAug, set_seed, save_and_print, get_daparam)


def save_grid(images, nrow, path, dataset, normalize=True, scale_each=True):
    if dataset != "ImageNet":
        images = torch.repeat_interleave(images, repeats=4, dim=2)
        images = torch.repeat_interleave(images, repeats=4, dim=3)
    grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=normalize, scale_each=scale_each)
    plt.imshow(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)))
    plt.savefig(path, dpi=300)
    plt.close()


def save_visualizations(images, num_classes, args, save_dir, tag):
    if args.ipc >= 600 and not args.force_save:
        return

    indices = np.concatenate([
        c * args.ipc + np.arange(min(10, args.ipc)) for c in range(num_classes)
    ])
    selected = images[indices]

    save_grid(selected, num_classes, f"{save_dir}/Synthetic_Images_{tag}.png", args.dataset)

    img_std, img_mean = torch.std(images), torch.mean(images)
    clipped = torch.clip(images, min=img_mean - 2.5 * img_std, max=img_mean + 2.5 * img_std)
    save_grid(clipped[indices], num_classes, f"{save_dir}/Clipped_Synthetic_Images_{tag}.png", args.dataset)


def main(args):
    save_and_print(args.log_path, f"CUDNN STATUS: {torch.backends.cudnn.enabled}")

    args.dsa = (args.dsa == 'True')
    args.zca = (args.zca == 'True')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    channel, im_size, num_classes, class_names, mean, std, \
        dst_train, dst_test, testloader, loader_train_dict, \
        class_map, class_map_inv = get_dataset(
            args.dataset, args.data_path, args.batch_real, args.subset, args=args)

    args.channel, args.im_size, args.num_classes = channel, im_size, num_classes
    args.mean, args.std = mean, std

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    if args.dsa:
        args.dc_aug_param = None
    else:
        # epoch() reads args.dc_aug_param when aug=True and dsa=False,
        # so it must always be set (see pool_dc.py / get_daparam).
        args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, args.ipc)
    args.dsa_param = ParamDiffAug()
    if not args.zca:
        args.zca_trans = None
    args.distributed = torch.cuda.device_count() > 1

    save_and_print(args.log_path, f'Hyper-parameters: {args.__dict__}')
    save_and_print(args.log_path, f'Evaluation model pool: {model_eval_pool}')

    # Initialize TensorPool
    sample_per_cls = [args.ipc] * num_classes
    slice_size = 200
    gpu_list = list(range(torch.cuda.device_count()))
    pool = TensorPool(
        num_classes, slice_size, sample_per_cls, gpu_list,
        ldb=args.ldb, img_size=im_size, nthread=num_classes,
        max_iter=500 * 10000, channel=channel, lr=args.lr_img,
        layers_v=args.layers_v, arm=args.arm, dim=args.dim)

    mean_ts = torch.Tensor(mean).view(1, channel, 1, 1).to(args.device)
    std_ts = torch.Tensor(std).view(1, channel, 1, 1).to(args.device)
    syn_lr = torch.tensor(args.syn_lr_set, device=args.device)

    save_and_print(args.log_path, f'{get_time()} quantization evaluation begins')

    # Load pre-trained pool checkpoint
    pool.load_slice_pool(args.pool_path)
    save_and_print(args.log_path, f"Pool loaded from {args.pool_path}")
    pool.set_training_phase(0)
    pool.init_solvers()

    save_dir = args.save_path
    mse_threshold = args.mse_err

    # --- Post-training quantization of the synthesis network ---
    save_and_print(args.log_path, "=" * 60)
    save_and_print(args.log_path, f"Quantizing (mse_threshold={mse_threshold})")
    final_bpp, distribution = pool.quantize_net(mse_threshold=mse_threshold)
    formatted_output = ", ".join(
        f"{k}={v:.6f} ({v / final_bpp * 100:.2f}%)" for k, v in distribution.items())
    save_and_print(args.log_path,
        f"Quantized: bpp={final_bpp:.6f}, "
        f"bits/class={final_bpp * args.ipc * im_size[0] * im_size[1]:.1f}")
    save_and_print(args.log_path, f"  {formatted_output}")

    # --- Evaluate accuracy on the quantized synthetic data ---
    best_acc = {m: 0.0 for m in model_eval_pool}
    best_std = {m: 0.0 for m in model_eval_pool}

    pool.test()
    image_syn_eval, label_syn_eval, rt = pool.get_data()
    image_syn_eval = (image_syn_eval - mean_ts.to(image_syn_eval.device)) / std_ts.to(image_syn_eval.device)
    save_and_print(args.log_path, f"Dataset size: {image_syn_eval.shape}, rt: {rt}")

    for model_eval in model_eval_pool:
        save_and_print(args.log_path,
            f'Evaluation: model_train={args.model}, model_eval={model_eval}')
        if args.dsa:
            save_and_print(args.log_path, f'DSA strategy: {args.dsa_strategy}')

        accs_test = []
        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
            args.lr_net = syn_lr.item()
            _, _, acc_test = evaluate_synset(
                it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
            accs_test.append(acc_test)

        accs_test = np.array(accs_test)
        acc_mean, acc_std = np.mean(accs_test), np.std(accs_test)
        if acc_mean > best_acc[model_eval]:
            best_acc[model_eval] = acc_mean
            best_std[model_eval] = acc_std

        save_and_print(args.log_path,
            f'Evaluate {len(accs_test)} random {model_eval}: '
            f'mean={acc_mean:.4f} std={acc_std:.4f}')
        save_and_print(args.log_path,
            f'  Best: {best_acc[model_eval]:.4f} +/- {best_std[model_eval]:.4f}')

    save_and_print(args.log_path, f"Synthetic LR: {syn_lr.detach().cpu()}")

    # --- Save quantized synthetic data and visualizations ---
    with torch.no_grad():
        pool.test()
        image_save, label_save, rt = pool.get_data()
        image_norm = (image_save - mean_ts.to(image_save.device)) / std_ts.to(image_save.device)

        pool.save_slice_pool(os.path.join(save_dir, "pool.pt"))
        torch.save(image_norm.cpu(), os.path.join(save_dir, f"images_{mse_threshold}.pt"))
        torch.save(label_save.cpu(), os.path.join(save_dir, f"labels_{mse_threshold}.pt"))

        tag = f"{mse_threshold}_{args.img_version}"
        save_visualizations(image_norm, num_classes, args, save_dir, tag)
        del image_norm

    # --- Summary ---
    save_and_print(args.log_path, "=" * 60)
    save_and_print(args.log_path, "Summary:")
    save_and_print(args.log_path, f"  MSE threshold: {mse_threshold}")
    save_and_print(args.log_path, f"  BPP: {final_bpp:.6f}")
    save_and_print(args.log_path,
        f"  Bits/class: {final_bpp * args.ipc * im_size[0] * im_size[1]:.1f}")
    for k, v in distribution.items():
        save_and_print(args.log_path, f"    {k}: {v:.6f}")
    for m in model_eval_pool:
        save_and_print(args.log_path,
            f"  {m}: best_acc={best_acc[m]:.4f} +/- {best_std[m]:.4f}")
    save_and_print(args.log_path, f"Results saved to {save_dir}")
    pool.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Post-quantization evaluation for distilled datasets')

    # Dataset
    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--subset', type=str, default='imagenette',
                        help='ImageNet subset (only used when --dataset=ImageNet)')
    parser.add_argument('--data_path', type=str, default='../data',
                        help='path to dataset root')
    parser.add_argument('--ipc', type=int, default=102, help='images per class')

    # Evaluation
    parser.add_argument('--model', type=str, default='ConvNetD5',
                        help='evaluation model architecture')
    parser.add_argument('--eval_mode', type=str, default='S')
    parser.add_argument('--num_eval', type=int, default=5,
                        help='number of evaluation runs')
    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help='epochs to train evaluation model on synthetic data')
    parser.add_argument('--batch_real', type=int, default=256,
                        help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256,
                        help='batch size for training evaluation networks')
    parser.add_argument('--syn_lr_set', type=float, default=0.01,
                        help='learning rate for evaluation training')

    # Augmentation
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='use differentiable Siamese augmentation')
    parser.add_argument('--dsa_strategy', type=str,
                        default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--zca', type=str, default='False', choices=['True', 'False'],
                        help='use ZCA whitening')

    # TensorPool (synthesis network)
    parser.add_argument('--layers_v', type=str, default='v5',
                        help='synthesis network layer version')
    parser.add_argument('--arm', type=int, default=32,
                        help='ARM context model size')
    parser.add_argument('--dim', type=int, default=4,
                        help='latent dimension')
    parser.add_argument('--ldb', type=float, default=0.1,
                        help='rate-distortion tradeoff lambda')
    parser.add_argument('--lr_img', type=float, default=0.01)

    # Quantization
    parser.add_argument('--mse_err', type=float, default=0.0000005,
                        help='MSE threshold for post-training quantization')
    parser.add_argument('--pool_path', type=str, required=True,
                        help='path to pre-trained pool checkpoint (.pt)')

    # Output
    parser.add_argument('--FLAG', type=str, default='quantize_eval',
                        help='experiment identifier for output directory')
    parser.add_argument('--save_path', type=str, default='./results/quantize',
                        help='base directory for saving results')
    parser.add_argument('--force_save', action='store_true',
                        help='save visualization images even for large ipc')
    parser.add_argument('--img_version', type=str, default='v1',
                        help='version tag for saved image filenames')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    set_seed(args.seed)

    args.save_path = os.path.join(args.save_path, args.FLAG)
    os.makedirs(args.save_path, exist_ok=True)

    args.log_path = os.path.join(args.save_path, "log.txt")
    save_and_print(args.log_path, f"Begin at: {get_time()}")
    main(args)
