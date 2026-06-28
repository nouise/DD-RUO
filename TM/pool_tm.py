import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
from tqdm import tqdm
from core.ts.tensor_pool import TensorPool
from core.utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug, set_seed, save_and_print, get_images
import copy
import random
from reparam_module import ReparamModule
import shutil
import matplotlib.pyplot as plt
from hyper_params import load_default

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_ref_dict(images_all,indices_class,args,mean,std,channel):
    mean_ts = torch.Tensor(mean).view(1, channel, 1, 1)
    std_ts = torch.Tensor(std).view(1, channel, 1, 1)
    if not args.zca:    
        ref_dict = {c: get_images(images_all, indices_class, c, args.ipc).detach() * std_ts + mean_ts for c in range(args.num_classes)}
    else:
        ref_dict = {c:get_images(images_all, indices_class, c, args.ipc).detach().data for c in range(args.num_classes)}
    return ref_dict
def main(args):
    torch.autograd.set_detect_anomaly(True)

    save_and_print(args.log_path, "CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.zca = True if args.zca == 'True' else False

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    save_and_print(args.log_path,f"eval_it_pool:{eval_it_pool}")
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    args.channel, args.im_size, args.num_classes, args.mean, args.std = channel, im_size, num_classes, mean, std
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    if args.dsa:
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    args.distributed = torch.cuda.device_count() > 1

    save_and_print(args.log_path, f'Hyper-parameters: {args.__dict__}')
    save_and_print(args.log_path, f'Evaluation model pool: {model_eval_pool}')

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    save_and_print(args.log_path, "BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    ''' initialize the synthetic data '''
    sample_per_cls = [args.ipc for _ in range(num_classes)]
    slice_size=1000
    gpu_list = list(range(torch.cuda.device_count()))
    pool = TensorPool(num_classes,slice_size,sample_per_cls,gpu_list,ldb=args.ldb,img_size=im_size,nthread=num_classes,max_iter=500*10000,channel=channel,lr=args.lr_img,layers_v=args.layers_v,arm=args.arm,dim=args.dim)
    save_and_print(args.log_path,f"slice_size:{slice_size}")
    #when zca no additional normalization
    pool_path_default=os.path.join(args.save_path,"pool_init.pt")
    if args.pool_path=="init":
            pool_path=pool_path_default
    else:
            pool_path=args.pool_path
    if not os.path.exists(pool_path):
                ref_dict = get_ref_dict(images_all,indices_class,args,mean,std,channel)
                pool.init_from_data(ref_dict)
                pool.save_slice_pool(pool_path_default)
                save_and_print(args.log_path, f"Pool  finished initialized,save to {pool_path_default}")
    else:
            pool.load_slice_pool(pool_path)
            save_and_print(args.log_path, f"Pool initialized from {pool_path}")
    pool.set_training_phase(0)
    pool.init_solvers()

    mean_ts = torch.Tensor(mean).view(1, channel, 1, 1).to(args.device)
    std_ts = torch.Tensor(std).view(1, channel, 1, 1).to(args.device)
    if args.batch_syn == 0:
        args.batch_syn = num_classes * args.ipc
    lr_it=float(args.lr_it)
    ldb_it=float(args.ldb_it)
    ''' training '''
    syn_lr = torch.tensor(args.lr_teacher, device=args.device)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)

    criterion = nn.CrossEntropyLoss().to(args.device)
    save_and_print(args.log_path, '%s training begins'%get_time())

    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    save_and_print(args.log_path, "Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        save_and_print(args.log_path, "loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)

    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}
    del images_all, labels_all

    for it in range(0, args.Iteration+1):
        save_this_it = False
        ''' Evaluate synthetic data '''
        if it in eval_it_pool and it >=0:
            for model_eval in model_eval_pool:
                save_and_print(args.log_path, '-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    save_and_print(args.log_path, f'DSA augmentation strategy: {args.dsa_strategy}')
                    save_and_print(args.log_path, f'DSA augmentation parameters: {args.dsa_param.__dict__}')
                else:
                    save_and_print(args.log_path, f'DC augmentation parameters: {args.dc_aug_param}')

                accs_test = []
                accs_train = []
                pool.test()
                image_syn_eval, label_syn_eval,rt = pool.get_data()
                image_syn_eval = (image_syn_eval-mean_ts.to(image_syn_eval.device))/std_ts.to(image_syn_eval.device)

                save_and_print(args.log_path, f"Evaluate dataset size: {image_syn_eval.shape} {label_syn_eval.shape}, rt: {rt}")
                run_num_eval=1
                if it%2000==0:
                    run_num_eval=args.num_eval
                for it_eval in range(run_num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                    args.lr_net = syn_lr.item()
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)
                del net_eval
                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)
                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    save_this_it = True
                save_and_print(args.log_path, 'Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
                save_and_print(args.log_path, f"{args.save_path}")
                save_and_print(args.log_path, f"{it:5d} | Accuracy/{model_eval}: {acc_test_mean}")
                save_and_print(args.log_path, f"{it:5d} | Max_Accuracy/{model_eval}: {best_acc[model_eval]}")
                save_and_print(args.log_path, f"{it:5d} | Std/{model_eval}: {acc_test_std}")
                save_and_print(args.log_path, f"{it:5d} | Max_Std/{model_eval}: {best_std[model_eval]}")
                del image_syn_eval, label_syn_eval

            save_and_print(args.log_path, f"{it:5d} | Synthetic_LR: {syn_lr.detach().cpu()}")

        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                pool.test()
                image_save, label_save,rt = pool.get_data()
                image_save = (image_save-mean_ts.to(image_save.device))/std_ts.to(image_save.device)
                save_dir = args.save_path

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                pool.save_slice_pool(os.path.join(save_dir, "pool_{}.pt".format(it)))
                if save_this_it:
                    pool.save_slice_pool(os.path.join(save_dir, "pool_best.pt"))
                if args.ipc < 600 or args.force_save:
                    upsampled = image_save
                    classes_save = np.random.permutation(num_classes)[:min(20, num_classes)]
                    indices_save = np.concatenate([c*args.ipc+np.arange(min(20, args.ipc)) for c in classes_save])
                    upsampled = upsampled[indices_save]
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=len(classes_save), normalize=True, scale_each=True)
                    plt.imshow(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)))
                    plt.savefig(f"{save_dir}/Synthetic_Images#{it}.png", dpi=300)
                    plt.close()

                    for clip_val in [2.5]:
                        std = torch.std(image_save)
                        mean = torch.mean(image_save)
                        upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                        upsampled = upsampled[indices_save]
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=len(classes_save), normalize=True, scale_each=True)
                        plt.imshow(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)))
                        plt.savefig(f"{save_dir}/Clipped_Synthetic_Images#{it}.png", dpi=300)
                        plt.close()

                    if args.zca:
                        image_save = image_save.to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save)
                        image_save.cpu()

                        torch.save(image_save.cpu(), os.path.join(save_dir, "images_zca_{}.pt".format(it)))

                        upsampled = image_save
                        upsampled = upsampled[indices_save]
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=len(classes_save), normalize=True, scale_each=True)
                        plt.imshow(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)))
                        plt.savefig(f"{save_dir}/Reconstructed_Images#{it}.png", dpi=300)
                        plt.close()

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            upsampled = upsampled[indices_save]
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=len(classes_save), normalize=True, scale_each=True)
                            plt.imshow(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)))
                            plt.savefig(f"{save_dir}/Clipped_Reconstructed_Images#{it}.png", dpi=300)
                            plt.close()

                    del image_save,  upsampled,grid

        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model

        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)
        student_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)

        start_epoch = np.random.randint(0, args.max_start_epoch)
        starting_params = expert_trajectory[start_epoch]

        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)
        pool.train()
        image_syn, label_syn,rt = pool.get_data()
        image_syn = (image_syn-mean_ts.to(image_syn.device))/std_ts.to(image_syn.device)
        syn_images = image_syn
        y_hat = label_syn.to(args.device)

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []

        for step in range(args.syn_steps):

            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))
            these_indices = indices_chunks.pop()

            x = syn_images[these_indices]
            this_y = y_hat[these_indices]
            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]

            x = student_net(x, flat_param=forward_params)
            ce_loss = criterion(x, this_y)

            grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]

            student_params.append(student_params[-1] - syn_lr * grad)

        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)

        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss

        optimizer_lr.zero_grad()

        grand_loss.backward()
        pool.fill_data_diff()
        pool.backward(lr_it,ldb_it)
        optimizer_lr.step()
   
        syn_lr.data = syn_lr.data.clip(min=0.001) # To avoid invalid syn_lr (refer to HaBa)
        for _ in student_params:
            del _

        if it % 10 == 0:
            save_and_print(args.log_path, '%s iter = %04d, loss = %.4f,grad = %.4f' % (get_time(), it, grand_loss.item(),torch.sum(torch.abs(pool.data.grad))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='ImageNet', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')
    parser.add_argument('--model', type=str, default='ConvNetD5', help='model')
    parser.add_argument('--ipc', type=int, default=102, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode, check utils.py for more info')
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    parser.add_argument('--eval_it', type=int, default=500, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=15000, help='how many distillation steps to perform')
    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'], help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='../data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='../buffers', help='buffer path')
    parser.add_argument('--zca', type=str,default='False',choices=['True','False'],help="do ZCA whitening")
    parser.add_argument('--load_all', action='store_true',default=False, help="only use if you can fit all expert trajectories into RAM")
    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')
    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')
    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')
    parser.add_argument('--res', type=int, default=128, help='resolution for imagenet')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sh_file', type=str)
    parser.add_argument('--FLAG', type=str, default="TEST")
    parser.add_argument('--save_path', type=str, default="./results")
    parser.add_argument('--pool_path', type=str, default='init', help='path to save results')

    parser.add_argument('--syn_steps', type=int)
    parser.add_argument('--expert_epochs', type=int)
    parser.add_argument('--max_start_epoch', type=int)
    parser.add_argument('--lr_lr', type=float)
    parser.add_argument('--lr_teacher', type=float)

    ### FreD ###
    parser.add_argument('--batch_syn', type=int)
    #parser.add_argument('--msz_per_channel', type=int)
    #parser.add_argument('--lr_freq', type=float)
    #parser.add_argument('--mom_freq', type=float)
    ### TM_POOL ###
    parser.add_argument('--ldb', type=float,default=0.1)  # joint-opt rate coeff (TM); also sets warmup beta=1/ldb
    parser.add_argument('--lr_img', type=float,default=0.001)
    parser.add_argument('--lr_it',type=float,default=1000,help="distillation-grad amplifier; numerator of lambda=lr_it/(ldb_it*ldb)")
    parser.add_argument('--ldb_it',type=float,default=10,help="rate-grad amplifier; TM stage1=10, stage2=150")
    parser.add_argument("--arm",type=int,default=32)
    parser.add_argument("--dim",type=int,default=4)
    parser.add_argument("--layers_v",type=str,default='v5')

    args = parser.parse_args()
    set_seed(args.seed)
    #way to process the args
    args = load_default(args)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    args.save_path = args.save_path + f"/{args.FLAG}"
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)


    shutil.copy(f"./scripts/{args.sh_file}", f"{args.save_path}/{args.sh_file}")
    args.log_path = f"{args.save_path}/log.txt"
    save_and_print(args.log_path, f"begin at time: {get_time()}")
    main(args)