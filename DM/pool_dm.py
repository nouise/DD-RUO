import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import copy
import argparse
import numpy as np
import torch
from core.ts.tensor_pool import TensorPool
from torchvision.utils import save_image
from core.utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug, set_seed, save_and_print, TensorDataset, get_images, epoch
import shutil
import torchvision
import matplotlib.pyplot as plt
def get_ref_dict(images_all,indices_class,args,mean,std,channel):
    mean_ts = torch.Tensor(mean).view(1, channel, 1, 1)
    std_ts = torch.Tensor(std).view(1, channel, 1, 1)
    if not args.zca:    
        ref_dict = {c: get_images(images_all, indices_class, c, args.ipc).detach() * std_ts + mean_ts for c in range(args.num_classes)}
    else:
        ref_dict = {c:get_images(images_all, indices_class, c, args.ipc).detach().data for c in range(args.num_classes)}
    return ref_dict

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DM', help='DC/DSA/DM')
    parser.add_argument('--dataset', type=str, default='ImageNet', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')
    parser.add_argument('--res', type=int, default=128, help='resolution for imagenet')
    parser.add_argument('--zca', type=str,default='False',choices=['True','False'],help="do ZCA whitening")
    parser.add_argument('--model', type=str, default='ConvNetD5', help='model')
    parser.add_argument('--ipc', type=int, default=96, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='../data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='./results', help='path to save results')
    parser.add_argument('--pool_path', type=str, default='init', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')


    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sh_file', type=str)
    parser.add_argument('--FLAG', type=str, default="TEST")

    parser.add_argument('--batch_syn', type=int)
    parser.add_argument('--ldb', type=float,default=5)  # joint-opt rate coeff (DM); warmup beta uses a separate ldb=1e-6 run
    parser.add_argument('--lr_img', type=float,default=0.001)
    parser.add_argument('--lr_it',type=float,default=1000,help="distillation-grad amplifier; numerator of lambda=lr_it/(ldb_it*ldb)")
    parser.add_argument('--ldb_it',type=float,default=300,help="rate-grad amplifier; DM stage1=10/50, stage2=300")
    parser.add_argument("--arm",type=int,default=32)
    parser.add_argument("--dim",type=int,default=4)
    parser.add_argument("--layers_v",type=str,default='v6')
    args = parser.parse_args()
    set_seed(args.seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    args.save_path = args.save_path + f"/{args.FLAG}"
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    shutil.copy(f"./scripts/{args.sh_file}", f"{args.save_path}/{args.sh_file}")
    args.log_path = f"{args.save_path}/log.txt"
    save_and_print(args.log_path,f"zca:{args.zca}")
    if args.zca=="True":
        args.zca=True
    else:
        args.zca=False
    eval_it_pool = np.arange(0, args.Iteration+1, 1000).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration]
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)

    args.channel, args.im_size, args.num_classes, args.mean, args.std = channel, im_size, num_classes, mean, std
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    save_and_print(args.log_path,f"channel:{channel},im_size:{im_size},num_classes:{num_classes}")
    save_and_print(args.log_path,f"class_map:{class_map},vs:class_map_inv:{class_map_inv}")
    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = dict()


    for exp in range(args.num_exp):
        save_and_print(args.log_path, f'\n================== Exp {exp} ==================\n ')
        save_and_print(args.log_path, f'Hyper-parameters: {args.__dict__}')

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]
        save_and_print(args.log_path, "BUILDING DATASET")
        for i in range(len(dst_train)):
            sample = dst_train[i]
            images_all.append(torch.unsqueeze(sample[0], dim=0))
            labels_all.append(class_map[torch.tensor(sample[1]).item()])

        for i, lab in enumerate(labels_all):
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
        ''' training '''
        save_and_print(args.log_path, '%s training begins'%get_time())
        lr_it=float(args.lr_it)
        ldb_it=float(args.ldb_it)
        for it in range(args.Iteration+1):
            ''' Evaluate synthetic data '''
            mse_item=0.0000005
            if it in eval_it_pool and it >=0:
                pool.test()
                image_syn_eval, label_syn_eval,rt = pool.get_data()
                image_syn_eval = (image_syn_eval-mean_ts.to(image_syn_eval.device))/std_ts.to(image_syn_eval.device)
                save_and_print(args.log_path, f"Evaluate dataset size: {args.ipc},{image_syn_eval.shape} {label_syn_eval.shape}, rt: {rt}")
                image_syn_eval.to(args.device)
                for model_eval in model_eval_pool:
                    save_and_print(args.log_path, '-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    save_and_print(args.log_path, f'DSA augmentation strategy: {args.dsa_strategy}')
                    save_and_print(args.log_path, f'DSA augmentation parameters: {args.dsa_param.__dict__}')
                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                    save_and_print(args.log_path, 'Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                    pool.save_slice_pool(f"{args.save_path}/{args.method}_{args.dataset}_{args.model}_{args.ipc}ipc_exp{exp}_{it}.pt")
                    data_save[it]={
                        "mean":np.mean(accs), 
                        "std":np.std(accs),
                        "grids_bpp":rt
                    }
                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs
                ''' visualize and save '''
                save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis=image_syn_eval
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                max_imgs = min(20,args.ipc)  # 每个类别展示 20 张
                # 每个类别选前 20 张图片，然后拼接起来
                selected_images = torch.cat([
                    image_syn_vis[i * args.ipc : i * args.ipc + max_imgs] 
                    for i in range(num_classes)
                ], dim=0)
                grid = torchvision.utils.make_grid(selected_images, nrow=max_imgs, normalize=True, scale_each=True)
                plt.imshow(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)))
                plt.savefig(save_name, dpi=300)
                plt.close()
                del image_syn_vis,image_syn_eval,label_syn_eval
            ''' Train synthetic data '''
            pool.train()
            image_syn_eval, label_syn,rt = pool.get_data()
            image_syn = (image_syn_eval-mean_ts.to(image_syn_eval.device))/std_ts.to(image_syn_eval.device)
            image_syn.to(args.device)
            net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False
            embed = net.module.features if torch.cuda.device_count() > 1 else net.features # for GPU parallel (feature extractor for DM distribution matching)
            loss_avg = 0
            if args.dataset == "TinyImageNet":
                ''' update synthetic data '''
                print("tiny ImageNet")
            else:
                ''' update synthetic data '''
                if 'BN' not in args.model: # for ConvNet
                    loss = torch.tensor(0.0).to(args.device)
                    for c in range(num_classes):
                        img_real = get_images(images_all, indices_class, c, args.batch_real).to(args.device)
                        img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                        if args.dsa:
                            seed = int(time.time() * 1000) % 100000
                            img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                            img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        output_real = embed(img_real).detach()
                        output_syn = embed(img_syn)

                        loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

                else: # for ConvNetBN
                    images_real_all = []
                    images_syn_all = []
                    loss = torch.tensor(0.0).to(args.device)
                    for c in range(num_classes):
                        img_real = get_images(c, args.batch_real)
                        img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                        if args.dsa:
                            seed = int(time.time() * 1000) % 100000
                            img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                            img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                        images_real_all.append(img_real)
                        images_syn_all.append(img_syn)

                    images_real_all = torch.cat(images_real_all, dim=0)
                    images_syn_all = torch.cat(images_syn_all, dim=0)

                    output_real = embed(images_real_all).detach()
                    output_syn = embed(images_syn_all)

                    loss += torch.sum((torch.mean(output_real.reshape(num_classes, args.batch_real, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, args.ipc, -1), dim=1))**2)

                loss.backward()
                pool.fill_data_diff()
                pool.backward(lr_it,ldb_it)
                loss_avg += loss.item()
                loss_avg /= (num_classes)
                if it%200 == 0:
                    save_and_print(args.log_path, '%s iter = %04d, loss = %.4f,grids = %.4f' % (get_time(), it, loss_avg,rt))
                if it == args.Iteration: # only record the final results
                   pool.save_slice_pool(f"{args.save_path}/{args.method}_{args.dataset}_{args.model}_{args.ipc}ipc_exp{exp}_{it}.pt")
                del image_syn

    save_and_print(args.log_path, '\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        save_and_print(args.log_path, 'Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))
    for key,value in data_save.items():
        save_and_print(args.log_path, 
                   f"{key} iteration: mean: {value['mean']}, std: {value['std']}, grids: {value['grids_bpp']}!")


if __name__ == '__main__':
    main()


