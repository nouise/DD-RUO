import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from core.ts.tensor_pool import TensorPool
from torchvision.utils import save_image
from core.utils import get_loops,match_loss,get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug, set_seed, save_and_print, TensorDataset, get_images, epoch
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
def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in ['ConvNetBN']:  # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param
def main():
    begin_time = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='ImageNet', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNetD5', help='model')
    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')
    parser.add_argument('--res', type=int, default=128, help='resolution for imagenet')
    parser.add_argument('--zca', type=str,default='False',choices=['True','False'],help="do ZCA whitening")
    parser.add_argument('--ipc', type=int, default=96, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=500, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.001, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='../data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='./results', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--pool_path', type=str, default='init', help='path to save pt')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sh_file', type=str)
    parser.add_argument('--FLAG', type=str, default="TEST")

    parser.add_argument('--batch_syn', type=int)

    parser.add_argument('--ldb', type=float,default=5)  # joint-opt rate coeff (DC/GM); warmup beta uses a separate ldb=1e-6 run
    parser.add_argument('--lr_it',type=float,default=100,help="distillation-grad amplifier; numerator of lambda=lr_it/(ldb_it*ldb)")
    parser.add_argument('--ldb_it',type=float,default=1500,help="rate-grad amplifier; DC stage1=700, stage2=1500")
    parser.add_argument("--arm",type=int,default=32)
    parser.add_argument("--dim",type=int,default=4)
    parser.add_argument("--layers_v",type=str,default='v6')
    args = parser.parse_args()
    set_seed(args.seed)
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {args.device},{torch.cuda.device_count()}")
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    args.save_path = args.save_path + f"/{args.FLAG}"
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    
    shutil.copy(f"./scripts/{args.sh_file}", f"{args.save_path}/{args.sh_file}")
    print (args)
    args.log_path = f"{args.save_path}/log.txt"
    if args.zca=="True":
        args.zca=True
    else:
        args.zca=False
    eval_it_pool = np.arange(0, args.Iteration+1, 10).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    args.channel, args.im_size, args.num_classes, args.mean, args.std = channel, im_size, num_classes, mean, std

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    
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
        slice_size=200
        pool = TensorPool(num_classes,slice_size,sample_per_cls,[0],ldb=args.ldb,img_size=im_size,nthread=num_classes,max_iter=500*10000,channel=channel,lr=args.lr_img,layers_v=args.layers_v,arm=args.arm,dim=args.dim)
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
        criterion = nn.CrossEntropyLoss().to(args.device)
        save_and_print(args.log_path, '%s training begins'%get_time())
        lr_it=float(args.lr_it)
        ldb_it=float(args.ldb_it)
        # the K = iteration .outer loop for the random initialization
        for it in range(args.Iteration+1):
            mse_item=0.0000005
            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                final_bpp,distribution=pool.quantize_net(mse_threshold=mse_item)
                formatted_output = ", ".join(
                        f"{key}={value:.6f}, ratio={value /final_bpp* 100:.6f}%" for key, value in distribution.items()
                    )
                save_and_print(args.log_path,f"right now,the mse is {mse_item},final_bpp:{final_bpp},bits/class={final_bpp*args.ipc*im_size[0]*im_size[1]}.\n details:{formatted_output}")
                pool.train()
                pool.test()
                image_syn_eval, label_syn_eval,rt = pool.get_data()
                image_syn_eval = (image_syn_eval-mean_ts.to(image_syn_eval.device))/std_ts.to(image_syn_eval.device)
                image_syn_eval.to(args.device)
                for model_eval in model_eval_pool:
                    save_and_print(args.log_path, '-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    if args.dsa:
                        args.epoch_eval_train = 1000
                        args.dc_aug_param = None
                        save_and_print(args.log_path, f'DSA augmentation strategy: {args.dsa_strategy}')
                        save_and_print(args.log_path, f'DSA augmentation parameters: {args.dsa_param.__dict__}')
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                        save_and_print(args.log_path, f'DC augmentation parameters: {args.dc_aug_param}')

                    if args.dsa or args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                    else:
                        args.epoch_eval_train = 300

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                    save_and_print(args.log_path, 'Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                if True:
                    save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%s_%dipc_exp%d_iter%d.png'%(begin_time,args.method, args.dataset, args.model, args.ipc, exp, it))
                    image_syn_vis = image_syn_eval
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
                pool.save_slice_pool(f'{args.save_path}/{begin_time}_{args.dataset}_{args.model}_{args.ipc}_{it}_dc_pool.pt')

                del image_syn_vis,image_syn_eval,label_syn_eval
            ''' Train synthetic data '''
            
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.

            pool.train()
            image_syn,label_syn,rt = pool.get_data()
            # the ol should be the t,0~T-1
            for ol in range(args.outer_loop):

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.
                pool.train()
                image_syn_eval, label_syn,rt = pool.get_data()
                image_syn = (image_syn_eval-mean_ts.to(image_syn_eval.device))/std_ts.to(image_syn_eval.device)
                image_syn.to(args.device)

                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(images_all, indices_class, c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train() # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real) # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer


                ''' update synthetic data '''
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(images_all, indices_class, c, args.batch_real).to(args.device)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))
                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                    
                    loss += match_loss(gw_syn, gw_real, args)

                loss.backward()
                pool.fill_data_diff()
                pool.backward(lr_it,ldb_it)
                loss_avg += loss.item()
                if ol%10==0:
                    save_and_print(args.log_path,f"exp:{exp} it:{it} ol:{ol}=> result: loss_syn:{loss_syn.item()},loss_real:{loss_real.item()},loss_match:{loss.item()},rt:{rt}")
                if ol == args.outer_loop - 1:
                    break


                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug = True if args.dsa else False)


            loss_avg /= (num_classes*args.outer_loop)

            if it%10 == 0:
                save_and_print(args.log_path, '%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == args.Iteration: # only record the final results
                pool.save_slice_pool(f'{args.save_path}/{begin_time}_{args.dataset}_{args.model}_{args.ipc}_dc_pool.pt')
    
    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        save_and_print(args.log_path, 'Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))
    time_elapsed = time.time() - time.mktime(time.strptime(begin_time, "%Y%m%d_%H%M%S"))
    save_and_print(args.log_path,f"time comsume: {time.strftime('%H:%M:%S', time.gmtime(time_elapsed))}")

if __name__ == '__main__':
    main()


