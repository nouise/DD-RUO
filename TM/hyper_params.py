SYN_STEPS = {"MNIST": {1: 50, 10: 30}, "FashionMNIST": {1: 50, 10: 60}, "SVHN": {1: 50, 10: 30, 50: 40},
             "CIFAR10": {1: 50, 2: 50, 10: 40, 11: 40, 50: 30, 51: 30},
             "CIFAR100": {1: 50, 10: 20, 50: 80}, "Tiny": {1: 30, 10: 40, 50: 40}, "ImageNet": {1: 20, 2: 20, 10: 20}}

EXPERT_EPOCHS = {"MNIST": {1: 2, 10: 2}, "FashionMNIST": {1: 2, 10: 2}, "SVHN": {1: 2, 10: 2, 50: 2},
                 "CIFAR10": {1: 2, 2: 2, 10: 2, 11: 2, 50: 2, 51: 2},
                 "CIFAR100": {1: 2, 10: 2, 50: 2}, "Tiny": {1: 2, 10: 2, 50: 2}, "ImageNet": {1: 2, 2: 2, 10: 2}}

MAX_START_EPOCH = {"MNIST": {1: 5, 10: 15}, "FashionMNIST": {1: 5, 10: 15}, "SVHN": {1: 5, 10: 15, 50: 40},
                   "CIFAR10": {1: 5, 2: 5, 10: 15, 11: 15, 50: 40, 51: 40},
                   "CIFAR100": {1: 15, 10: 40, 50: 40}, "Tiny": {1: 30, 10: 40, 50: 40}, "ImageNet": {1: 10, 2: 10, 10: 10}}

LR_LR = {"MNIST": {1: 1e-7, 10: 1e-5}, "FashionMNIST": {1: 1e-7, 10: 1e-5}, "SVHN": {1: 1e-7, 10: 1e-5, 50: 1e-5},
         "CIFAR10": {1: 1e-7, 2: 1e-7, 10: 1e-5, 11: 1e-5, 50: 1e-5, 51: 1e-5},
         "CIFAR100": {1: 1e-5, 10: 1e-5, 50: 1e-5}, "Tiny": {1: 1e-4, 10: 1e-4, 50: 1e-4}, "ImageNet": {1: 1e-6, 2: 1e-6, 10: 1e-6}}

LR_TEACHER = {"MNIST": {1: 1e-2, 10: 1e-2}, "FashionMNIST": {1: 1e-2, 10: 1e-2}, "SVHN": {1: 1e-2, 10: 1e-2, 50: 1e-3},
              "CIFAR10": {1: 1e-2, 2: 1e-2, 10: 1e-2, 11: 1e-2, 50: 1e-3, 51: 1e-3},
              "CIFAR100": {1: 1e-2, 10: 1e-2, 50: 1e-2}, "Tiny": {1: 1e-2, 10: 1e-2, 50: 1e-2}, "ImageNet": {1: 1e-2, 2: 1e-2, 10: 1e-2}}

config_default = {
     "CIFAR100": {
        64: {
            "syn_steps": 60,
            "expert_epochs": 2,
            "max_start_epoch": 40,
            "lr_lr": 1e-5,
            "lr_teacher": 1e-2,
        },
        48: {
            "syn_steps": 60,
            "expert_epochs": 2,
            "max_start_epoch": 40,
            "lr_lr": 1e-5,
            "lr_teacher": 1e-2,
        },
        240:{
            "syn_steps": 60,
            "expert_epochs": 2,
            "max_start_epoch": 40,
            "lr_lr": 1e-5,
            "lr_teacher": 1e-2,
        },
        120:{
            "syn_steps": 60,
            "expert_epochs": 2,
            "max_start_epoch": 10,
            "lr_lr": 1e-5,
            "lr_teacher": 1e-2,
        },
        360:{
            "syn_steps": 60,
            "expert_epochs": 2,
            "max_start_epoch": 40,
            "lr_lr": 1e-5,
            "lr_teacher": 1e-2,
        }
    },
    "CIFAR10": {
        64: {
            "syn_steps": 60,
            "expert_epochs": 2,
            "max_start_epoch": 40,
            "lr_lr": 1e-5,
            "lr_teacher": 1e-2,
        },
        240:{
            "syn_steps": 60,
            "expert_epochs": 2,
            "max_start_epoch": 40,
            "lr_lr": 1e-5,
            "lr_teacher": 1e-2,
        },
        120:{
            "syn_steps": 60,
            "expert_epochs": 2,
            "max_start_epoch": 10,
            "lr_lr": 1e-5,
            "lr_teacher": 1e-2,
        },
        360:{
            "syn_steps": 60,
            "expert_epochs": 2,
            "max_start_epoch": 10,
            "lr_lr": 1e-5,
            "lr_teacher": 1e-2,
        },
        718:{
            "syn_steps": 60,
            "expert_epochs": 2,
            "max_start_epoch": 10,
            "lr_lr": 1e-5,
            "lr_teacher": 1e-2,
        }
    }
}

def load_default(args):
    if args.zca =="True":
        print("Default FreD does not use ZCA,but you set it to True")
    if args.dataset=="ImageNet":
        args.syn_steps=40
        args.expert_epochs=2
        args.max_start_epoch=20
        args.lr_lr=1e-5
        args.lr_teacher=1e-2
    elif args.dataset=="CIFAR10" or args.dataset=="CIFAR100":
        parameters_dict=config_default[args.dataset][args.ipc]
        args.syn_steps = parameters_dict['syn_steps']
        args.expert_epochs = parameters_dict['expert_epochs']
        args.max_start_epoch = parameters_dict['max_start_epoch']
        args.lr_lr = parameters_dict['lr_lr']
        args.lr_teacher = parameters_dict['lr_teacher']
    else :
        dataset = args.dataset
        if args.ipc in [1, 2, 10, 11, 50, 51,64]:
            ipc = args.ipc
        else:
            exit("Undefined IPC")

        if args.syn_steps == None:
            args.syn_steps = SYN_STEPS[dataset][ipc]

        if args.expert_epochs == None:
            args.expert_epochs = EXPERT_EPOCHS[dataset][ipc]

        if args.max_start_epoch == None:
            args.max_start_epoch = MAX_START_EPOCH[dataset][ipc]

        if args.lr_lr == None:
            args.lr_lr = LR_LR[dataset][ipc]

        if args.lr_teacher == None:
            args.lr_teacher = LR_TEACHER[dataset][ipc]
    return args