import torch
from utils.utils import train_DNN, train_BNN, get_model, get_dataset, test_DNN
from distill import distill, set_martern_prior, Multivariate_MOPED
import argparse
from torch.utils.tensorboard import SummaryWriter
import datetime
from termcolor import colored
import torch.nn.utils.prune as prune
import torch.nn as nn
import os 
import logging

# def prune_model(model, sparsity):
#     """
#     Prunes the model to achieve the desired sparsity level.
#     Args:
#         model (torch.nn.Module): The model to prune.
#         sparsity (float): Desired sparsity level (0.0 to 1.0).
#     """
#     for name, module in model.named_modules():
#         # Check if the module is a pruning target (Conv2D or Linear layers)
#         # if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
#         if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
#             # Apply or update pruning
#             prune.l1_unstructured(module, name='weight', amount=sparsity)

#             # Calculate the percentage of weights pruned
#             pruned_percentage = 1 - float(module.weight_mask.sum()) / module.weight.numel()
#             # print(colored(f"Pruned {name}: {pruned_percentage:.2%} of weights set to 0", 'yellow'))
#         else:
#             # print(colored(f"Skipping {name}: Not a prunable layer", 'cyan'))
#             pass
        
#     # Calculate the total sparsity in the model
#     total_params = sum(module.weight.numel() for module in model.modules() if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)))
#     remaining_params = sum(module.weight_mask.sum() for module in model.modules() if hasattr(module, 'weight_mask'))
#     total_sparsity = 1 - remaining_params / total_params
#     print(colored(f"Total sparsity: {total_sparsity:.2%}", 'yellow'))


def prune_model(model, sparsity, logger):
    """
    모델 전체의 Conv2d 및 Linear 레이어에 대해 global unstructured pruning을 적용합니다.
    Args:
        model (torch.nn.Module): 가지치기할 모델.
        sparsity (float): 가지치기 비율 (0.0 ~ 1.0). 전체 weight 중 프루닝할 비율.
    """
    # 프루닝할 (module, parameter) 쌍을 모읍니다.
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    
    # global unstructured pruning을 적용합니다.
    prune.global_unstructured(
        parameters_to_prune,
        # pruning_method=prune.L1Unstructured,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )
    
    # 각 모듈별 가지치기 결과 출력
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total_params = module.weight.numel()
            # 프루닝 후 각 모듈에는 'weight_mask' 버퍼가 생성됩니다.
            remaining_params = module.weight_mask.sum().item() if hasattr(module, 'weight_mask') else total_params
            pruned_percentage = 1 - (remaining_params / total_params)
        else:
            pass
            
    # Calculate Total Sparsity in the model
    total_params = sum(module.weight.numel() for module in model.modules() if isinstance(module, (nn.Conv2d, nn.Linear)))
    remaining_params = sum(module.weight_mask.sum().item() for module in model.modules() if hasattr(module, 'weight_mask'))
    total_sparsity = 1 - (remaining_params / total_params)
    logger.info(colored(f"Total sparsity: {total_sparsity:.2%}", 'yellow'))
    
def main(args):
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model = get_model(args = args, logger = logger)

    logging.info(f"The number of parameters in the model: {sum(p.numel() for p in model.parameters()):,}")
    # Optimizer
    if args.optimizer == 'sgd':
        
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov = args.nesterov)
    
    elif args.optimizer == 'adam':
    
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        args.momentum = None
        args.nesterov = None
        args.weight_decay = None
        
    # if args.model == 'vit-tiny':
    #     optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     args.momentum = None
    #     args.nesterov = None
    #     logging.info(colored(f"Optimizer is set to AdamW for ViT", 'green'))
    
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")        


        
    logging.info(colored(f"Optimizer: {args.optimizer}, Learning rate: {args.lr}, Weight decay: {args.weight_decay}, Momentum: {args.momentum}", 'green'))
    
    if args.data == 'cifar100':
        # Multi Step Learning rate Schedule
        args.scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100, 200], gamma=0.1)
    else:
        args.scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100000], gamma=0.1) # We don't want to change the learning rate schedule for now.
    log_params = {
        'data': args.data,
        'model': args.model,
        'date': date.split('-')[0],
        'type': args.type,
        'bs': args.bs,
        'opt': args.optimizer,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'nesterov': args.nesterov,
        'lr': args.lr,
        'mc_runs': args.mc_runs,
        'epochs': args.epochs,
        'moped': args.moped,
        'timestamp': date
    }

    params_str = "_".join([f"{key}_{value}" for key, value in log_params.items() if key not in ['data', 'model', 'date', 'type']])
    
    log_path = f"runs/{log_params['data']}/{log_params['model']}/{log_params['date']}/{log_params['type']}/{params_str}"
    
    writer = SummaryWriter(log_path)
    
    file_handler = logging.FileHandler(log_path + '/log.txt')
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    train_loader, test_loader = get_dataset(args = args, logger = logger)
   
    # Save the training arguments
    with open(f"{log_path}/config.txt", "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    
    if args.type == 'dnn':

        if args.prune:
            model.load_state_dict(torch.load(args.weight))
            args.best_acc, args.best_nll = test_DNN(model, test_loader)
            logger.info(colored(f"Test accuracy of DNN: {args.best_acc:.4f}, Test NLL: {args.best_nll:.4f}", 'green'))
            
            save_path = os.path.join(writer.log_dir, f'original_model.pth')
            torch.save(model.state_dict(), save_path)
            logger.info(colored(f"Original model is saved at {save_path}", 'green'))
            
            args.total_epoch = 0
            for i in range(1, 100):

                args.prune_iter = i

                # Pruning step
                prune_model(model, sparsity=i/100.0, logger=logger)
                args.scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100, 200], gamma=0.1)
                # Training
                if train_DNN(epoch=args.epochs, 
                        model=model, 
                        train_loader=train_loader, 
                        test_loader=test_loader, 
                        optimizer=optim, 
                        writer=writer,
                        device=device,
                        args=args,
                        logger=logger): break
                
        else:
            train_DNN(epoch=args.epochs, 
                    model=model, 
                    train_loader=train_loader, 
                    test_loader=test_loader, 
                    optimizer=optim, 
                    writer=writer,
                    device=device,
                    args=args,
                    logger=logger)

    else:
        train_BNN(epoch=args.epochs, 
                  model=model, 
                  train_loader=train_loader, 
                  test_loader=test_loader, 
                  optimizer=optim, 
                  mc_runs=args.mc_runs, 
                  bs=args.bs, 
                  writer=writer,
                  device=device,
                  args=args,
                  logger=logger)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a Bayesian Neural Network')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--mc_runs', type=int, default=30, help='Number of Monte Carlo runs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--model', type=str, help='Model to train [resnet18, resnet20, densenet30, densenet121, mobilenetv2]')
    parser.add_argument('--type', type=str, default='dnn', help='Type of model [dnn, uni, multi]')
    parser.add_argument('--multi-gpu', action='store_true', help='Use multi-GPU')
    parser.add_argument('--t', type=float, default=1.0, help='Cold Posterior temperature')
    parser.add_argument('--data', type=str, help='Dataset to use [cifar10, cifar100, svhn, tinyimagenet]')
    parser.add_argument('--train_sampler', type=bool, default=False, help='Do not use this argument')
    parser.add_argument('--weight', type=str, help='DNN weight path for ')
    parser.add_argument('--moped', action='store_true', help='Use MOPED')
    parser.add_argument('--alpha', type=float, default= 0.0, help = 'Distill Coefficient')
    parser.add_argument('--martern', action='store_true', help='Use Martern Prior')
    parser.add_argument('--multi_moped', action='store_true', help='Use Multi-MOPED')
    parser.add_argument('--prune', action='store_true', help='Use pruning')
    parser.add_argument('--optimizer', type=str, help='Optimizer to use [sgd, adam]')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--nesterov', action='store_true', help='Use Nesterov')
    args = parser.parse_args()
    
    print(colored(args, 'blue'))
    
    
    
    main(args)
