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


def prune_model(model, sparsity):
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
            # print(colored(f"Pruned {name}: {pruned_percentage:.2%} of weights set to 0", 'yellow'))
        else:
            # print(colored(f"Skipping {name}: Not a prunable layer", 'cyan'))
            pass
            
    # Calculate Total Sparsity in the model
    total_params = sum(module.weight.numel() for module in model.modules() if isinstance(module, (nn.Conv2d, nn.Linear)))
    remaining_params = sum(module.weight_mask.sum().item() for module in model.modules() if hasattr(module, 'weight_mask'))
    total_sparsity = 1 - (remaining_params / total_params)
    print(colored(f"Total sparsity: {total_sparsity:.2%}", 'yellow'))
    
def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model(args)
    
    train_loader, test_loader = get_dataset(args)
        
    # 현재 날짜와 시간을 포맷팅
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # log_path를 위한 파라미터들을 dict로 구성
    log_params = {
        'data': args.data,
        'model': args.model,
        'date': date.split('-')[0],
        'type': args.type,
        'bs': args.bs,
        'lr': args.lr,
        'mc_runs': args.mc_runs,
        'temp': args.t,
        'epochs': args.epochs,
        'kd': args.distill,
        'martern': args.martern,
        'alpha': args.alpha,
        'moped': args.moped,
        'multi_moped': args.multi_moped,
        'timestamp': date
    }

    # log_params의 항목들을 key=value 형식으로 자동으로 조합하여 log_path 구성
    params_str = "_".join([f"{key}_{value}" for key, value in log_params.items() if key not in ['data', 'model', 'date', 'type']])
    
    log_path = f"runs/{log_params['data']}/{log_params['model']}/{log_params['date']}/{log_params['type']}/{params_str}"
    
    writer = SummaryWriter(log_path)
    
    # Save the arguments
    with open(f"{log_path}/config.txt", "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
                                
    # assert args.distill or args.martern or args.moped or args.multi_moped, "Please specify the unique method"
    if args.distill:
        print("Distillation is used")
        dnn_model = get_model(args, distill=True)
        dnn_model.load_state_dict(torch.load(args.weight))
        print(colored(f"Distilling from {args.weight}", 'green'))
        print(colored(f"Test accuracy of DNN: {test_DNN(dnn_model, test_loader)}", 'green'))
        model = distill(dnn_model, model, data_loader = train_loader, args = args, alpha= args.alpha, device = device, writer = writer)
        args.type = 'multi'
        
    elif args.martern:
        print("Martern Prior is used")
        dnn_model = get_model(args, distill=True)
        dnn_model.load_state_dict(torch.load(args.weight))
        print(colored(f"Weight is loaded from {args.weight}", 'green'))
        print(colored(f"Test accuracy of DNN: {test_DNN(dnn_model, test_loader)}", 'green'))
        model = set_martern_prior(dnn_model, model, device = device)
        
    elif args.multi_moped:
        print("Multi-MOPED is used")
        dnn_model = get_model(args, distill=True)
        dnn_model.load_state_dict(torch.load(args.weight))
        print(colored(f"Pretrained weight is loaded from {args.weight}", 'green'))
        print(colored(f"Test accuracy of DNN: {test_DNN(dnn_model, test_loader)}", 'green'))
        model = Multivariate_MOPED(dnn = dnn_model, bnn = model, device = device)
   
    # Optimizer
    # optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Learning rate scheduler
    # args.scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[args.epochs], gamma=1.0) # Never decay the learning rate
    
    if args.type == 'dnn':

        if args.prune:
            model.load_state_dict(torch.load(args.weight))
            args.best_acc, args.best_nll = test_DNN(model, test_loader)
            print(colored(f"Test accuracy of DNN: {args.best_acc:.4f}, Test NLL: {args.best_nll:.4f}", 'green'))
            
            save_path = os.path.join(writer.log_dir, f'original_model.pth')
            torch.save(model.state_dict(), save_path)
            print(colored(f"Original model is saved at {save_path}", 'green'))
            
            args.total_epoch = 0
            for i in range(1, 100):

                args.prune_iter = i

                # Pruning step
                prune_model(model, sparsity=i/100.0)
                
                # Training
                if train_DNN(epoch=args.epochs, 
                        model=model, 
                        train_loader=train_loader, 
                        test_loader=test_loader, 
                        optimizer=optim, 
                        writer=writer,
                        device=device,
                        args=args): break
                
                
        else:
            train_DNN(epoch=args.epochs, 
                    model=model, 
                    train_loader=train_loader, 
                    test_loader=test_loader, 
                    optimizer=optim, 
                    writer=writer,
                    device=device,
                    args=args)

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
                  args=args)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a Bayesian Neural Network')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--mc_runs', type=int, default=30, help='Number of Monte Carlo runs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--model', type=str, default='resnet20', help='Model to train [simple, lenet, vgg7, resnet20]')
    parser.add_argument('--type', type=str, default='dnn', help='Type of model [dnn, uni, multi]')
    parser.add_argument('--multi-gpu', action='store_true', help='Use multi-GPU')
    parser.add_argument('--t', type=float, default=1.0, help='Cold Posterior temperature')
    parser.add_argument('--data', type=str, default='cifar', help='Dataset to use [mnist, cifar]')
    parser.add_argument('--train_sampler', type=bool, default=False, help='Do not use this argument')
    parser.add_argument('--distill', action='store_true', help='Use distillation')
    parser.add_argument('--weight', type=str, help='DNN weight path for distillation')
    parser.add_argument('--moped', action='store_true', help='Use MOPED')
    parser.add_argument('--alpha', type=float, default= 0.0, help = 'Distill Coefficient')
    parser.add_argument('--martern', action='store_true', help='Use Martern Prior')
    parser.add_argument('--multi_moped', action='store_true', help='Use Multi-MOPED')
    parser.add_argument('--prune', action='store_true', help='Use pruning')
    args = parser.parse_args()
    
    print(colored(args, 'blue'))
    main(args)
