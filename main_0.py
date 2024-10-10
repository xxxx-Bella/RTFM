from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import sys
from utils import save_best_record, visulization, print_training_info, StdRedirect
from model import Model
from dataset import Dataset
from train_0 import train
from test_10crop_0 import test
import option
from tqdm import tqdm
# from utils import Visualizer
from config import *
import wandb
from datetime import datetime


# # 实时可视化训练过程, env 指定 Visdom 的环境名称
# viz = Visualizer(env='shanghai tech 10 crop', use_incoming_socket=False)


if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)
    log_dir = f'./log/run-{args.run_name}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, f'{args.run_name}.txt')
    
    ######################### Print setting #########################
    print_training_info(args, all=True)
    # sys.stdout=StdRedirect(log_path)
    #########################
    wandb.init(
            project="drone_anomaly", 
            config=args, 
            job_type="train")
    wandb.config.update({
            "batch_size": args.batch_size,
            "lr": config.lr[0],
            "max_epoch": args.max_epoch
        }, allow_val_change=True)
    

    # data loader
    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)
    print(f'train_Nloader: {len(train_nloader)}')
    print(f'train_Aloader: {len(train_aloader)}')
    print(f'test_loader: {len(test_loader)}')


    # model define and optimizer
    model = Model(args.feature_size, args.batch_size)

    # for name, value in model.named_parameters():
    #     print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.Adam(model.parameters(),
                            lr=config.lr[0], weight_decay=0.005)

    # train & test
    test_info = {"epoch": [], "test_AUC": [], "test_AP": [], "test_F1": []}
    best_AUC = -1
    best_AP = -1
    best_F1 = -1
    # output_path = ''   # put your own path here
    # auc = test(test_loader, model, args, viz, device)

    # train iter
    for epoch in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if epoch > 1 and config.lr[epoch - 1] != config.lr[epoch - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[epoch - 1]
        
        # 分别从正常和异常的数据加载器中获取数据，并训练模型
        if (epoch - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (epoch - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, wandb, device, log_dir, epoch, args)

        # 每 5 个epoch进行一次测试，并保存表现最好的模型
        if epoch % 5 == 0 and epoch > 200:
            auc, ap, f1_info, pred = test(test_loader, model, args, wandb, device, log_dir)

            test_info["epoch"].append(epoch)
            test_info["test_AUC"].append(auc)
            test_info["test_AP"].append(ap)
            test_info["test_F1"].append(f1_info)

            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                # torch.save(model.state_dict(), './ckpt/' + args.model_name + f'{epoch}-i3d.pkl')
                save_best_record(test_info, log_path, 'auc')
                wandb.log({"epoch": test_info["epoch"], 
                            "best_AUC": test_info["test_AUC"] })
                visulization(epoch, pred, log_dir, args.scene)
            
            if test_info["test_AP"][-1] > best_AP:
                best_AP = test_info["test_AP"][-1]
                save_best_record(test_info, log_path, 'ap')
                wandb.log({"epoch": test_info["epoch"], 
                            "best_AP": test_info["test_AP"] })
            
            # breakpoint()
            if test_info["test_F1"][-1]['f1'] > best_F1:
                best_F1 = test_info["test_F1"][-1]['f1']
                save_best_record(test_info, log_path, 'f1')

    torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')

