from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from model_new import Model
from dataset import Dataset
from train import train
from test_10crop import test
import option
from tqdm import tqdm  # 方便地显示循环进度条
# from utils import Visualizer
from config import *
import wandb


if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)

    # wandb.init(project="shanghai_tech_10_crop", config=args, name="training")
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
    train_nset = Dataset(args, test_mode=False, is_normal=True)
    train_aset = Dataset(args, test_mode=False, is_normal=False)
    print(f'train_Nset: {len(train_nset)}')
    print(f'train_Aset: {len(train_aset)}')

    train_nloader = DataLoader(train_nset,
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(train_aset,
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    print(f'train_Nloader: {len(train_nloader)}')
    print(f'train_Aloader: {len(train_aloader)}')
    print(f'test_loader: {len(test_loader)}')
    # breakpoint()

    # model define and optimizer
    model = Model(args.feature_size, args.batch_size, args)

    # for name, value in model.named_parameters():
    #     print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')
    
    log_dir = f'./log/run-{args.run_name}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    # optimizer = optim.Adam(model.parameters(), lr=config.lr[0], weight_decay=0.005)

    # new-lr
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 每10个epoch，学习率衰减10倍
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # T_max是周期的步长

    # train & test
    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    output_path = log_dir   # put your own path here
    # print('auc = test(test_loader, model,..)')
    # auc = test(test_loader, model, args, wandb, device)  # what for? deleted

    # train iter
    print('Training ...')
    for epoch in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True  # 动态调整进度条的宽度
    ):
        if epoch > 1 and config.lr[epoch - 1] != config.lr[epoch - 2]:
            for param_group in optimizer.param_groups:
                # param_group["lr"] = config.lr[epoch - 1]
                # print(f'lr = {param_group["lr"]}') 
                pass
        
        # 分别从 train_nloader和train_aloader中获取数据，以便在训练模型时交替使用正常和异常数据
        if (epoch - 1) % len(train_nloader) == 0:  # len(train_nloader) = 15
            # 每经历15个epoch 后，重新创建一个 train_nloader 的迭代器 loadern_iter
            loadern_iter = iter(train_nloader)
        if (epoch - 1) % len(train_aloader) == 0:  # len(train_aloader) = 6
            loadera_iter = iter(train_aloader)

        loss = train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, scheduler, wandb, device, log_dir, epoch, args)
        # breakpoint()

        # 每 5 个 epoch 进行一次测试，并保存表现最好的模型
        if epoch % 5 == 0 and epoch > 100:
            print('Testing ...')
            # print('loss:', loss.item())
            print('auc = test(test_loader, model,..)')
            auc = test(test_loader, model, args, wandb, device)
            test_info["epoch"].append(epoch)
            test_info["test_AUC"].append(auc)

            wandb.log({"epoch": epoch, "test_AUC": auc})

            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                # torch.save(model.state_dict(), './ckpt/' + args.model_name + f'{epoch}-i3d.pkl')
                save_best_record(test_info, os.path.join(output_path, log_dir, f'{args.run_name}.txt'))
                wandb.log({"epoch": test_info["epoch"], "best_AUC": test_info["test_AUC"]})
    torch.save(model.state_dict(), './ckpt/' + f'{args.run_name}-final.pkl')
