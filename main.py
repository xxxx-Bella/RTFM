from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from model import Model
from dataset import Dataset
from train import train
from test_10crop import test
import option
from tqdm import tqdm
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
    model = Model(args.feature_size, args.batch_size)

    # for name, value in model.named_parameters():
    #     print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')
    
    log_dir = f'./log/run-{args.run_name}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    optimizer = optim.Adam(model.parameters(),
                            lr=config.lr[0], weight_decay=0.005)

    # train & test
    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    output_path = ''   # put your own path here
    auc = test(test_loader, model, args, wandb, device)
    # wandb.log({"epoch": step, "test_AUC": auc})


    # train iter
    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]
        
        # 分别从正常和异常的数据加载器中获取数据，并训练模型
        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, wandb, device)

        # 每 5 个 epoch 进行一次测试，并保存表现最好的模型
        if step % 5 == 0 and step > 200:

            auc = test(test_loader, model, args, wandb, device)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)

            wandb.log({"epoch": step, "test_AUC": auc})

            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                torch.save(model.state_dict(), './ckpt/' + args.model_name + f'{step}-i3d.pkl')
                save_best_record(test_info, os.path.join(output_path, log_dir, f'{args.run_name}_test_auc.txt'))
                wandb.save('./ckpt/' + args.model_name + f'{step}-i3d.pkl') 

    torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')
    wandb.save('./ckpt/' + args.model_name + 'final.pkl')
