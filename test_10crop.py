import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
import wandb 

# wandb
def test(dataloader, model, args, wandb, device):
    with torch.no_grad():  # 评估模式，不进行反向传播
        model.eval()
        # 处理输入数据，得到预测的 logits
        pred = torch.zeros(0, device=device)

        for i, input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, logits, feat_magnitudes = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))

        # 加载 ground truth 标签文件
        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy')
        elif args.dataset == 'ucf':
            gt = np.load('list/gt-ucf.npy')
        elif args.dataset == 'drone_anomaly':
            gt = np.load('list/gt-da.npy')

        # 将 pred 中的每个值重复 16 次
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        
        # print(f"Length of gt: {len(gt)}")
        # print(f"Length of pred: {len(pred)}")

        if len(gt) != len(pred):
            print(f"Error: gt and pred have different lengths: {len(gt)} vs {len(pred)}")
            breakpoint()
        else:
            fpr, tpr, threshold = roc_curve(list(gt), pred)

        np.save(f'log/run-{args.run_name}/fpr.npy', fpr)
        np.save(f'log/run-{args.run_name}/tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save(f'log/run-{args.run_name}/precision.npy', precision)
        np.save(f'log/run-{args.run_name}/recall.npy', recall)

        # vis.plot_lines('pr_auc', pr_auc)
        # vis.plot_lines('auc', rec_auc)
        # vis.lines('scores', pred)
        # vis.lines('roc', tpr, fpr)
        wandb.log({
            "pr_auc": pr_auc,
            "auc": rec_auc,
            "scores": pred,
            'roc': wandb.plot.line_series(
                xs=fpr,  
                ys=[tpr], 
                keys=["tpr"], 
                title="ROC Curve", 
                xname="FPR" 
            )})

        return rec_auc