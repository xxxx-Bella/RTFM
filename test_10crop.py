import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score
import numpy as np
import wandb 

# wandb
def test(dataloader, model, args, wandb, device):
    with torch.no_grad():  # 评估模式，不进行反向传播
        model.eval()
        # 处理输入数据，得到预测的 logits
        pred = torch.zeros(0, device=device)

        for i, input in enumerate(dataloader):
            input = input.to(device)  # torch.Size([1, 37, 10, 2048])
            input = input.permute(0, 2, 1, 3)  # torch.Size([1, 10, 37, 2048])
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, y_pred, feat_magnitudes = model(inputs=input)
            # [1, 1], [1, 1], [10, 3, 2048], [10, 3, 2048], [1, 37, 1], [1, 37] test: bs=1
            # [4, 1], [4, 1], [40, 3, 2048], [40, 3, 2048], [8, 32, 1], [8, 32] train: bs=4, T=32
            logits = torch.squeeze(y_pred, 1)  # torch.Size([1, 37, 1])
            breakpoint()
            logits = torch.mean(logits, 0)  # torch.Size([37, 1])
            sig = logits
            pred = torch.cat((pred, sig))
        
        # pred: torch.Size([2322, 1])
        

        # 加载 ground truth 标签文件
        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy')
        elif args.dataset == 'ucf':
            gt = np.load('list/gt-ucf.npy')
        elif args.dataset == 'drone_anomaly':
            gt = np.load('list/gt-da.npy')
        # len(gt) = 37152

        # 将 pred 中的每个值重复 16 次
        pred = list(pred.cpu().detach().numpy())  # len=2322
        pred = np.repeat(np.array(pred), 16)  # len=37152

        if len(gt) != len(pred):
            print(f"Error: gt and pred have different lengths: {len(gt)} vs {len(pred)}")
            breakpoint()
        else:
            fpr, tpr, threshold = roc_curve(list(gt), pred)

        np.save(f'log/run-{args.run_name}/fpr.npy', fpr)
        np.save(f'log/run-{args.run_name}/tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)

        print('auc : ' + str(rec_auc))

        # Precision, Recall, AP
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)  # AP?
        np.save(f'log/run-{args.run_name}/precision.npy', precision)
        np.save(f'log/run-{args.run_name}/recall.npy', recall)
        
        # # F1-score
        # pred_labels = [1 if p >= 0.5 else 0 for p in pred] 
        # f1 = f1_score(gt, pred_labels)
        # print('F1 Score : ' + str(f1))
        
        # # Overall Accuracy (OA)
        # oa = accuracy_score(gt, pred_labels)
        # print('Overall Accuracy (OA) : ' + str(oa))

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