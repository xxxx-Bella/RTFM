import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score, average_precision_score
import numpy as np
import wandb 
from utils import visulization


# wandb
def test(dataloader, model, args, wandb, device, log_dir):
    with torch.no_grad():  # 评估模式，不进行反向传播
        model.eval()
        # 处理输入数据，得到预测的 logits
        pred = torch.zeros(0, device=device)
        # breakpoint()
        for i, input in enumerate(dataloader):
            input = input.to(device)  # torch.Size([1, 37, 10, 2048])
            input = input.permute(0, 2, 1, 3)  # torch.Size([1, 10, 37, 2048])
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, y_pred, feat_magnitudes = model(inputs=input)
            # [1, 1], [1, 1], [10, 3, 2048], [10, 3, 2048], [1, 37, 1], [1, 37] test: bs=1
            # [4, 1], [4, 1], [40, 3, 2048], [40, 3, 2048], [8, 32, 1], [8, 32] train: bs=4, T=32
            logits = torch.squeeze(y_pred, 1)  # torch.Size([1, 37, 1])
            # breakpoint()
            logits = torch.mean(logits, 0)  # torch.Size([37, 1])
            sig = logits
            pred = torch.cat((pred, sig))
        
        # all: pred shape = torch.Size([2322, 1])
        

        # 加载 ground truth 标签文件
        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy')
        elif args.dataset == 'ucf':
            gt = np.load('list/gt-ucf.npy')
        elif args.dataset == 'drone_anomaly':
            if args.scene == 'all':
                gt = np.load('list/gt-da.npy')
            else:
                gt = np.load(f'list/gt-da-{args.scene}.npy')

        # all: len(gt) = 37056

        # 将 pred 中的每个值重复 16 次
        pred = list(pred.cpu().detach().numpy())  # len=2322, 99
        pred_tmp = np.repeat(np.array(pred), 1)
        # breakpoint()
        pred = np.repeat(np.array(pred), 16)  # len=37056, vehicle:(1584,), railway:(1360,)
        # visulization(pred, log_dir, args.scene)
        print(f'Scene: {args.scene}, Pred: {len(pred_tmp)}')

        if len(gt) != len(pred):
            print(f"Error: gt and pred have different lengths: {len(gt)} vs {len(pred)}")
            breakpoint()
        else:
            fpr, tpr, th1 = roc_curve(list(gt), pred)

        np.save(f'{log_dir}/fpr.npy', fpr)
        np.save(f'{log_dir}/tpr.npy', tpr)
        np.save(f'{log_dir}/threshold1.npy', th1)

        # AUC, AP, Precision, Recall, F1, OA
        rec_auc = auc(fpr, tpr)
        # ap = average_precision_score(list(gt), pred)
        precision, recall, th2 = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        print(f'auc : {str(rec_auc)}, AP : {str(pr_auc)}')
        
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_th = th2[np.argmax(f1_scores)]
        best_f1 = np.max(f1_scores)
        best_precision = precision[np.argmax(f1_scores)]
        best_recall = recall[np.argmax(f1_scores)]
        binary_pred = (pred >= best_th).astype(int)
        oa = accuracy_score(list(gt), binary_pred)
        f1_info = {'precision': float(best_precision), 'recall': float(best_recall), 'f1': float(best_f1), 'threshold': float(best_th), 'OA' : oa}

        # f1_info = [[f'precision: {str(best_precision)}'], 
        #             [f'recall: {str(best_recall)}'], 
        #             [f'f1: {str(best_f1)}'], 
        #             [f'threshold: {str(best_th)}'], 
        #             [f'OA : {str(oa)}']]
        print(f1_info)
        # breakpoint()
        
        np.save(f'{log_dir}/precision.npy', precision)  # tp / (tp + fp)
        np.save(f'{log_dir}/recall.npy', recall)  # tp / (tp + fn)
        np.save(f'{log_dir}/threshold2.npy', th2)  # tp / (tp + fn)

        wandb.log({
            # "pr_auc": pr_auc,
            "auc": rec_auc,
            "AP" : pr_auc,
            "OA" : oa, 
            "threshold" : best_th,
            "precision" : best_precision, 
            "recall" : best_recall, 
            "f1" : best_f1,
            # "scores": pred,
            'roc': wandb.plot.line_series(
                xs=fpr,  
                ys=[tpr], 
                keys=["tpr"], 
                title="ROC Curve", 
                xname="FPR" 
            )})

        return rec_auc, pr_auc, f1_info, pred