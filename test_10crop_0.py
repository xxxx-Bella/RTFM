import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score, average_precision_score
import numpy as np
import wandb 
from utils import visulization


def test(dataloader, model, args, wandb, device, log_dir, epoch):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)

        for i, input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, logits, feat_magnitudes = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))

        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy')
        elif args.dataset == 'ucf':
            gt = np.load('list/gt-ucf.npy')
        elif args.dataset == 'drone_anomaly':
            if args.scene == 'all':
                gt = np.load('/home/featurize/work/yuxin/WVAD/DALE/list/gt-da.npy')
            else: 
                gt = np.load(f'/home/featurize/work/yuxin/WVAD/DALE/list/gt-da-{args.scene}.npy')
        

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        # visulization(epoch, pred, log_dir, args.scene, args.smooth, args.window_size)


        if len(gt) != len(pred):
            print(f"Error: gt and pred have different lengths: {len(gt)} vs {len(pred)}")
            breakpoint()
        else:
            fpr, tpr, th1 = roc_curve(list(gt), pred)

        np.save(f'{log_dir}/fpr.npy', fpr)
        np.save(f'{log_dir}/tpr.npy', tpr)
        np.save(f'{log_dir}/threshold1.npy', th1)

        # AUC, average_precision
        rec_auc = auc(fpr, tpr)
        ap = average_precision_score(list(gt), pred)
        print(f'auc : {str(rec_auc)}, AP : {str(ap)}')

        # Precision, Recall, F1, Overall_Accuracy
        precision, recall, th2 = precision_recall_curve(list(gt), pred)
        
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_th = th2[np.argmax(f1_scores)]
        best_f1 = np.max(f1_scores)
        best_precision = precision[np.argmax(f1_scores)]
        best_recall = recall[np.argmax(f1_scores)]
        binary_pred = (pred >= best_th).astype(int)
        oa = accuracy_score(list(gt), binary_pred)
        f1_info = {'precision': float(best_precision), 'recall': float(best_recall), 'f1': float(best_f1), 'threshold': float(best_th), 'OA' : oa}

        # viz.plot_lines('pr_auc', pr_auc)
        # viz.plot_lines('auc', rec_auc)
        # viz.lines('scores', pred)
        # viz.lines('roc', tpr, fpr)
        # return rec_auc
        pr_auc = auc(recall, precision)
        np.save(f'{log_dir}/precision.npy', precision)  # tp / (tp + fp)
        np.save(f'{log_dir}/recall.npy', recall)  # tp / (tp + fn)
        np.save(f'{log_dir}/threshold2.npy', th2)  # tp / (tp + fn)

        wandb.log({
            # "pr_auc": pr_auc,
            "auc": rec_auc,
            "AP" : ap,
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

        return rec_auc, ap, f1_info, pred