import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_tensor_type('torch.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss
import matplotlib.pyplot as plt
import os
import seaborn as sns

# # new_ls1
# class TripletLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(TripletLoss, self).__init__()
#         self.margin = margin

#     def forward(self, anchor, positive, negative):
#         pos_dist = torch.norm(anchor - positive, p=2, dim=1)  # Anchor 和 Positive 的距离
#         neg_dist = torch.norm(anchor - negative, p=2, dim=1)  # Anchor 和 Negative 的距离
        
#         # Triplet Loss：使正样本距离最小化，负样本距离最大化
#         loss = F.relu(pos_dist - neg_dist + self.margin)
#         return loss.mean()




def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return lamda1*loss


def l1_penalty(var):
    return torch.mean(torch.norm(var, dim=0))


class SigmoidMAELoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)


class SigmoidCrossEntropyLoss(torch.nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(- torch.abs(x))
        return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))


def draw_distribution(feat_n, feat_a, log_dir,step):
    # feat_a = torch.Size([40, 3, 2048])  # 异常特征
    # feat_n = torch.Size([40, 3, 2048])  # 正常特征

    # 1. 对 top-k 维度 (dim=1) 取平均，获得 [40, 2048] 的张量
    feat_a_mean = torch.mean(feat_a, dim=2)  # Shape: [40, 3]
    feat_n_mean = torch.mean(feat_n, dim=2)  # Shape: [40, 3]
    # feat_a_mean = torch.mean(feat_a, dim=1)  # Shape: [40, 2048]
    # feat_n_mean = torch.mean(feat_n, dim=1)  # Shape: [40, 2048]

    # 2. 转为 numpy 数组，方便绘图
    feat_a_np = feat_a_mean.cpu().detach().numpy()  # Shape: [40, 2048]
    feat_n_np = feat_n_mean.cpu().detach().numpy()  # Shape: [40, 2048]

    # 3. 选择一些特征维度进行可视化，比如第 0 个和第 1 个维度
    feature_dim = [0, 1]  # 你可以选择多个维度进行绘制

    for dim in feature_dim:
        plt.figure(figsize=(8, 6))

        # 绘制异常样本的特征分布
        plt.hist(feat_a_np[:, dim], bins=30, alpha=0.5, label='Abnormal Feature', color='red', density=True)
        # 绘制正常样本的特征分布
        plt.hist(feat_n_np[:, dim], bins=30, alpha=0.5, label='Normal Feature', color='blue', density=True)
        
        # 使用Seaborn绘制异常样本的密度曲线
        sns.kdeplot(feat_a_np[:, dim], color='red', label='Abnormal Density', linewidth=2)
        # 使用Seaborn绘制正常样本的密度曲线
        sns.kdeplot(feat_n_np[:, dim], color='blue', label='Normal Density', linewidth=2)

        plt.title(f'Feature Distribution')
        plt.xlabel('Feature Value')  # 特征值：异常样本在特定维度上的特征值
        plt.ylabel('Density')  # 概率密度：特征值在各个区间上的概率密度，表示该维度上特征值出现的相对频率
        plt.legend()

        # dimension = 'bs*ncrops' if dim == 0 else 'feat-dim'
        output_path = os.path.join(log_dir, f'step{step}-dim{dim}.png')
        plt.savefig(output_path)
        plt.close()


class My_loss(torch.nn.Module):
    def __init__(self, alpha, margin):
        super(My_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin  # m = 100
        self.sigmoid = torch.nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.criterion = torch.nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        '''
        score_normal, score_abnormal: torch.Size([4, 1])
        feat_n, feat_a: torch.Size([40, 3, 2048])  # (bs*n_crops, topk, feat_dim)
        nlabel, alabel: torch.Size([4])  # (bs) label for each video
        '''
        score_abnormal = score_abnormal  # 异常样本的预测得分
        score_normal = score_normal      # 正常样本的预测得分

        score = torch.cat((score_normal, score_abnormal), 0)  # 完整的得分集
        score = score.squeeze()  # torch.Size([8])

        label = torch.cat((nlabel, alabel), 0)  # 完整的标签集
        label = label.cuda()  # torch.Size([8]) tensor([0., 0., 0., 0., 1., 1., 1., 1.])

        # (Eq.1) l_f: 衡量 模型输出的预测得分 和真实标签 之间的误差
        loss_cls = self.criterion(score, label)  # BCE loss in the score space (Eq.7)

        # (Eq.1) l_s
        feat_a_l2 = torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1)  # 每个异常样本的L2范数
        loss_abn = torch.abs(self.margin - feat_a_l2)  # 控制异常特征大小，使与 m 接近 ([40])
        loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)  # torch.Size([40])
        l_s = torch.mean((loss_abn + loss_nor) ** 2)  # float


        # new-var-loss
        # 计算异常和正常特征的方差
        variance_abn = torch.var(feat_a, dim=1)  # variance of 3 abnormal samples. torch.Size([40, 2048])
        variance_nor = torch.var(feat_n, dim=1)  # variance of 3 normal samples. torch.Size([40, 2048])
        mean_abn = torch.mean(feat_a, dim=1)
        mean_nor = torch.mean(feat_n, dim=1)

        variance_loss = torch.mean(torch.abs(variance_abn - variance_nor))  # tensor float. Reduce variance difference
        # mean_diff = torch.mean(torch.abs(mean_abn - mean_nor))  
        # mean_loss = 1 - mean_diff  # Maximize mean difference 
        mean_diff = torch.norm(mean_abn - mean_nor, p=2, dim=1)
        mean_loss = -torch.mean(mean_diff)
        # print(f'variance_abn = {variance_abn}, variance_nor = {variance_nor}')
        # print(f'mean_abn = {mean_abn}, mean_nor = {mean_nor}')
        # print(f'variance_loss = {variance_loss}, mean_diff = {mean_diff}')
        # print(f'mean_loss = {mean_loss}')
        # breakpoint()

        # new-total-loss
        loss_total = loss_cls + self.alpha * (mean_loss + variance_loss)
        # print()
        # loss_total = loss_cls + self.alpha * l_s  # original (Eq.1)


        return loss_total
        # return loss_cls, l_s


def train(nloader, aloader, model, batch_size, optimizer, scheduler, wandb, device, log_dir, step, args):
    with torch.set_grad_enabled(True):
        
        model.train()

        ninput, nlabel = next(nloader)  # torch.Size([4, 10, 32, 2048]), torch.Size([4])
        ainput, alabel = next(aloader)  # torch.Size([4, 10, 32, 2048]), torch.Size([4])
        # (bs, n_crops, n_segments, feat_dim), (bs)

        input = torch.cat((ninput, ainput), 0).to(device)  # torch.Size([8, 10, 32, 2048])

        score_abnormal, score_normal, feat_select_abn, feat_select_normal, scores, _ = model(input)   # b*32  x 2048
        # [4, 1], [4, 1], [40, 3, 2048], [40, 3, 2048], [8, 32, 1], [8, 32]

        if step % 200 == 0 and step > 199:
            draw_distribution(feat_select_normal, feat_select_abn, log_dir, step)
            # breakpoint()

        scores = scores.view(batch_size * 32 * 2, -1)
        scores = scores.squeeze()

        abn_scores = scores[batch_size * 32:]

        # one batch
        nlabel = nlabel[0:batch_size]  # torch.Size([4])
        alabel = alabel[0:batch_size]  # torch.Size([4])
        # breakpoint()
        
        loss_sparse = sparsity(abn_scores, batch_size, 8e-3)
        loss_smooth = smooth(abn_scores, 8e-4)

        # alpha = 0.0001  # l_s = around 15000
        alpha = 0.1  # l_variance = around 8
        my_loss_fn = My_loss(alpha, 100)
        loss_my = my_loss_fn(score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn)
        # loss_cls, loss_s = my_loss_fn(score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn)

        # # new_ls1: 使用该损失函数时，可以将 mean_vector_miu 作为 anchor, feat_select_abn 作为 negative，feat_select_normal 作为 positive
        # triplet_loss = TripletLoss(margin=1.0)
        # loss_triplet = triplet_loss(anchor, feat_select_normal, feat_select_abn)

        # # new_ls2: 假设 label 表示正样本与负样本对的标签 (1:相似, 0:非相似)
        # contrastive_loss = ContrastiveLoss(margin=1.0)
        # loss_contrastive = contrastive_loss(feat_select_normal, feat_select_abn, torch.zeros_like(nlabel))

        cost_main = loss_my     # original: loss_cls + alpha * loss_s
        # cost_main = loss_cls + alpha * loss_contrastive
        cost = cost_main + loss_smooth + loss_sparse
        # print(f'loss_main = {loss_my}')
        # print(f'loss_smooth = {loss_smooth}, loss_sparse = {loss_sparse}')
        # print(f'cost = {cost}')
        # breakpoint()

        wandb.log({
            "loss": cost.item(),
            "main loss": cost_main.item(),
            "smooth loss": loss_smooth.item(),
            "sparsity loss": loss_sparse.item()
        })
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        scheduler.step()  # update lr

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{step+1}/{args.max_epoch}], Loss: {cost.item():.4f}, main loss: {cost_main.item():.4f}, LR: {current_lr:.6f}')

        return cost
