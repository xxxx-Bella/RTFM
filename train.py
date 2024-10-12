import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_tensor_type('torch.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss
from utils import draw_distribution


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



class My_loss(torch.nn.Module):
    def __init__(self, alpha, beta, lambda1, lambda2, margin):
        super(My_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.margin = margin  # 3
        self.sigmoid = torch.nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.criterion = torch.nn.BCELoss()

    def forward(self, y_pred_normal, y_pred_abnormal, nlabel, alabel, feat_n, feat_a):
        '''
        y_pred_normal, y_pred_abnormal: torch.Size([4, 1])
        feat_n, feat_a: torch.Size([40, 3, 2048])  # (bs*n_crops, topk, feat_dim)
        nlabel, alabel: torch.Size([4])  # (bs) label for each video
        '''
        y_pred_abnormal = y_pred_abnormal  # 异常样本的预测得分
        y_pred_normal = y_pred_normal      # 正常样本的预测得分

        y_pred = torch.cat((y_pred_normal, y_pred_abnormal), 0)  # 完整的得分集
        y_pred = y_pred.squeeze()  # torch.Size([8])

        label = torch.cat((nlabel, alabel), 0)  # 完整的标签集
        label = label.cuda()  # torch.Size([8]) tensor([0., 0., 0., 0., 1., 1., 1., 1.])

        # (Eq.1) l_f: 衡量 模型输出的预测得分 和真实标签 之间的误差
        loss_cls = self.criterion(y_pred, label)  # BCE loss (Eq.7)

        # (Eq.1) l_s
        mean_abn = torch.mean(feat_a, dim=1)
        feat_a_l2 = torch.norm(mean_abn, p=2, dim=1)  # 每个异常样本的L2范数 
        loss_abn = torch.abs(100 - feat_a_l2)  # 控制异常特征大小，使与 m 接近 ([40]) (self.margin=100)
        loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)  # torch.Size([40])
        l_s = torch.mean((loss_abn + loss_nor) ** 2)  # float


        # new-var-loss
        # 计算异常和正常特征的方差
        variance_abn = torch.var(feat_a, dim=1)  # variance of 3 abnormal samples. torch.Size([40, 2048])
        variance_nor = torch.var(feat_n, dim=1)  # variance of 3 normal samples. torch.Size([40, 2048])
        mean_abn = torch.mean(feat_a, dim=1)  # torch.Size([40, 2048])
        mean_nor = torch.mean(feat_n, dim=1)  # torch.Size([40, 2048])

        # epsilon = 1e-5  # 防止除零
        # relative_variance_diff = torch.abs((variance_abn - variance_nor) / (variance_nor + epsilon))  # torch.Size([40, 2048])
        # variance_loss_2 = torch.mean(relative_variance_diff)  # 求平均

        variance_diff = variance_abn - variance_nor  # 异常方差 - 正常方差
        clipped_variance_diff = torch.clamp(variance_diff, min=0)  # 只保留异常方差大于正常方差的部分
        variance_loss = torch.mean(clipped_variance_diff)

        # variance_diff = torch.abs(variance_abn - variance_nor)  # torch.Size([40, 2048]) 方差差异通常为非负值，且只关心差异大小。用 torch.abs 保证所有差异为正，并能逐特征地衡量每个维度的差异
        # variance_loss = torch.mean(variance_diff)  # tensor float.

        # mean_diff = torch.mean(torch.abs(mean_abn - mean_nor))  
        # mean_loss = 1 - mean_diff  # Maximize mean difference 
        # m = 3
        mean_diff = torch.norm(mean_abn - mean_nor, p=2, dim=1)  # l2 norm, dim 1, torch.Size([40]) 均值差异涉及所有特征维度的整体差异，用 L2 范数（torch.norm）可以更好地衡量总体差异。这样能得到每个样本的整体均值差异，而不仅是单个特征维度的差异
        # mean_loss = torch.mean(mean_diff)  # 最大化 正常和异常视频均值的 差异
        mean_loss = torch.mean(torch.clamp(self.margin - mean_diff, min=0))
        # print(f'variance_abn = {variance_abn}, variance_nor = {variance_nor}')
        # print(f'mean_abn = {mean_abn}, mean_nor = {mean_nor}')
        # print(f'mean_diff = {mean_diff}')
        # print(f'mean_loss = {mean_loss}, variance_loss = {variance_loss}') 
        # breakpoint()

        loss_dual = self.alpha * mean_loss + self.beta * variance_loss
        # loss_dual = 0.5 * mean_loss + 0.5 * variance_loss

        # new-total-loss
        loss_total = self.lambda1 * loss_cls + self.lambda2 * loss_dual

        # print(f'loss_cls = {loss_cls}, loss_dual = {loss_dual}')
        
        # loss_total = loss_cls + self.alpha * l_s  # original (Eq.1)


        return loss_total
        # return loss_cls, l_s


def train(nloader, aloader, model, batch_size, optimizer, scheduler, wandb, device, log_dir, epoch, args):
    '''
    nloader, aloader: (bs, n_crops, n_segments, feature_dim)
    '''
    with torch.set_grad_enabled(True):
        
        model.train()

        ninput, nlabel = next(nloader)  # torch.Size([4, 10, 32, 2048]), torch.Size([4])
        ainput, alabel = next(aloader)  # torch.Size([4, 10, 32, 2048]), torch.Size([4])
        # (bs, n_crops, n_segments, feat_dim), (bs)

        input = torch.cat((ninput, ainput), 0).to(device)  # torch.Size([8, 10, 32, 2048])

        y_pred_abnormal, y_pred_normal, feat_select_abn, feat_select_normal, y_pred, _ = model(input)
        # [4, 1], [4, 1], [40, 3, 2048], [40, 3, 2048], [8, 32, 1], [8, 32] train: bs=4, T=32
        # [1, 1], [1, 1], [10, 3, 2048], [10, 3, 2048], [1, 37, 1], [1, 37] test: bs=1

        # if epoch % 500 == 0 and epoch > 199:
        #     draw_distribution(feat_select_normal, feat_select_abn, log_dir, epoch)

        y_pred = y_pred.view(batch_size * 32 * 2, -1)
        y_pred = y_pred.squeeze()  # 

        abn_y_pred = y_pred[batch_size * 32:]

        # one batch
        nlabel = nlabel[0:batch_size]  # torch.Size([4])
        alabel = alabel[0:batch_size]  # torch.Size([4])
        
        loss_sparse = sparsity(abn_y_pred, batch_size, 8e-3)
        loss_smooth = smooth(abn_y_pred, 8e-4)

        # alpha = 1   # loss_mean
        # beta = 1    # loss_var
        # margin = 3  # loss_mean
        # # lambda1, lambda2 = 1, 0.5  # loss_cls, loss_dd (all scenes except bike)
        # lambda1, lambda2 = 1, 1  # loss_cls, loss_dd (bike)
        # lambda3, lambda4 = 0.1, 0.1  # loss_smooth, loss_sparse


        my_loss_fn = My_loss(args.alpha, args.beta, args.lambda1, args.lambda2, args.margin)
        loss_my = my_loss_fn(y_pred_normal, y_pred_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn)
        # loss_cls, loss_s = my_loss_fn(y_pred_normal, y_pred_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn)

        # # new_ls1: 使用该损失函数时，可以将 mean_vector_miu 作为 anchor, feat_select_abn 作为 negative，feat_select_normal 作为 positive
        # triplet_loss = TripletLoss(margin=1.0)
        # loss_triplet = triplet_loss(anchor, feat_select_normal, feat_select_abn)

        # # new_ls2: 假设 label 表示正样本与负样本对的标签 (1:相似, 0:非相似)
        # contrastive_loss = ContrastiveLoss(margin=1.0)
        # loss_contrastive = contrastive_loss(feat_select_normal, feat_select_abn, torch.zeros_like(nlabel))

        cost_main = loss_my     # original: loss_cls + alpha * loss_s
        # cost_main = loss_cls + alpha * loss_contrastive
        cost = cost_main + args.lambda3*loss_smooth + args.lambda4*loss_sparse
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
        print(f'Epoch [{epoch+1}/{args.max_epoch}], Loss: {cost.item():.4f}, main loss: {cost_main.item():.4f}, LR: {current_lr:.6f}')

        return cost
