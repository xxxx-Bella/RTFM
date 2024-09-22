import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_tensor_type('torch.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss

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

# # new_ls2
# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, output1, output2, label):
#         # label=1 表示相似样本，label=0 表示非相似样本
#         dist = torch.norm(output1 - output2, p=2, dim=1)  # L2 范数
#         loss = (1 - label) * F.relu(self.margin - dist) + label * dist
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


class RTFM_loss(torch.nn.Module):
    def __init__(self, alpha, margin):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin  # m = 100
        self.sigmoid = torch.nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.criterion = torch.nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        
        score_abnormal = score_abnormal  # 异常样本的预测得分
        score_normal = score_normal      # 正常样本的预测得分

        score = torch.cat((score_normal, score_abnormal), 0)  # 完整的得分集
        score = score.squeeze()

        label = torch.cat((nlabel, alabel), 0)  # 完整的标签集
        label = label.cuda()

        # (Eq.1) l_f: 衡量 模型输出的预测得分 和真实标签 之间的误差
        loss_cls = self.criterion(score, label)  # BCE loss in the score space (Eq.7)

        # (Eq.1) l_s
        feat_a_l2 = torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1)  # 每个异常样本的L2范数
        loss_abn = torch.abs(self.margin - feat_a_l2)  # 控制异常特征大小，使与 m 接近 ([40])
        loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)  # torch.Size([40])
        l_s = torch.mean((loss_abn + loss_nor) ** 2)  # float

        loss_total = loss_cls + self.alpha * l_s  # original (Eq.1)

        return loss_total
        # return loss_cls, l_s


def train(nloader, aloader, model, batch_size, optimizer, wandb, device):
    with torch.set_grad_enabled(True):
        
        model.train()

        ninput, nlabel = next(nloader)
        ainput, alabel = next(aloader)

        input = torch.cat((ninput, ainput), 0).to(device)

        score_abnormal, score_normal, feat_select_abn, feat_select_normal, scores, _ = model(input)   # b*32  x 2048
        # [4, 1], [4, 1], [40, 3, 2048], [40, 3, 2048], [8, 32, 1], [8, 32]
        # breakpoint()

        scores = scores.view(batch_size * 32 * 2, -1)
        scores = scores.squeeze()

        abn_scores = scores[batch_size * 32:]

        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size]
        
        loss_sparse = sparsity(abn_scores, batch_size, 8e-3)
        loss_smooth = smooth(abn_scores, 8e-4)


        alpha = 0.0001
        rtfm_loss_fn = RTFM_loss(0.0001, 100)
        loss_rftm = rtfm_loss_fn(score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn)
        # loss_cls, loss_s = rtfm_loss_fn(score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn)

        # # new_ls1: 使用该损失函数时，可以将 mean_vector_miu 作为 anchor, feat_select_abn 作为 negative，feat_select_normal 作为 positive
        # triplet_loss = TripletLoss(margin=1.0)
        # loss_triplet = triplet_loss(anchor, feat_select_normal, feat_select_abn)

        # # new_ls2: 假设 label 表示正样本与负样本对的标签 (1:相似, 0:非相似)
        # contrastive_loss = ContrastiveLoss(margin=1.0)
        # loss_contrastive = contrastive_loss(feat_select_normal, feat_select_abn, torch.zeros_like(nlabel))

        cost_main = loss_rftm     # original  # :loss_cls + alpha * loss_s
        # cost_main = loss_cls + alpha * loss_contrastive
        cost = cost_main + loss_smooth + loss_sparse

        wandb.log({"cost": cost})
        wandb.log({
            "loss": cost.item(),
            "smooth loss": loss_smooth.item(),
            "sparsity loss": loss_sparse.item()
        })

        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()


