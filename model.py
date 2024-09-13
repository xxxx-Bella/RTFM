import torch
import torch.nn as nn
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.FloatTensor')

def weight_init(m):
    '''初始化模型中的权重参数'''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class _NonLocalBlockND(nn.Module):
    '''通用的 Non-Local Block 模块，适用于 1D, 2D 或 3D 数据.
    Non-Local 模块: 捕获远距离依赖关系的机制 
    the global temporal dependencies between video snippets
    '''
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample  # 是否对输入进行下采样

        self.in_channels = in_channels
        self.inter_channels = inter_channels  # 内部处理时的通道数

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d
        
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer: # 是否使用 Batch Normalization
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)  # 使用 Batch Normalization
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        # 将输入通道数缩减到 inter_channels，并可能进行下采样
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # 提取非局部信息
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        # 非局部操作
        f = torch.matmul(theta_x, phi_x)  # 矩阵乘法, 计算特征间的相似性
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)  # W 输出通道数恢复到 in_channels
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    ''' _NonLocalBlockND 的 1D 特化版本'''
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)



class Aggregate(nn.Module):
    '''特征聚合模块，用于处理输入的特征. 包含多个卷积层和 Non-Local Block.
    目的是在时间维度上 聚合特征 并捕获远程依赖关系
    '''
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = len_feature  # 输入特征的长度（即输入的通道数，通常是 2048）

        # 使用了不同膨胀率 (dilation) 的卷积核 来捕获不同范围的上下文信息
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3, 
                      stride=1, dilation=1, padding=1),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7),
        )

        # 使用 1x1 卷积对特征进行进一步压缩
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1,
                      stride=1, padding=0, bias = False), 
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        
        # 将所有特征融合在一起
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3,
                      stride=1, padding=1, bias=False), # should we keep the bias?
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            # nn.dropout(0.7)
        )

        self.non_local = NONLocalBlock1D(512, sub_sample=False, bn_layer=True)


    def forward(self, x):
            # x: (B, T, F), which means (bs, timestep, feature)
            out = x.permute(0, 2, 1)  # 交换维度以适应卷积操作
            residual = out

            # 三层不同膨胀率的卷积 捕获不同范围的上下文信息
            out1 = self.conv_1(out)
            out2 = self.conv_2(out)
            out3 = self.conv_3(out)
            out_d = torch.cat((out1, out2, out3), dim = 1)

            # 1x1卷积和Non-Local操作
            out = self.conv_4(out)  # 使用 1x1 卷积对特征进行进一步压缩
            out = self.non_local(out)  # 引入 Non-Local Block 来捕获长距离依赖
            
            out = torch.cat((out_d, out), dim=1)
            out = self.conv_5(out)   # fuse all the features together

            # 残差连接：输出与输入相加，形成残差连接，保留原始特征的部分信息
            out = out + residual
            out = out.permute(0, 2, 1)  
            # out: (B, T, 1)

            return out


class Model(nn.Module):
    def __init__(self, n_features, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.num_segments = 32  # T snippets each video
        self.k_abn = self.num_segments // 10  # k = 3 = 32 // 10 (topk: top 10%)
        self.k_nor = self.num_segments // 10

        self.Aggregate = Aggregate(len_feature=2048)  # 特征聚合模块, 通常用于聚合时序特征
        self.fc1 = nn.Linear(n_features, 512)  # 全连接层1，将输入特征数减少到512
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)  # 三层全连接层，将特征压缩 并生成最后的异常得分

        self.drop_out = nn.Dropout(0.7)  # 防止过拟合
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)  # 使用自定义的权重初始化方法初始化网络参数

    def forward(self, inputs):
        # inputs: torch.cat((ninput, ainput), 0) 
        k_abn = self.k_abn
        k_nor = self.k_nor

        out = inputs  
        bs, ncrops, t, f = out.size() # bs, num_crops(10), num_frame, features_dim

        out = out.view(-1, t, f)  # (bs*num_crops, time, features)
        out = self.Aggregate(out)  # 聚合特征，输出大小保持不变
        out = self.drop_out(out)

        features = out
        scores = self.relu(self.fc1(features))  # 通过全连接层1和ReLU激活
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))  # 通过全连接层3和Sigmoid激活，得到分数
        scores = scores.view(bs, ncrops, -1).mean(1)  # 平均多个crop的结果
        scores = scores.unsqueeze(dim=2)  # 在最后一维增加一个维度

        # inputs: torch.cat((ninput, ainput), 0) 
        normal_features = features[0:self.batch_size*10]  # 前面10倍bs的视频为正常视频特征
        normal_scores = scores[0:self.batch_size]

        abnormal_features = features[self.batch_size*10:]  # 后面的为异常视频特征
        abnormal_scores = scores[self.batch_size:]

        feat_magnitudes = torch.norm(features, p=2, dim=2)  # 计算特征的 L2 范数, 衡量每个特征的大小
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)
        nfea_magnitudes = feat_magnitudes[0:self.batch_size]  # normal feature magnitudes
        afea_magnitudes = feat_magnitudes[self.batch_size:]  # abnormal feature magnitudes
        n_size = nfea_magnitudes.shape[0]

        if nfea_magnitudes.shape[0] == 1:  # this is for inference, the batch size is 1 (initialize abnormal variables)
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features

        #######  process abnormal videos -> select top3 feature magnitude  #######
        select_idx = torch.ones_like(nfea_magnitudes)
        select_idx = self.drop_out(select_idx)
        afea_magnitudes_drop = afea_magnitudes * select_idx  # 使用Dropout后的特征大小
        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]  # 从特征中选择Top-k的片段
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])

        abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
        abnormal_features = abnormal_features.permute(1, 0, 2,3)

        total_select_abn_feature = torch.zeros(0, device=inputs.device)
        for abnormal_feature in abnormal_features:
            # 根据选出的index (idx_abn_feat), 将特征从特征张量 (abnormal_features) 中选出来 
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat) # top 3 features magnitude in abnormal bag
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)  # top 3 scores in abnormal bag based on the top-3 magnitude


        ####### process normal videos -> select top3 feature magnitude #######
        select_idx_normal = torch.ones_like(nfea_magnitudes)
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]  # 从特征中选择Top-k的片段
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

        normal_features = normal_features.view(n_size, ncrops, t, f)
        normal_features = normal_features.permute(1, 0, 2, 3)

        total_select_nor_feature = torch.zeros(0, device=inputs.device)
        # 根据选出的index (idx_normal_feat), 将特征从特征张量 (normal_features) 中选出来 
        for nor_fea in normal_features:
            feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)  # top 3 features magnitude in normal bag (hard negative)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1)  # top 3 scores in normal bag

        feat_select_abn = total_select_abn_feature
        feat_select_normal = total_select_nor_feature

        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_select_abn, feat_select_abn, scores, feat_select_abn, feat_select_abn, feat_magnitudes
        # return score_abnormal, score_normal, feat_select_abn, feat_select_normal, scores, feat_magnitudes