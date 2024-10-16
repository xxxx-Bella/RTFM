import torch
import torch.nn as nn
import torch.nn.init as torch_init
# from normal_head import NormalHead

torch.set_default_tensor_type('torch.FloatTensor')

def weight_init(m):
    '''初始化模型中的权重参数'''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

'''
new?: 
非局部块的计算复杂度较高，尤其是对于长时间序列。可以考虑使用更轻量级的注意力机制，如Self-Attention或Transformer替代，减少计算量;如果模型的数据特征在时间或空间维度上局部相关性较强，可以尝试局部注意力机制，减少全局范围的计算负担
'''
class _NonLocalBlockND(nn.Module):
    '''通用的 Non-Local Block 模块，适用于 1D, 2D 或 3D 数据. Non-local Neural Networks
    Non-Local 模块: 捕获远距离依赖关系的机制 
    the global temporal dependencies between video snippets. 
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
        # x.shape = torch.Size([80, 512, 32]) (bs*n_crops, feature_dim/4, n_segments)
        batch_size = x.size(0)

        # 将输入通道数缩减到 inter_channels，并可能进行下采样
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # torch.Size([80, 256, 32])
        g_x = g_x.permute(0, 2, 1)  # torch.Size([80, 32, 256])

        # 提取非局部信息
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # conv_nd torch.Size([80, 256, 32])
        theta_x = theta_x.permute(0, 2, 1)  # for matmul torch.Size([80, 32, 256])
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # conv_nd torch.Size([80, 256, 32])

        # 非局部操作
        f = torch.matmul(theta_x, phi_x)  # 矩阵乘法, 计算特征间的相似性 torch.Size([80, 32, 32])
        N = f.size(-1)  # 32
        f_div_C = f / N  # torch.Size([80, 32, 32])

        y = torch.matmul(f_div_C, g_x) # Eq.1 in paper "Non-local Neural Networks" torch.Size([80, 32, 256])
        y = y.permute(0, 2, 1).contiguous()  # torch.Size([80, 256, 32])
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # torch.Size([80, 256, 32])
        W_y = self.W(y)  # W 输出通道数恢复到 in_channels  torch.Size([80, 512, 32])
        z = W_y + x  # torch.Size([80, 512, 32])

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


# new-c
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1)
        self.conv2 = nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(out_channels // 4, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out + residual)
        return out


class TransformerBlock(nn.Module):
    '''基于 Transformer 的自注意力机制，用于捕获时间序列中的长距离依赖'''
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (T, bs, D/4) torch.Size([32, 80, 512])
        breakpoint()
        attn_output, _ = self.attention(x, x, x)  # 自注意力机制 attn_output: (T, bs, D/4) torch.Size([32, 80, 512])
        x = self.layernorm1(x + self.dropout(attn_output))  # 残差连接 torch.Size([32, 80, 512])
        ff_output = self.feed_forward(x)  # torch.Size([32, 80, 512])
        x = self.layernorm2(x + self.dropout(ff_output))  # 残差连接 torch.Size([32, 80, 512])
        return x


class TransformerFeatureAggregator(nn.Module):
    def __init__(self, len_feature, nhead=8, num_layers=2):
        super(TransformerFeatureAggregator, self).__init__()
        self.len_feature = len_feature  # 输入特征的长度
        self.conv_1x1 = nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=1)

        # 使用多个 Transformer Block 捕获时序特征
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model=512, nhead=nhead) for _ in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(in_channels=512, out_channels=len_feature, kernel_size=1)
    
    def forward(self, x):
        # x.shape = torch.Size([80, 512, 32])  (bs, D/4, T)  ### X'
        out = x.permute(2, 0, 1)  # 变换维度为 (T, bs, D/4) torch.Size([32, 80, 512])
        out = self.conv_1x1(out.permute(1, 2, 0))  # 先做 1x1 卷积压缩特征维度, (bs, D/4, T) torch.Size([80, 512, 32])
        out = out.permute(2, 0, 1)  # 变换回 (T, bs, D/4), torch.Size([32, 80, 512]) 

        for transformer in self.transformer_blocks:  # len(self.transformer_blocks) = 2
            out = transformer(out) # torch.Size([32, 80, 512]) (T, bs, D/4)
        breakpoint()
        out = out.permute(1, 2, 0)  # 转换维度 torch.Size([80, 512, 32]) (bs, D/4, T)
        out = self.conv_out(out)  # 通过1x1卷积恢复维度  torch.Size([80, 512, 32])
        return out



class FeatureAggregator(nn.Module):
    '''特征聚合模块，用于处理输入的特征. 包含多个卷积层和 Non-Local Block.
    目的是在 时间维度上 聚合特征 并捕捉时间依赖关系
    '''
    def __init__(self, len_feature):
        super(FeatureAggregator, self).__init__()
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
        '''
        new-b: 引入更多的卷积核和膨胀率
         现有模型只使用了三种膨胀率（dilation=1, 2, 4）进行卷积操作。可以进一步扩展这个多尺度卷积机制，比如引入不同大小的卷积核（如 kernel_size=1, 3, 5），或者再增加更多的膨胀率
        self.conv_4 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=5, dilation=8, padding=8) 通过引入更多的卷积核和膨胀率，可以捕捉到更多不同尺度的信息，提升模型的特征表达能力
        '''
        # self.conv_4 = nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=5, dilation=8, padding=8)
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=5, 
                        dilation=4, padding=8),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7),
        )

        # 使用 1x1 卷积对特征进行进一步压缩 (D --> D/4)
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1,
                      stride=1, padding=0, bias = False), 
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        
        # 将所有特征融合在一起
        self.conv_6 = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=2048, kernel_size=3, # new-a 1024; only new-b 2560; original 2048
                      stride=1, padding=1, bias=False), # should we keep the bias?
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            # nn.dropout(0.7)
        )

        '''new-c: 使用 Bottleneck 模块: class Bottleneck(nn.Module)
        当前模型使用了简单的残差连接，可以进一步优化残差连接的结构，例如使用 Bottleneck 残差块来减少参数量，同时提高特征表达能力。Bottleneck 可以有效减少计算量，同时保持残差网络的优势
        '''
        self.bottleneck = Bottleneck(in_channels=2048, out_channels=2048)

        self.non_local = NONLocalBlock1D(512, sub_sample=False, bn_layer=True)

        self.transformer_fa = TransformerFeatureAggregator(512, nhead=8, num_layers=2)
        
        '''new-a: 权重融合, 用可学习的权重对不同尺度的特征进行加权求和，增强模型的自适应性.
        当前多尺度卷积的结果通过 torch.cat 拼接在一起。可以进一步改进为 权重融合。
        self.weights = nn.Parameter(torch.ones(3))
        out = self.weights[0] * out1 + self.weights[1] * out2 + self.weights[2] * out3
        这样网络可以动态调整不同尺度特征的重要性，从而提高模型的灵活性
        '''
        self.weights = nn.Parameter(torch.ones(4))
        


    def forward(self, x):
            # x: (bs*n_crops, n_segments, feature_dim), torch.Size([80, 32, 2048]) (T, D)
            out = x.permute(0, 2, 1)  # 交换维度以适应卷积操作 torch.Size([80, 2048, 32])
            residual = out

            # 三层不同膨胀率的卷积 捕获不同范围的上下文信息
            out1 = self.conv_1(out)  # torch.Size([80, 512, 32])  (D/4, T)
            out2 = self.conv_2(out)  # torch.Size([80, 512, 32])
            out3 = self.conv_3(out)  # torch.Size([80, 512, 32])
            # out_d = torch.cat((out1, out2, out3), dim = 1)  # origin  [10, 1536, 37]

            # New-b: 引入更多的卷积核和膨胀率
            out4 = self.conv_4(out) # torch.Size([80, 512, 32])
            # out_d = torch.cat((out1, out2, out3, out4), dim = 1)  # [10, 2048, 37]

            # print(out1.shape, out2.shape, out3.shape, out_d0.shape)
            # print(out4.shape, out_d.shape) 

            # New-a: 权重融合
            out_d = self.weights[0] * out1 + self.weights[1] * out2 + self.weights[2] * out3 + self.weights[3] * out4  # torch.Size([80, 512, 32])
            # breakpoint()

            # 1x1卷积
            out = self.conv_5(out)  # 使用 1x1 卷积对特征进行进一步压缩 torch.Size([80, 512, 32])  (bs, D/4, T)
            # print('after conv_5:', out.shape) 

            # out = self.non_local(out)  # 引入 Non-Local Block 来捕获长距离依赖 torch.Size([80, 512, 32])

            out = self.transformer_fa(out)  # torch.Size([80, 512, 32])
            # print('after transformer_fa:', out.shape) 

            out = torch.cat((out_d, out), dim=1)  # only new-b: [80, 2560, 32]; new-a: torch.Size([80, 1024, 32])
            # print(out.shape) (D/2, T)

            out = self.conv_6(out)   # fuse all the features together, torch.Size([80, 2048, 32])  (D, T)

            # New-c
            out = self.bottleneck(out)  # torch.Size([80, 2048, 32])

            # 残差连接：输出与输入相加，形成残差连接，保留原始特征的部分信息
            out = out + residual   # torch.Size([80, 2048, 32])
            out = out.permute(0, 2, 1)  # torch.Size([80, 32, 2048])

            return out


class Model(nn.Module):
    def __init__(self, n_features, batch_size, args):
        super(Model, self).__init__()

        self.args = args
        self.batch_size = batch_size
        self.num_segments = 32  # T snippets each video
        self.k_abn = self.num_segments // 10  # k = 3 = 32 // 10 (topk: top 10%)
        self.k_nor = self.num_segments // 10

        self.FeatureAggregator = FeatureAggregator(len_feature=2048)  # 特征聚合模块, 通常用于聚合时序特征
        self.fc1 = nn.Linear(n_features, 512)  # 全连接层1，将输入特征数减少到512
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)  # 三层全连接层，将特征压缩 并生成最后的异常得分

        self.drop_out = nn.Dropout(0.7)  # 防止过拟合
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)  # 使用自定义的权重初始化方法初始化网络参数


    def forward(self, inputs):
        '''
        inputs.shape = (bs, n_crops, n_segments, feature_dim)  
        torch.Size([8, 10, 32, 2048]), or torch.Size([1, 10, 18, 2048])
        n_crops: each video, per n_segments step, has n_crops crop versions. ("look at" the input from different angles)
        '''
        # print('Calling Model.forward()...')
        # print(f"inputs: {inputs.shape}")
        # anchors = [bn.running_mean for bn in self.normal_head.bns]  # Mean Vector of BatchNorm
        # # anchors[0].shape = torch.Size([32]), anchors[1].shape = torch.Size([16]) 

        # inputs: torch.cat((ninput, ainput), 0) 
        k_abn = self.k_abn  # 3
        k_nor = self.k_nor

        out = inputs  # (bs, n_crops, n_segments, feature_dim)
        bs, ncrops, t, f = out.size()  # torch.Size([8, 10, 32, 2048])
        # breakpoint()
        out = out.view(-1, t, f)  # (bs*n_crops, n_segments, feature_dim) torch.Size([80, 32, 2048])
        out = self.FeatureAggregator(out)  # 聚合特征，输出大小保持不变 torch.Size([80, 32, 2048])
        out = self.drop_out(out)

        features = out  # (bs*n_crops, n_segments, feature_dim)  torch.Size([80, 32, 2048])
        normal_features = features[0:self.batch_size*10]  # 前面 n_crops*bs 个为正常视频特征 torch.Size([40, 32, 2048])
        abnormal_features = features[self.batch_size*10:]  # 后面的为异常视频特征 torch.Size([40, 32, 2048])
        # breakpoint()

        y_pred = self.relu(self.fc1(features))  # 通过全连接层1和ReLU激活 torch.Size([80, 32, 512])
        y_pred = self.drop_out(y_pred)
        y_pred = self.relu(self.fc2(y_pred))  # 通过全连接层2和ReLU激活 torch.Size([80, 32, 128])
        y_pred = self.drop_out(y_pred)
        y_pred = self.sigmoid(self.fc3(y_pred))  # 通过全连接层3和Sigmoid激活，得到y_pred # torch.Size([80, 32, 1])
        y_pred = y_pred.view(bs, ncrops, -1).mean(1)  # 平均多个crop的结果  torch.Size([8, 32])
        y_pred = y_pred.unsqueeze(dim=2)  # 在最后一维增加一个维度  torch.Size([8, 32, 1])

        normal_y_pred = y_pred[0:self.batch_size]  # torch.Size([4, 32, 1])
        abnormal_y_pred = y_pred[self.batch_size:]  # torch.Size([4, 32, 1])

        # in Eq.2, ||x_t||2
        feat_magnitudes = torch.norm(features, p=2, dim=2)  # 计算特征的 L2 范数, 衡量每个特征的大小。对于每个特征向量（每个时间步的特征），计算其 L2 范数 --> (bs*n_crops, n_segments)  torch.Size([80, 32])
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)  # torch.Size([8, 32])
        # 对每个视频的多个 crop 计算特征大小的均值/方差，得到每个时间步上最终的特征大小
        # view(, -1): torch.Size([8, 10, 32])  (bs, n_crops, n_segments); 
        # mean(1): 对 n_crops 维度的均值计算，合并所有 crop 的特征大小 (bs, n_segments)
        

        # New-var: 根据特征的方差来衡量多个 crop 之间的特征波动程度
        # feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).var(1) 

        nfea_magnitudes = feat_magnitudes[0:self.batch_size]  # normal feature magnitudes, torch.Size([4, 32])
        afea_magnitudes = feat_magnitudes[self.batch_size:]   # abnormal feature magnitudes, torch.Size([4, 32])
        n_size = nfea_magnitudes.shape[0]  # 4

        if nfea_magnitudes.shape[0] == 1:  # if inference, batch_size = 1 (initialize abnormal variables)
            afea_magnitudes = nfea_magnitudes
            abnormal_y_pred = normal_y_pred
            abnormal_features = normal_features

        ##########################################################################
        #######  Process Abnormal videos -> select top3 feature magnitude  #######
        ##########################################################################
        select_idx = torch.ones_like(nfea_magnitudes)  # torch.Size([4, 32])
        select_idx = self.drop_out(select_idx)  # torch.Size([4, 32])
        afea_magnitudes_drop = afea_magnitudes * select_idx  # dropout afea_magnitudes, torch.Size([4, 32])

        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]  # 从特征中选择Top-k的片段 torch.Size([4, 3])
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])  # torch.Size([4, 3, 2048])

        abnormal_features = abnormal_features.view(n_size, ncrops, t, f)  # torch.Size([40, 32, 2048]) -> torch.Size([4, 10, 32, 2048])
        abnormal_features = abnormal_features.permute(1, 0, 2, 3)  # torch.Size([10, 4, 32, 2048])

        total_select_abn_feature = torch.zeros(0, device=inputs.device)
        for abnormal_feature in abnormal_features:
            # abnormal_feature: torch.Size([4, 32, 2048])
            # 根据 idx_abn_feat，从 abnormal_feature 中选出 top-k abnormal feature
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat) # top 3 instances in abnormal bag, torch.Size([4, 3, 2048])
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))
        
        # total_select_abn_feature: torch.Size([40, 3, 2048])
        # breakpoint()
        idx_abn_y_pred = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_y_pred.shape[2]])  # torch.Size([4, 3, 1])

        # top 3 y_pred in abnormal bag based on the top-3 magnitude
        y_pred_abnormal = torch.mean(torch.gather(abnormal_y_pred, 1, idx_abn_y_pred), dim=1) # 计算top 3 y_pred的均值 torch.Size([4, 1])
        # y_pred_abnormal = torch.var(torch.gather(abnormal_y_pred, 1, idx_abn_y_pred), dim=1)  # 计算top 3 y_pred的方差

        ######################################################################
        ####### Process Normal videos -> select top3 feature magnitude #######
        ######################################################################
        select_idx_normal = torch.ones_like(nfea_magnitudes)  # torch.Size([4, 32])
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal  # torch.Size([4, 32])
        idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]  # 从特征中选择Top-k的片段 torch.Size([4, 3])
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])  # torch.Size([4, 3, 2048])

        normal_features = normal_features.view(n_size, ncrops, t, f)  # torch.Size([40, 32, 2048]) -> torch.Size([4, 10, 32, 2048])
        normal_features = normal_features.permute(1, 0, 2, 3)  # torch.Size([10, 4, 32, 2048])

        total_select_nor_feature = torch.zeros(0, device=inputs.device)
        for nor_fea in normal_features:
            # nor_fea: torch.Size([4, 32, 2048])
            # 根据选出的index (idx_normal_feat), 将特征从特征张量 (normal_features) 中选出来 
            feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)  # top 3 features magnitude in normal bag (hard negative), torch.Size([4, 3, 2048])
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))
        
        # total_select_nor_feature: torch.Size([40, 3, 2048])
        # breakpoint()
        idx_normal_y_pred = idx_normal.unsqueeze(2).expand([-1, -1, normal_y_pred.shape[2]])  # torch.Size([4, 3, 1])

        # top 3 y_pred in normal bag
        y_pred_normal = torch.mean(torch.gather(normal_y_pred, 1, idx_normal_y_pred), dim=1) # 计算top 3 y_pred的均值 torch.Size([4, 1])
        # y_pred_normal = torch.var(torch.gather(normal_y_pred, 1, idx_normal_y_pred), dim=1)  # 计算top 3 y_pred的方差 

        ################## Final Selected Features #########################
        feat_select_abn = total_select_abn_feature  # torch.Size([40, 3, 2048])
        feat_select_normal = total_select_nor_feature  # torch.Size([40, 3, 2048])

        # return y_pred_abnormal, y_pred_normal, feat_select_abn, feat_select_normal, feat_select_abn, feat_select_abn, y_pred, feat_select_abn, feat_select_abn, feat_magnitudes

        # [4, 1], [4, 1], [40, 3, 2048], [40, 3, 2048], [8, 32, 1], [8, 32]
        return y_pred_abnormal, y_pred_normal, feat_select_abn, feat_select_normal, y_pred, feat_magnitudes