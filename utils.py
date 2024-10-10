# import visdom
import numpy as np
import torch
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def print_training_info(args, all=False):
    print('==================== Training Setting ====================')

    if all: print(args)
    else:
        try: print(f'Epoch: {args.max_epoch}')
        except: pass

        try: print(f'LR: {args.lr}')
        except: pass

        try: print(f'Batch size: {args.batch_size}')
        except: pass

        try: print(f'GPU ID: {args.gpuid}')
        except: pass

    print('==========================================================')

class StdRedirect:
    def __init__(self, filename):
        self.stream = sys.stdout
        self.file = open(filename,'w')

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.file.write(data)
        self.file.flush()

    def flush(self):
        pass

    def __del__(self):
        self.file.close()


def process_feat(feat, length):
    '''对输入的特征 feat 进行重新采样，将其处理为固定长度. 
    length - 输出的目标长度
    feat: (37, 2048)
    '''
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    # 均匀划分特征的区间 (32, 2048)
    indexs = np.linspace(0, len(feat), length+1, dtype=int)  # (start, stop, num)=(0, 37, 33)  # array([ 0,  1,  2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 37])  将0~10个划分为33
    # breakpoint()
    for i in range(length):  # 32
        if indexs[i]!=indexs[i+1]:
            new_feat[i,:] = np.mean(feat[indexs[i]:indexs[i+1],:], 0)  # 计算每个区间内的均值
        else:
            new_feat[i,:] = feat[indexs[i],:]
    return new_feat


def minmax_norm(act_map, min_val=None, max_val=None):
    '''最小最大归一化. 将输入的激活图 act_map 归一化到 [0, 1] 区间
    '''
    if min_val is None or max_val is None:
        relu = torch.nn.ReLU()
        max_val = relu(torch.max(act_map, dim=0)[0])
        min_val = relu(torch.min(act_map, dim=0)[0])

    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta  # 标准化

    ret[ret > 1] = 1
    ret[ret < 0] = 0

    return ret


def modelsize(model, input, type_size=4):
    '''计算模型的参数 和中间变量的大小'''
    # check GPU utilisation
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums


    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))


# def save_best_record(test_info, file_path):
#     fo = open(file_path, "w")
#     fo.write("epoch: {}\n".format(test_info["epoch"][-1]))
#     fo.write(str(test_info["test_AUC"][-1]))
#     fo.close()

def save_best_record(content, file_path, type):
    with open(file_path, "a") as fo:  # 使用 "a" 模式打开文件，表示追加写入
        if type == 'auc':
            fo.write(f"epoch: {content['epoch'][-1]}\n")  # 写入最新 epoch
            fo.write(f"test_AUC: {content['test_AUC'][-1]}\n")  # 写入最新 AUC
        elif type == 'ap':
            fo.write(f"epoch: {content['epoch'][-1]}\n") 
            fo.write(f"test_AP: {content['test_AP'][-1]}\n")  
        elif type == 'f1':
            fo.write(f"epoch: {content['epoch'][-1]}\n")
            fo.write(f"test_F1: {content['test_F1'][-1]}\n")  # 写入最新 f1-info     
        elif type == 'args':
            fo.write(f"==================== Training Setting ====================\n: {content}\n==========================================================") 
        elif type == 'time':
            fo.write(f"current_time: {content}\n") 

        fo.write("\n")  # 换行，方便下次记录


# training
def draw_distribution(feat_n, feat_a, log_dir, epoch):
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
        output_path = os.path.join(log_dir, f'epoch{epoch}-dim{dim}.png')
        plt.savefig(output_path)
        plt.close()


def smooth_curve(pred, window_size=5):
    # 创建滑动平均卷积核
    window = np.ones(window_size) / window_size
    # 使用卷积计算平滑后的数据
    smooth_pred = np.convolve(pred, window, mode='same')
    return smooth_pred


# testing
def visulization(epoch, pred, log_dir, scene, smooth=False, window_size=5):
    frame_index = list(range(len(pred)))
    
    if smooth:
        pred = smooth_curve(pred, window_size)

    plt.figure(figsize=(10, 6))
    plt.plot(frame_index, pred, color='#ff7f0e', linewidth=2)
    
    # plt.title('Prediction Scores vs Frame Index', fontsize=16)
    plt.xlabel('Frame Index', fontsize=14)
    plt.ylabel('Prediction Score', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True)
    # plt.legend()
    output_path = os.path.join(log_dir, f'epoch{epoch}-{scene}-score.png')
    plt.savefig(output_path)
    plt.close()


# # visdom
# class Visualizer(object):
#     def __init__(self, env='default', **kwargs):
#         '''创建一个 Visdom 实例 self.vis，用于与 Visdom 服务器交互'''
#         self.vis = visdom.Visdom(env=env, **kwargs)
#         self.index = {}

#     def plot_lines(self, name, y, **kwargs):
#         '''
#         self.plot('loss', 1.00)
#         绘制折线图. name - chart name
#         '''
#         x = self.index.get(name, 0)
#         self.vis.line(Y=np.array([y]), X=np.array([x]),
#                       win=str(name),
#                       opts=dict(title=name),
#                       update=None if x == 0 else 'append', # update='append' 追加数据点
#                       **kwargs
#                       )
#         self.index[name] = x + 1
    
#     def disp_image(self, name, img):
#         '''显示图像. name 是窗口名称, img 是图像数据'''
#         self.vis.image(img=img, win=name, opts=dict(title=name))
    
#     def lines(self, name, line, X=None):
#         '''绘制折线图或多条线, 支持同时传递 X 和 Y 轴的数据'''
#         if X is None:
#             self.vis.line(Y=line, win=name)
#         else:
#             self.vis.line(X=X, Y=line, win=name)

#     def scatter(self, name, data):
#         '''绘制散点图'''
#         self.vis.scatter(X=data, win=name)