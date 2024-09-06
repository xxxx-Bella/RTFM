import numpy as np
import os
import glob
from scipy.io import loadmat
from os import walk

'''
FOR Shanghai Tech. 
为 Shanghai Tech 数据集生成 ground truth (GT) 标签. 
根据视频的编号将视频分为“正常”和“异常”类别，分别为每个视频的每一帧生成标签.
正常视频的标签全为 0, 异常视频的标签根据 ground truth 文件生成.
'''
# root_path = "/home/yu/yu_ssd/SH_Test_center_crop_i3d/"
# dirs = os.listdir(root_path)

rgb_list_file ='shanghai-i3d-test.list'
file_list = list(open(rgb_list_file)) # 打开 rgb_list_file 文件并读取所有行, 每一行是一个 I3D 特征文件的路径或名称

temporal_root = 'test_frame_mask/' # ground truth 标签文件（frame-level label）的根目录路径
# mat_name_list = os.listdir(temporal_root)
gt_files = os.listdir(temporal_root) # 获取 temporal_root 目录下的所有文件名，存储在 gt_files 列表中

num_frame = 0
gt = []
index = 0
total = 0
abnormal_count =0

# 将每个 video 的 label 合成一个 gt
for  file in file_list:
    # load each npy (I3D feature) in rgb_list_file
    features = np.load(file.strip('\n'), allow_pickle=True)

    # features = [t.cpu().detach().numpy() for t in features]
    features = np.array(features, dtype=np.float32)
    features = np.squeeze(features, axis=1)  # squeeze移除不必要的单维度 axis=1 

    num_frame = features.shape[0] * 16  # 计算该视频的帧数 (视频的片段数量 * 每个片段包含16帧)

    count = 0
    # normal video: gt is (0*num_frame)
    if index > 43:
        print('normal video' + str(file))
        for i in range(0, num_frame):
            gt.append(0)
            count += 1
    
    # abnormal video: gt is ground_annotation (file)
    else:
        print('abnormal video' + str(file)) # file name
        gt_file = file.split('_i3d.npy')[0] + '.npy' # 将 '_i3d.npy' 切掉，保留前半部分，拼接.npy。如 video1_i3d.npy --> video1.npy
        gt_file = gt_file.split('/')[-1] # 将路径按照目录分割，取最后一级文件名
        # 检查 ground truth 文件是否存在
        if not os.path.isfile(os.path.join(temporal_root, gt_file)):
            print('no such file')
            exit(1)
        abnormal_count += 1 # 统计异常视频的数量
        
        ground_annotation = np.load(os.path.join(temporal_root, gt_file)) # frame-level label
        ground_annotation = list(ground_annotation)
        # 如果 ground truth 数据的长度（帧数）小于视频帧数 num_frame
        if len(ground_annotation) < num_frame:
            last_frame_label = ground_annotation[-1] # 最后一个标签值
            for i in range(len(ground_annotation), num_frame): # 用 last_frame_label 填充
                ground_annotation.append(last_frame_label) 

        # 再次检查帧数是否一致
        if len(ground_annotation)!= num_frame:
            print("wrong frame number")
            exit(1)
        
        count += len(ground_annotation)
        gt.extend(ground_annotation)

    index = index + 1
    total += count # 统计该视频的 ground truth 标签数

print(abnormal_count)

# # supplement by me
# gt_array = np.array(gt)
# np.save('gt-sh.npy', gt_array)
# # loaded_gt = np.load('gt-sh.npy')






