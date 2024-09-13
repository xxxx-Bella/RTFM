import numpy as np
import os
import glob
from scipy.io import loadmat
from os import walk

'''
For Drone Anomaly. 
为 Drone Anomaly 数据测试集生成 ground truth (GT) 标签. 
根据视频的文件名将视频分为“正常”和“异常”类别，分别为每个视频的每一帧生成标签.
正常视频的标签全为 0, 异常视频的标签根据 ground truth 文件生成.
'''

rgb_list_file ='list/DA-i3d-test.list'
file_list = [file.strip() for file in open(rgb_list_file)]  # 每一行是一个 I3D 特征文件的路径, 去除换行符
# gt（frame-level label）root path (for load abnormal gt)
gt_root = '/home/featurize/work/yuxin/WVAD/I3D/output/drone_anomaly/'  

# all_files = os.listdir(gt_root)  # 获取 gt_root 目录下的所有文件名
# gt_files = [] # all anomaly_gt file in test
# # 遍历 gt_root 目录下的文件
# for file_name in all_files:
#     if '_gt' in file_name:
#         # get the part before '_gt' 
#         prefix = file_name.split('_gt')[0]
#         # check if prefix in file_list
#         if any(prefix in file for file in file_list):
#             gt_files.append(file_name)

# print("test gt files: ", gt_files)
# print("-----------------")

num_frame = 0
gt = []
total = 0
abnormal_count =0
wrong_num = []
wrong_num_path = f'list/wrong_num.list'

# 将每个 video 的 label 合成一个 gt
for  file in file_list:
    # load each npy (I3D feature) in rgb_list_file
    features = np.load(file, allow_pickle=True)

    # features = [t.cpu().detach().numpy() for t in features]
    features = np.array(features, dtype=np.float32)
    print(f"features.shape: {features.shape}")
    if features.shape[1] == 1:
        features = np.squeeze(features, axis=1)
    # else:
    #     print(f"Skipping squeeze as axis 1 has size {features.shape[1]}")

    num_frame = features.shape[0] * 16  # 计算该视频的帧数 (视频的片段数量 * i3d_frequency)
    print(f'num_frame:{num_frame}')
    count = 0

    # normal video: gt is (0*num_frame)
    if 'label_0' in file:
        print('normal video:', str(file))
        for i in range(0, num_frame):
            gt.append(0)
            count += 1
    
    # abnormal video: load '_gt.npy' file
    else:
        print('ABnormal video:', str(file))  # file name
        gt_file = file.split('.npy')[0] + '_gt.npy' # xxx_label_1_gt.npy
        gt_file = gt_file.split('/')[-1]  # 将路径按照目录分割，取最后一级文件名
        # 检查 ground truth 文件是否存在
        if not os.path.isfile(os.path.join(gt_root, gt_file)):
            print('no such file')
            exit(1)
        abnormal_count += 1  # 统计异常视频的数量
        
        ground_annotation = np.load(os.path.join(gt_root, gt_file)) # frame-level label
        ground_annotation = list(ground_annotation)
        
        # 如果 ground truth 数据的长度（帧数）小于视频帧数 num_frame
        if len(ground_annotation) < num_frame:
            last_frame_label = ground_annotation[-1] # 最后一个标签值
            for i in range(len(ground_annotation), num_frame): # 用 last_frame_label 填充
                ground_annotation.append(last_frame_label) 

        # 再次检查帧数是否一致
        if len(ground_annotation)!= num_frame:
            # print("wrong frame number")
            # breakpoint()
            one = {str(file): [len(ground_annotation), num_frame]}
            wrong_num.append(one)
            
            ground_annotation = ground_annotation[:num_frame]
            # exit(1)
        
        count += len(ground_annotation)
        gt.extend(ground_annotation)

    total += count # 统计所有test视频的 ground truth 标签数

print()
print('all gt in DA-test:', total)
print('abnormal video in DA-test:', abnormal_count)
with open(wrong_num_path, 'w') as f:
    for one in wrong_num:
        f.write(str(one) + '\n')


gt_array = np.array(gt)
np.save('list/gt-da.npy', gt_array)  # test frame-level label
# # loaded_gt = np.load('gt-sh.npy')
