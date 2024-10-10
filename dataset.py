import torch.utils.data as data
import numpy as np
from utils import process_feat
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.FloatTensor')


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False, scene='all'):
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset
        self.scene = args.scene

        if self.dataset == 'shanghai':
            if test_mode:
                self.rgb_list_file = 'list/shanghai-i3d-test-10crop.list'
            else:
                self.rgb_list_file = 'list/shanghai-i3d-train-10crop.list'
        elif self.dataset == 'ucf':
            if test_mode:
                self.rgb_list_file = 'list/ucf-i3d-test.list'
            else:
                self.rgb_list_file = 'list/ucf-i3d.list'
        elif self.dataset == 'drone_anomaly':
            if test_mode:
                self.rgb_list_file = 'list/DA-i3d-test.list'
            else:
                self.rgb_list_file = 'list/DA-i3d-train.list'
        

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list() # get train self.list
        self.num_frame = 0
        self.labels = None


    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:  # train mode 
            print("Loading training dateset...")

            if self.dataset == 'shanghai':
                if self.is_normal:
                    self.list = self.list[63:]
                    print('normal list for shanghai tech')
                    print(self.list)
                else:
                    self.list = self.list[:63]
                    print('abnormal list for shanghai tech')
                    print(self.list)

            elif self.dataset == 'ucf':
                if self.is_normal:
                    self.list = self.list[810:]
                    print('normal list for ucf')
                    print(self.list)
                else:
                    self.list = self.list[:810]
                    print('abnormal list for ucf')
                    print(self.list)
            
            elif self.dataset == 'drone_anomaly':
                if self.scene == 'all':
                    index_n = [i for i, item in enumerate(self.list) if 'label_0' in item]
                    index_a = [i for i, item in enumerate(self.list) if 'label_1' in item]
                else:
                    index_n = [i for i, item in enumerate(self.list) if ('label_0' in item and self.scene in item)]
                    index_a = [i for i, item in enumerate(self.list) if ('label_1' in item and self.scene in item)]
                
                if self.is_normal:
                    self.list = [self.list[i] for i in index_n]
                    print('normal list for DA:', len(self.list))
                    # print(self.list)
                else:
                    self.list = [self.list[i] for i in index_a]
                    print('abnormal list for DA:', len(self.list))
                    # print(self.list)
        
        else:  # test mode
            if self.dataset == 'drone_anomaly': 
                if self.scene == 'all':
                    index = [i for i, item in enumerate(self.list)]
                else:
                    index = [i for i, item in enumerate(self.list) if self.scene in item]
                self.list = [self.list[i] for i in index]
                print('test list for DA:', len(self.list))
                # print(self.list)


    def __getitem__(self, index):

        label = self.get_label()  # get video level label 0/1
        vid_name = self.list[index].strip('\n')
        features = np.load(vid_name, allow_pickle=True)  # each video
        features = np.array(features, dtype=np.float32)  # (37, 10, 2048), (18, 10, 2048)...
        # print(f'vid_name: {vid_name}. feature shape = {features.shape}')
        # breakpoint()

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features
        else:
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, 18, 2048]
            divided_features = []
            # breakpoint()
            for feature in features:
                # feature.shape = (18, 2048)
                feature = process_feat(feature, 32)   # divide a video into 32 segments (T=32)
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)  # (10, 32, 2048)
            # breakpoint()
            return divided_features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
