import argparse

parser = argparse.ArgumentParser(description='RTFM')
parser.add_argument('--feat-extractor', default='i3d', choices=['i3d', 'c3d'])
parser.add_argument('--feature-size', type=int, default=2048, help='size of feature (default: 2048)')
parser.add_argument('--modality', default='RGB', help='the type of the input, RGB,AUDIO, or MIX')
# parser.add_argument('--rgb-list', default='list/shanghai-i3d-train-10crop.list', help='list of rgb features ')
# parser.add_argument('--test-rgb-list', default='list/shanghai-i3d-test-10crop.list', help='list of test rgb features ')
parser.add_argument('--rgb-list', default='list/DA-i3d-train-10crop.list', help='list of rgb features ')
parser.add_argument('--test-rgb-list', default='list/DA-i3d-test-10crop.list', help='list of test rgb features ')
parser.add_argument('--gt', default='list/gt-sh.npy', help='file of ground truth ')
parser.add_argument('--gpus', default=1, type=int, choices=[0], help='gpus')
parser.add_argument('--lr', type=str, default='[0.001]*15000', help='learning rates for steps(list form)')
parser.add_argument('--batch-size', type=int, default=4, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=4, help='number of workers in dataloader')
parser.add_argument('--model-name', default='rtfm', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--num-classes', type=int, default=1, help='number of class')
# parser.add_argument('--dataset', default='shanghai', help='dataset to train on (default: )')
parser.add_argument('--dataset', default='drone_anomaly', help='dataset to train on (default: )')
parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting (default: 10)')
parser.add_argument('--max-epoch', type=int, default=5000, help='maximum iteration to train (default: 100)')
# parser.add_argument('--fps', type=int, default=30, help='frame per second')
parser.add_argument('--run-name', type=str, default='this-run', help='run name')

# anchors
parser.add_argument('--ratios', type=int, nargs='+', default = [16, 32])
parser.add_argument('--kernel_sizes', type=int, nargs='+', default = [1, 1, 1])

parser.add_argument('--scene', type=str, default='all', choices = ['all', 'Bike_Roundabout', 'Crossroads', 'Farmland_Inspection', 'Highway', 'Railway_Inspection', 'Solar_Panel_Inspection', 'Vehicle_Roundabout'])
parser.add_argument('--smooth', type=bool, default=True, help = 'for visualization')
parser.add_argument('--window_size', type=int, default=5, help = 'for visualization')

# loss terms
parser.add_argument('--lambda1', type=int, default=1, help = 'weight of loss_cls')
parser.add_argument('--lambda2', type=int, default=0.5, help = 'weight of loss_dd')
parser.add_argument('--lambda3', type=int, default=0.1, help = 'weight of loss_smooth')
parser.add_argument('--lambda4', type=int, default=0.1, help = 'weight of loss_sparse')
parser.add_argument('--alpha', type=int, default=1, help = 'weight of loss_mean')
parser.add_argument('--beta', type=int, default=1, help = 'weight of loss_var')
parser.add_argument('--margin', type=int, default=3, help = 'margin of loss_mean')







