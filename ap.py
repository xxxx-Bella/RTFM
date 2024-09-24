import numpy as np
from sklearn.metrics import auc


run_name = 'x-var-loss9'
# 加载 precision 和 recall 的数据
precision = np.load(f'log/run-{run_name}/precision.npy')
recall = np.load(f'log/run-{run_name}/recall.npy')

# 计算 PR 曲线下的面积（AP）# pr_auc
ap = auc(recall, precision)  # AUC 用于计算曲线下面积
print('AP (Area under PR Curve):', ap)