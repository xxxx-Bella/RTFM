import numpy as np
from sklearn.metrics import auc, f1_score

run_name = 'x-scene-cross'

# 加载 precision 和 recall 的数据
precision = np.load(f'log/run-{run_name}/precision.npy')
recall = np.load(f'log/run-{run_name}/recall.npy')
threshold = np.load(f'log/run-{run_name}/threshold.npy')
fpr = np.load(f'log/run-{run_name}/fpr.npy')
tpr = np.load(f'log/run-{run_name}/tpr.npy')

max_precision = precision[np.argmax(threshold)]
max_recall = recall[np.argmax(threshold)]

print(f'Precision: {max_precision}')
print(f'Recall: {max_recall}')

breakpoint()

# 遍历不同阈值计算 f1 score
best_f1 = 0
best_precision = 0
best_recall = 0

for i, th in enumerate(threshold):
    current_precision = precision[i]
    current_recall = recall[i]
    f1 = 2 * (current_precision * current_recall) / (current_precision + current_recall)
    
    if f1 > best_f1:
        best_f1 = f1
        best_precision = current_precision
        best_recall = current_recall

print(f"Best Precision: {best_precision}")
print(f"Best Recall: {best_recall}")
print(f"Best F1 Score: {best_f1}")


