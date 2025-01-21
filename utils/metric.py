from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np

def get_metrics(y_true, y_pred):
    decimal_place = 4

    # 计算 auc（ROC曲线下面积）
    try:
        auc = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        auc = -1

    # 计算 spauc（特定假阳性率下的AUC）
    try:
        spauc = roc_auc_score(y_true, y_pred, average='macro', max_fpr=0.1)
    except ValueError:
        spauc = -1

    # 概率值转化为二元类别
    y_pred = np.around(np.array(y_pred)).astype(int)
    # y_pred = (np.array(y_pred) >= 0.5).astype(int)

    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)

    # 计算精确率
    precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    try:
        precision_real, precision_fake = precision_score(y_true, y_pred, average=None, zero_division=0)
    except ValueError:
        precision_real, precision_fake = -1, -1

    # 计算召回率，以及每个类别的召回率
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    try:
        recall_real, recall_fake = recall_score(y_true, y_pred, average=None, zero_division=0)
    except ValueError:
        recall_real, recall_fake = -1, -1

    # 计算F1分数，以及每个类别的F1分数
    f1_score_ = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    try:
        # average=None设置为None,计算每个类别的f1_score
        f1_real, f1_fake = f1_score(y_true, y_pred, average=None, zero_division=0)
    except ValueError:
        f1_real, f1_fake = -1, -1



    all_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'precision_real': precision_real,
        'precision_fake': precision_fake,
        'recall': recall,
        'recall_real': recall_real,
        'recall_fake': recall_fake,
        'f1_score': f1_score_,
        'f1_real': f1_real,
        'f1_fake': f1_fake,
        'auc': auc,
        'spauc': spauc
    }

    for key, value in all_metrics.items():
        all_metrics[key] = round(value, decimal_place)

    return all_metrics
