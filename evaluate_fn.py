import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import collections
import json
from sklearn.model_selection import StratifiedKFold, train_test_split, ShuffleSplit
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, roc_curve, average_precision_score,
                            precision_recall_curve, precision_score, recall_score, f1_score, matthews_corrcoef, auc)
try:
    from joblib import dump, load
except ImportError:
    from sklearn.externals.joblib import dump, load

def predict_from_realtime_predictor(data, predictor, rows=500):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    preds = ''
    for array in split_array:
        preds = ','.join([preds, predictor.predict(array).decode('utf-8')])
    return np.fromstring(preds[1:], sep=',')

def predict_from_realtime_lgb_predictor(data, predictor, rows=500):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    preds = []
    for array in split_array:
        tmp_str = predictor.predict(array).decode('utf-8')
        preds_chunk = json.loads(tmp_str)['results']
        preds += preds_chunk
    return np.array(preds)
    
def get_nans_df(df, ratio_thres=0):
    """
    Return NaN statistics for each variable
    """
    num_rows = len(df)
    num_null = df.isnull().sum()
    ratio_null = num_null / num_rows
    unique_counts = df.apply(lambda x: len(x.value_counts()), axis=0)
    nans_df = pd.concat([num_null, ratio_null, unique_counts, df.dtypes], axis=1)
    nans_df.columns = ['nan_freq', 'nan_ratio', 'cardinality', 'type']
    nans_df = nans_df.sort_values(['nan_ratio'], ascending=False)\
            .query('nan_ratio>{}'.format(ratio_thres))
    nans_df = nans_df.query('nan_ratio > 0')
    return nans_df

class oset(collections.Set):

    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, element):
        return element in self.d

    def __iter__(self):
        return iter(self.d)
    
def plot_roc_curve(y_true, y_score, is_single_fig=False):
    """
    Plot ROC Curve and show AUROC score
    """    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.title('AUROC = {:.4f}'.format(roc_auc))
    plt.plot(fpr, tpr, 'b')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel('TPR(True Positive Rate)')
    plt.xlabel('FPR(False Positive Rate)')
    if is_single_fig:
        plt.show()
    
def plot_pr_curve(y_true, y_score, is_single_fig=False):
    """
    Plot Precision Recall Curve and show AUPRC score
    """
    prec, rec, thresh = precision_recall_curve(y_true, y_score)
    avg_prec = average_precision_score(y_true, y_score)
    plt.title('AUPRC = {:.4f}'.format(avg_prec))
    plt.step(rec, prec, color='b', alpha=0.2, where='post')
    plt.fill_between(rec, prec, step='post', alpha=0.2, color='b')
    plt.plot(rec, prec, 'b')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    if is_single_fig:
        plt.show()

def plot_conf_mtx(y_true, y_score, thresh=0.5, class_labels=['0','1'], is_single_fig=False):
    """
    Plot Confusion matrix
    """    
    y_pred = np.where(y_score >= thresh, 1, 0)
    print("confusion matrix (cutoff={})".format(thresh))
    print(classification_report(y_true, y_pred, target_names=class_labels))
    conf_mtx = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_mtx, xticklabels=class_labels, yticklabels=class_labels, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    if is_single_fig:
        plt.show()

def prob_barplot(y_score, bins=np.arange(0.0, 1.11, 0.1), right=False, filename=None, figsize=(10,4), is_single_fig=False):
    """
    Plot barplot by binning predicted scores ranging from 0 to 1
    """    
    c = pd.cut(y_score, bins, right=right)
    counts = c.value_counts()
    percents = 100. * counts / len(c)
    percents.plot.bar(rot=0, figsize=figsize)
    plt.title('Histogram of score')
    print(percents)
    if filename is not None:
        plt.savefig('{}.png'.format(filename))   
    if is_single_fig:
        plt.show()
    
def evaluate_prediction(y_true, y_score, thresh=0.5):
    """
    All-in-one function for evaluation. 
    """    
    plt.figure(figsize=(14,4))
    plt.subplot(1,3,1)
    plot_roc_curve(y_true, y_score)
    plt.subplot(1,3,2)    
    plot_pr_curve(y_true, y_score)
    plt.subplot(1,3,3)    
    plot_conf_mtx(y_true, y_score, thresh) 
    plt.show()

def get_score_df(y_true, y_score, start_score=0.0, end_score=0.7, cutoff_interval=0.05):
    """
    Get a dataframe contains general metrics
    """    
    import warnings
    warnings.filterwarnings("ignore")
    score = []
    
    for cutoff in np.arange(start_score, end_score+0.01, cutoff_interval)[1:]:
        y_pred = np.where(y_score >= cutoff, 1, 0)
        conf_mat = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = conf_mat[0,0], conf_mat[0,1], conf_mat[1,0], conf_mat[1,1]
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        if precision !=0 and recall !=0 :
            f1 = f1_score(y_true, y_pred)
        else:
            f1 = 0     
        mcc = matthews_corrcoef(y_true, y_pred)
        score.append([cutoff, tp, fp, tn, fn, precision, recall, f1, mcc])
        
    score_df = pd.DataFrame(score, columns = ['Cutoff', 'TP', 'FP', 'TN' ,'FN', 'Precision', 'Recall', 'F1', 'MCC'])
    return score_df
