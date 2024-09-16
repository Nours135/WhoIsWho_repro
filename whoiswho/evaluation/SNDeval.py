from typing import List, Tuple
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, silhouette_score
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import os
from os.path import join
from tqdm import tqdm
from datetime import datetime

from whoiswho.dataset.data_process import read_pubs,read_raw_pubs
from whoiswho.utils import load_json, save_json


def evaluate(predict_result, ground_truth):
    if isinstance(predict_result, str):
        predict_result = load_json(predict_result)
    if isinstance(ground_truth, str):
        ground_truth = load_json(ground_truth)

    name_nums = 0
    result_list = []
    metrics_dict_l = []
    gt_author_wise_precision, gt_author_wise_recall, gt_author_wise_f1 = [], [], []
    author_count_distinction_l = []
    for name in predict_result:
        #Get clustering labels in predict_result
        predicted_pubs = dict()
        for idx,pids in enumerate(predict_result[name]):
            for pid in pids:
                predicted_pubs[pid] = idx
        # Get paper labels in ground_truth
        pubs = []
        ilabel = 0
        true_labels = []
        for aid in ground_truth[name]:
            pubs.extend(ground_truth[name][aid])
            true_labels.extend([ilabel] * len(ground_truth[name][aid]))
            ilabel += 1

        predict_labels = []
        for pid in pubs:
            predict_labels.append(predicted_pubs[pid])

        pairwise_precision, pairwise_recall, pairwise_f1 = pairwise_evaluate(true_labels, predict_labels)
        result_list.append((pairwise_precision,pairwise_recall,pairwise_f1))
        
        # run cluster_evaluate
        metrics = cluster_evaluate(true_labels, predict_labels)
        metrics_dict_l.append(metrics)
        
        # run gt_author_wise_eval
        precision, recall, f1, author_count_distinction = gt_author_wise_metrics(true_labels, predict_labels)
        gt_author_wise_precision += precision
        gt_author_wise_recall += recall
        gt_author_wise_f1 += f1
        author_count_distinction_l.append(author_count_distinction)
        name_nums += 1

    avg_pairwise_precision = sum([result[0] for result in result_list])/name_nums
    avg_pairwise_recall = sum([result[1] for result in result_list])/name_nums
    avg_pairwise_f1 = sum([result[2] for result in result_list])/name_nums
    
    print(f'Average Pairwise Precision: {avg_pairwise_precision:.3f}')
    print(f'Average Pairwise Recall: {avg_pairwise_recall:.3f}')
    print(f'Average Pairwise F1: {avg_pairwise_f1:.3f}')
    avg_metrics = dict()
    for k in metrics_dict_l[0].keys():
        avg_metrics[k] = sum([metrics[k] for metrics in metrics_dict_l])/name_nums
    
    for k, v in avg_metrics.items():
        print(f'Average {k}: {v:.3f}')
        
    # print gt author wise metirc
    print(f'Average GT Author Wise Precision: {sum(gt_author_wise_precision)/len(gt_author_wise_f1):.3f}')
    print(f'Average GT Author Wise Recall: {sum(gt_author_wise_recall)/len(gt_author_wise_f1):.3f}')
    print(f'Average GT Author Wise F1: {sum(gt_author_wise_f1)/len(gt_author_wise_f1):.3f}')
    print(f'Author Count Distinction (GT - Pred): {sum([abs(item) for item in author_count_distinction_l])/len(author_count_distinction_l)}')
    print(f'Author Count Distinction bias: {sum(author_count_distinction_l)/len(author_count_distinction_l)}')
    print('\n')
    return avg_pairwise_f1



def pairwise_evaluate(correct_labels: List[int], pred_labels: List[int]) -> Tuple[float, float, float]:
    '''
    Input: Lists of clustering labels for each paper
    Output: Pairwise precision, recall, and f1
    '''
    TP = 0.0  # Pairs Correctly Predicted To SameAuthor
    TP_FP = 0.0  # Total Pairs Predicted To SameAuthor
    TP_FN = 0.0  # Total Pairs To SameAuthor

    for i in range(len(correct_labels)):
        for j in range(i + 1, len(correct_labels)):
            if correct_labels[i] == correct_labels[j]:
                TP_FN += 1
            if pred_labels[i] == pred_labels[j]:
                TP_FP += 1
            if (correct_labels[i] == correct_labels[j]) and (pred_labels[i] == pred_labels[j]):
                TP += 1

    if TP == 0:
        pairwise_precision = 0
        pairwise_recall = 0
        pairwise_f1 = 0
    else:
        pairwise_precision = TP / TP_FP
        pairwise_recall = TP / TP_FN
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (pairwise_precision + pairwise_recall)

    return pairwise_precision, pairwise_recall, pairwise_f1

def cluster_evaluate(correct_labels: List[int], pred_labels: List[int]):
    '''
    Input: Lists of clustering labels for each paper
        clustering metrics
    '''
    # mutual information
    metrics = {}
    metrics['Normalized Mutual Information'] = normalized_mutual_info_score(correct_labels, pred_labels)
    metrics['Homogeneity'] = homogeneity_score(correct_labels, pred_labels)
    metrics['Completeness'] = completeness_score(correct_labels, pred_labels)
    return metrics

def gt_author_wise_metrics(correct_labels: List[int], pred_labels: List[int]) -> Tuple[List[float], List[float], List[float], int]:
    '''
    for each gt author
        find the most matched predicted author, and calculate the precision, recall and f1
        
    return: precision, recall, f1, auhtor_count_distinction
    '''
    y_true = np.array(correct_labels)
    y_pred = np.array(pred_labels)
    # step 1, for each gt author find the most matched predited author
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1]
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    y_true = y_true.astype(int)
    y_voted_labels = y_voted_labels.astype(int)
    avg = 'macro'
    avg = None
    precision = precision_score(y_true, y_voted_labels, average=avg, zero_division=0)  # 使用这个参数能避免报错，但是使得准确率错误偏高 zero_division=1
    recall = recall_score(y_true, y_voted_labels, average=avg)
    f1 = f1_score(y_true, y_voted_labels, average=avg) 
    
    # print(labels)
    # print(labels.shape)
    # print()
    # print(f'Precision: {precision}')
    # print(f'Recall: {recall}')
    # print(f'F1: {f1}')
    # print(f'len(precision): {len(precision)}')
    # raise Exception('stop here')
    
    gt_author_count = len(labels)
    predicted_author_count = len(np.unique(y_pred))
    return list(precision), list(recall), list(f1), gt_author_count - predicted_author_count
    

if __name__ == '__main__':
    predict = 'Input the path of result.valid.json'
    ground_truth = 'Input the path of sna_valid_ground_truth.json'
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    bond_root_dir = os.path.join(root_dir, 'bond')
    print(bond_root_dir)
    predict = os.path.join(bond_root_dir, './out/res.json')
    ground_truth = os.path.join(bond_root_dir, './dataset/data/src/train/train_author.json')
    
    evaluate(predict, ground_truth)

