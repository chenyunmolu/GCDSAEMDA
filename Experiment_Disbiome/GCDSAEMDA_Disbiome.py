import argparse
import os
import random
import timeit
import warnings

import sklearn.metrics
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    precision_recall_curve, matthews_corrcoef
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from DASE_Keras import get_negative_sample_by_KMeans_and_cosine_distances, deep_sparse_auto_encoder, \
    get_negative_sample_by_KMeans, get_negative_sample_by_MiniBatchKMeans_and_cosine_distances, kfold_by_CV, \
    draw_ROC_curve, draw_PR_curve, data_toExcel, get_negative_sample_by_randomSample

start_time = timeit.default_timer()
warnings.filterwarnings("ignore")
# 设置CV的参数
parser = argparse.ArgumentParser(description='GCDSAEMDA')
parser.add_argument("--cv", default=2, type=int, choices=[1, 2, 3])
args = parser.parse_args()
print("dataset: Disbiome | cv:", args.cv)

seed = 36
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# 读取微生物和疾病的相似性融合矩阵，以及微生物和疾病关联矩阵
print("-----------------------------------------读取微生物和疾病的相似性融合矩阵，以及微生物和疾病关联矩阵-----------------------------------------")
MD_association_matrix = pd.read_csv('../Dataset/Disbiome/mircobe_disease_association_matrix.csv', index_col=0)
microbe_similarity_fusion_matrix = pd.read_csv('../Dataset/Disbiome/microbe_similarity_fusion_matrix.csv', index_col=0)
disease_similarity_fusion_matrix = pd.read_csv('../Dataset/Disbiome/disease_similarity_fusion_matrix.csv', index_col=0)

MD = np.array(MD_association_matrix)
MM = np.array(microbe_similarity_fusion_matrix)
DD = np.array(disease_similarity_fusion_matrix)
print(
    "-----------------------------------------读取GAT_CNN处理过的特征矩阵cnn_outputs_dataframe-----------------------------------------")
cnn_outputs = pd.read_csv("cnn_outputs_dataframe.csv", index_col=0)
cnn_outputs = np.array(cnn_outputs)
# 根据cnn_outputs生成复杂的特征向量
mic_nums = MD.shape[0]
dis_nums = MD.shape[1]
features_embedding_mic = cnn_outputs[0:mic_nums, :]

features_embedding_dis = cnn_outputs[mic_nums:cnn_outputs.shape[0], :]

print("-----------------------------------------根据特征矩阵生成正样本集-----------------------------------------")
positive_index_tuple = np.where(MD == 1)
positive_index_list = list(zip(positive_index_tuple[0], positive_index_tuple[1]))
all_train_features_input = []
all_train_lable = []
for (r, d) in positive_index_list:
    # 将正样本的特征乘积结果作为输入，添加到train_features_input列表中。
    all_train_features_input.append(np.hstack((features_embedding_mic[r, :], features_embedding_dis[d, :])))
    # 将标签值1添加到train_lable列表中
    all_train_lable.append(1)
# 接下来的代码块处理负样本
print("-----------------------------------------根据特征矩阵生成负样本集-----------------------------------------")
negative_index_tuple = np.where(MD == 0)
negative_index_list = list(zip(negative_index_tuple[0], negative_index_tuple[1]))
NEGATIVE_SAMPLE_CHA_ALL = []

for (r, d) in negative_index_list:
    NEGATIVE_SAMPLE_CHA_ALL.append(np.hstack((features_embedding_mic[r, :], features_embedding_dis[d, :])))

NEGATIVE_SAMPLE_CHA_ALL = np.array(NEGATIVE_SAMPLE_CHA_ALL)
# 使用KMeans聚类方法选择最佳的负样本
print("-----------------------------------------使用KMeans聚类方法选择最佳的负样本-----------------------------------------")
NEGATIVE_SAMPLE_CHA, NEGATIVE_SAMPLE_CHA_LABEL = get_negative_sample_by_MiniBatchKMeans_and_cosine_distances(
    NEGATIVE_SAMPLE_CHA_ALL,
    len(positive_index_list))
all_train_features_input = np.array(all_train_features_input)
all_features_input = np.vstack((all_train_features_input, NEGATIVE_SAMPLE_CHA))

all_label = np.array(all_train_lable).reshape(-1, 1)
all_label = np.vstack((all_label, NEGATIVE_SAMPLE_CHA_LABEL))
print("-----------------------------------------使用深度稀疏自编码器降维----------------------------------------")
CHA_data = deep_sparse_auto_encoder(all_features_input)
row = 280
col = 62
cv = args.cv
train_index_all, test_index_all = kfold_by_CV(CHA_data, 5, row, col, cv)
# kfold = KFold(n_splits=5, shuffle=True, random_state=36)

all_auc = []
all_aupr = []
all_accuracy = []
all_precision = []
all_recall = []
all_f1 = []
all_mcc = []
# 用于绘制ROC、PR曲线的参数列表
FPR = []
TPR = []
PRECISION = []
RECALL = []
test_label_all = []
test_predict_prob_all = []
print("-----------------------------------------开始进行五折交叉验证----------------------------------------")
for i in range(5):
    train_features_input, train_label = CHA_data[train_index_all[i]], all_label[train_index_all[i]]
    test_features_input, test_label = CHA_data[test_index_all[i]], all_label[test_index_all[i]]

    scaler = StandardScaler()
    train_features_input = scaler.fit_transform(train_features_input)
    test_features_input = scaler.transform(test_features_input)

    lgbm = LGBMClassifier(num_leaves=31, learning_rate=0.1, max_depth=-1, random_state=36)
    # train_label.ravel() 将训练标签展平为一维数组
    lgbm.fit(train_features_input, train_label.ravel())
    test_predict = lgbm.predict(test_features_input)
    test_predict_prob = lgbm.predict_proba(test_features_input)[:, 1]
    # 使用test_labels_predict
    accuracy = accuracy_score(test_label, test_predict)
    precision = precision_score(test_label, test_predict, average='macro')
    recall = recall_score(test_label, test_predict, average='macro')
    f1 = f1_score(test_label, test_predict, average='macro')
    mcc = matthews_corrcoef(test_label, test_predict)
    # test_labels_predict_positive_proba
    auc = roc_auc_score(test_label, test_predict_prob)
    fpr, tpr, thresholds1 = roc_curve(test_label, test_predict_prob, pos_label=1)
    pre, rec, thresholds2 = precision_recall_curve(test_label, test_predict_prob, pos_label=1)
    aupr = sklearn.metrics.auc(rec, pre)

    FPR.append(fpr)
    TPR.append(tpr)
    PRECISION.append(pre)
    RECALL.append(rec)
    test_label_all.append(test_label)
    test_predict_prob_all.append(test_predict_prob)

    print("auc:{}".format(auc))
    print("aupr:{}".format(aupr))
    print("accuracy:{}".format(accuracy))
    print("precision:{}".format(precision))
    print("recall:{}".format(recall))
    print("f1_score:{}".format(f1))
    print("mcc:{}".format(mcc))
    all_auc.append(auc)
    all_aupr.append(aupr)
    all_accuracy.append(accuracy)
    all_precision.append(precision)
    all_recall.append(recall)
    all_f1.append(f1)
    all_mcc.append(mcc)

mean_auc = np.around(np.mean(np.array(all_auc)), 4)
mean_aupr = np.around(np.mean(np.array(all_aupr)), 4)
mean_accuracy = np.around(np.mean(np.array(all_accuracy)), 4)
mean_precision = np.around(np.mean(np.array(all_precision)), 4)
mean_recall = np.around(np.mean(np.array(all_recall)), 4)
mean_f1 = np.around(np.mean(np.array(all_f1)), 4)
mean_mcc = np.around(np.mean(np.array(all_mcc)), 4)
# 计算标准差
std_auc = np.around(np.std(np.array(all_auc)), 4)
std_aupr = np.around(np.std(np.array(all_aupr)), 4)
std_accuracy = np.around(np.std(np.array(all_accuracy)), 4)
std_precision = np.around(np.std(np.array(all_precision)), 4)
std_recall = np.around(np.std(np.array(all_recall)), 4)
std_f1 = np.around(np.std(np.array(all_f1)), 4)
std_mcc = np.around(np.std(np.array(all_mcc)), 4)
print()
print("MEAN AUC:{} ± {}".format(mean_auc, std_auc))
print("MEAN AUPR:{} ± {}".format(mean_aupr, std_aupr))
print("MEAN ACCURACY:{} ± {}".format(mean_accuracy, std_accuracy))
print("MEAN PRECISION:{} ± {}".format(mean_precision, std_precision))
print("MEAN RECALL:{} ± {}".format(mean_recall, std_recall))
print("MEAN F1_SCORE:{} ± {}".format(mean_f1, std_f1))
print("MEAN MCC:{} ± {}".format(mean_mcc, std_mcc))
# 计算运行时间
end_time = timeit.default_timer()
print("Running time: %s Seconds" % (end_time - start_time))

draw_ROC_curve(FPR, TPR, cv)

draw_PR_curve(test_label_all, test_predict_prob_all, cv)
