import numpy as np
import pandas as pd
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

# 基于HMDAD数据库实现CV1、CV2、CV3的AUC值，可视化为柱状图

randomSample_CV1 = pd.read_excel('../GCDSAEMDA_KMeans.xlsx', sheet_name='HMDAD_CV1_randomSample_AUC', header=None)
randomSample_cv1_auc = auc(randomSample_CV1[0], randomSample_CV1[1])
randomSample_CV2 = pd.read_excel('../GCDSAEMDA_KMeans.xlsx', sheet_name='HMDAD_CV2_randomSample_AUC', header=None)
randomSample_cv2_auc = auc(randomSample_CV2[0], randomSample_CV2[1])
randomSample_CV3 = pd.read_excel('../GCDSAEMDA_KMeans.xlsx', sheet_name='HMDAD_CV3_randomSample_AUC', header=None)
randomSample_cv3_auc = auc(randomSample_CV3[0], randomSample_CV3[1])

KMeans_CV1 = pd.read_excel('../GCDSAEMDA_KMeans.xlsx', sheet_name='HMDAD_CV1_KMeans_AUC', header=None)
KMeans_cv1_auc = auc(KMeans_CV1[0], KMeans_CV1[1])
KMeans_CV2 = pd.read_excel('../GCDSAEMDA_KMeans.xlsx', sheet_name='HMDAD_CV2_KMeans_AUC', header=None)
KMeans_cv2_auc = auc(KMeans_CV2[0], KMeans_CV2[1])
KMeans_CV3 = pd.read_excel('../GCDSAEMDA_KMeans.xlsx', sheet_name='HMDAD_CV3_KMeans_AUC', header=None)
KMeans_cv3_auc = auc(KMeans_CV3[0], KMeans_CV3[1])

cosineKMeans_CV1 = pd.read_excel('../GCDSAEMDA.xlsx', sheet_name='HMDAD_CV1_AUC', header=None)
cosineKMeans_cv1_auc = auc(cosineKMeans_CV1[0], cosineKMeans_CV1[1])
cosineKMeans_CV2 = pd.read_excel('../GCDSAEMDA.xlsx', sheet_name='HMDAD_CV2_AUC', header=None)
cosineKMeans_cv2_auc = auc(cosineKMeans_CV2[0], cosineKMeans_CV2[1])
cosineKMeans_CV3 = pd.read_excel('../GCDSAEMDA.xlsx', sheet_name='HMDAD_CV3_AUC', header=None)
cosineKMeans_cv3_auc = auc(cosineKMeans_CV3[0], cosineKMeans_CV3[1])
#
plt.figure(figsize=(10, 8))
# 示例数据
labels = ['CV1', 'CV2', 'CV3']
# randomSample = [0.9664, 0.9554, 0.9632]
# KMeans = [0.9565, 0.9203, 0.9512]
# cosineKMeans = [0.9873, 0.9886, 0.9900]
randomSample = [randomSample_cv1_auc, randomSample_cv2_auc, randomSample_cv3_auc]
KMeans = [KMeans_cv1_auc, KMeans_cv2_auc, KMeans_cv3_auc]
cosineKMeans = [cosineKMeans_cv1_auc, cosineKMeans_cv2_auc, cosineKMeans_cv3_auc]
print(randomSample)
# 柱的宽度
bar_width = 0.3

# x轴的位置
r1 = np.arange(len(labels))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
# 绘制柱状图
bars1 = plt.bar(r1, randomSample, color='#ffc089', width=bar_width, edgecolor='grey', label='random sample')
bars2 = plt.bar(r2, KMeans, color='#62a0ca', width=bar_width, edgecolor='grey', label='k-means')
bars3 = plt.bar(r3, cosineKMeans, color='#9ad19a', width=bar_width, edgecolor='grey', label='cosine k-means')


def add_value_labels(bars, fontsize='x-large'):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom',
                 fontsize=fontsize)


add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

plt.xticks(fontsize='xx-large')
plt.yticks(fontsize='xx-large')
plt.xlabel('HMDAD dataset', fontsize='xx-large')
plt.xticks([r + bar_width for r in range(len(labels))], labels)
plt.ylabel('AUC', fontsize='xx-large')

plt.legend(loc='upper center', framealpha=0, bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize='xx-large')
plt.savefig('./HMDAD_KMeans_AUC.png')
plt.show()

# 基于HMDAD数据库实现CV1、CV2、CV3的AUC值，可视化为柱状图

randomSample_CV1 = pd.read_excel('../GCDSAEMDA_KMeans.xlsx', sheet_name='HMDAD_CV1_randomSample_AUPR', header=None)
randomSample_cv1_aupr = auc(randomSample_CV1[0], randomSample_CV1[1])
randomSample_CV2 = pd.read_excel('../GCDSAEMDA_KMeans.xlsx', sheet_name='HMDAD_CV2_randomSample_AUPR', header=None)
randomSample_cv2_aupr = auc(randomSample_CV2[0], randomSample_CV2[1])
randomSample_CV3 = pd.read_excel('../GCDSAEMDA_KMeans.xlsx', sheet_name='HMDAD_CV3_randomSample_AUPR', header=None)
randomSample_cv3_aupr = auc(randomSample_CV3[0], randomSample_CV3[1])

KMeans_CV1 = pd.read_excel('../GCDSAEMDA_KMeans.xlsx', sheet_name='HMDAD_CV1_KMeans_AUPR', header=None)
KMeans_cv1_aupr = auc(KMeans_CV1[0], KMeans_CV1[1])
KMeans_CV2 = pd.read_excel('../GCDSAEMDA_KMeans.xlsx', sheet_name='HMDAD_CV2_KMeans_AUPR', header=None)
KMeans_cv2_aupr = auc(KMeans_CV2[0], KMeans_CV2[1])
KMeans_CV3 = pd.read_excel('../GCDSAEMDA_KMeans.xlsx', sheet_name='HMDAD_CV3_KMeans_AUPR', header=None)
KMeans_cv3_aupr = auc(KMeans_CV3[0], KMeans_CV3[1])

cosineKMeans_CV1 = pd.read_excel('../GCDSAEMDA.xlsx', sheet_name='HMDAD_CV1_AUPR', header=None)
cosineKMeans_cv1_aupr = auc(cosineKMeans_CV1[0], cosineKMeans_CV1[1])
cosineKMeans_CV2 = pd.read_excel('../GCDSAEMDA.xlsx', sheet_name='HMDAD_CV2_AUPR', header=None)
cosineKMeans_cv2_aupr = auc(cosineKMeans_CV2[0], cosineKMeans_CV2[1])
cosineKMeans_CV3 = pd.read_excel('../GCDSAEMDA.xlsx', sheet_name='HMDAD_CV3_AUPR', header=None)
cosineKMeans_cv3_aupr = auc(cosineKMeans_CV3[0], cosineKMeans_CV3[1])
#
plt.figure(figsize=(10, 8))
# 示例数据
labels = ['CV1', 'CV2', 'CV3']
randomSample = [randomSample_cv1_aupr, randomSample_cv2_aupr, randomSample_cv3_aupr]
KMeans = [KMeans_cv1_aupr, KMeans_cv2_aupr, KMeans_cv3_aupr]
cosineKMeans = [cosineKMeans_cv1_aupr, cosineKMeans_cv2_aupr, cosineKMeans_cv3_aupr]
print(randomSample)
# 柱的宽度
bar_width = 0.3

# x轴的位置
r1 = np.arange(len(labels))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# 绘制柱状图
bars1 = plt.bar(r1, randomSample, color='#ffc089', width=bar_width, edgecolor='grey', label='random sample')
bars2 = plt.bar(r2, KMeans, color='#62a0ca', width=bar_width, edgecolor='grey', label='k-means')
bars3 = plt.bar(r3, cosineKMeans, color='#9ad19a', width=bar_width, edgecolor='grey', label='cosine k-means')


def add_value_labels(bars, fontsize='x-large'):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom',
                 fontsize=fontsize)


add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

plt.xticks(fontsize='xx-large')
plt.yticks(fontsize='xx-large')
plt.xlabel('HMDAD dataset', fontsize='xx-large')
plt.xticks([r + bar_width for r in range(len(labels))], labels)
plt.ylabel('AUPR', fontsize='xx-large')

plt.legend(loc='upper center', framealpha=0, bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize='xx-large')
plt.savefig('./HMDAD_KMeans_AUPR.png')
plt.show()
