import numpy as np
import pandas as pd
import sklearn.metrics
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

# 基于HMDAD数据库的CV1的ROC曲线绘制
plt.figure(figsize=(10, 8))

sheetName = 'HMDAD_CV1_AUC'
GCDSAEMDA = pd.read_excel('../GCDSAEMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(GCDSAEMDA[0], GCDSAEMDA[1])
plt.plot(GCDSAEMDA[0], GCDSAEMDA[1], color="#D81C38", lw=2,
         label='GCDSAEMDA(AUC=%0.4f)' % mean_auc)

DSAE_RF = pd.read_excel('../DSAE_RF.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(DSAE_RF[0], DSAE_RF[1])
plt.plot(DSAE_RF[0], DSAE_RF[1], color="#62a0ca", label='DSAE_RF(AUC=%0.4f)' % mean_auc)

SAELGMDA = pd.read_excel('../SAELGMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(SAELGMDA[0], SAELGMDA[1])
plt.plot(SAELGMDA[0], SAELGMDA[1], color='#ffc089', label='SAELGMDA(AUC=%0.4f)' % mean_auc)

MNNMDA = pd.read_excel('../MNNMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(MNNMDA[0], MNNMDA[1])
plt.plot(MNNMDA[0], MNNMDA[1], color='#9ad19a', label='MNNMDA(AUC=%0.4f)' % mean_auc)

GPUDMDA = pd.read_excel('../GPUDMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(GPUDMDA[0], GPUDMDA[1])
plt.plot(GPUDMDA[0], GPUDMDA[1], color='#9b7ebb', label='GPUDMDA(AUC=%0.4f)' % mean_auc)

# plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize='xx-large')
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize='xx-large')
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))
plt.gca().xaxis.set_major_locator(MultipleLocator(0.2))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
plt.tick_params(axis='both', which='major', direction='in', length=6)
plt.tick_params(axis='both', which='minor', direction='in', length=3)
plt.xlabel("False Positive Rate", fontsize='xx-large')
plt.ylabel("True Positive Rate", fontsize='xx-large')
plt.title("Receiver Operating Characteristic Curve", fontsize='xx-large')
# 将图例固定在右下
plt.legend(loc=4, framealpha=0, bbox_to_anchor=(1, 0), borderaxespad=1, fontsize='x-large')
plt.savefig('./HMDAD_CV1_AUC.png')
plt.show()

# 基于HMDAD数据库的CV1的PR曲线绘制
plt.figure(figsize=(10, 8))
sheetName = 'HMDAD_CV1_AUPR'
GCDSAEMDA = pd.read_excel('../GCDSAEMDA.xlsx', sheet_name=sheetName, header=None)
mean_aupr = sklearn.metrics.auc(GCDSAEMDA[0], GCDSAEMDA[1])
plt.plot(GCDSAEMDA[0], GCDSAEMDA[1], color="#D81C38", lw=2, label='GCDSAEMDA(AUPR=%0.4f)' % mean_aupr)

DSAE_RF = pd.read_excel('../DSAE_RF.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(DSAE_RF[0], DSAE_RF[1])
plt.plot(DSAE_RF[0], DSAE_RF[1], color="#62a0ca", label='DSAE_RF(AUPR=%0.4f)' % mean_auc)

SAELGMDA = pd.read_excel('../SAELGMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(SAELGMDA[0], SAELGMDA[1])
plt.plot(SAELGMDA[0], SAELGMDA[1], color='#ffc089', label='SAELGMDA(AUPR=%0.4f)' % mean_auc)

MNNMDA = pd.read_excel('../MNNMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(MNNMDA[0], MNNMDA[1])
plt.plot(MNNMDA[0], MNNMDA[1], color='#9ad19a', label='MNNMDA(AUPR=%0.4f)' % mean_auc)

GPUDMDA = pd.read_excel('../GPUDMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(GPUDMDA[0], GPUDMDA[1])
plt.plot(GPUDMDA[0], GPUDMDA[1], color='#9b7ebb', label='GPUDMDA(AUPR=%0.4f)' % mean_auc)

plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize='xx-large')
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize='xx-large')
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))
plt.gca().xaxis.set_major_locator(MultipleLocator(0.2))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
plt.tick_params(axis='both', which='major', direction='in', length=6)
plt.tick_params(axis='both', which='minor', direction='in', length=3)
plt.xlabel('Recall', fontsize='xx-large')
plt.ylabel('Precision', fontsize='xx-large')
plt.title('Precision-Recall Curve', fontsize='xx-large')
plt.legend(loc=3, framealpha=0, bbox_to_anchor=(0, 0), borderaxespad=1, fontsize='x-large')
plt.savefig('./HMDAD_CV1_AUPR.png')
plt.show()

# 原始配色
# plt.figure(figsize=(10, 8))
# sheetName = 'HMDAD_CV1_AUPR'
# GCDSAEMDA = pd.read_excel('../GCDSAEMDA.xlsx', sheet_name=sheetName, header=None)
# mean_aupr = sklearn.metrics.auc(GCDSAEMDA[0], GCDSAEMDA[1])
# plt.plot(GCDSAEMDA[0], GCDSAEMDA[1], color="#D81C38", lw=2, label='GCDSAEMDA(AUC=%0.4f)' % mean_aupr)
#
# DSAE_RF = pd.read_excel('../DSAE_RF.xlsx', sheet_name=sheetName, header=None)
# mean_auc = sklearn.metrics.auc(DSAE_RF[0], DSAE_RF[1])
# plt.plot(DSAE_RF[0], DSAE_RF[1], label='DSAE_RF(AUC=%0.4f)' % mean_auc)
#
# SAELGMDA = pd.read_excel('../SAELGMDA.xlsx', sheet_name=sheetName, header=None)
# mean_auc = sklearn.metrics.auc(SAELGMDA[0], SAELGMDA[1])
# plt.plot(SAELGMDA[0], SAELGMDA[1], label='SAELGMDA(AUC=%0.4f)' % mean_auc)
#
# MNNMDA = pd.read_excel('../MNNMDA.xlsx', sheet_name=sheetName, header=None)
# mean_auc = sklearn.metrics.auc(MNNMDA[0], MNNMDA[1])
# plt.plot(MNNMDA[0], MNNMDA[1], label='MNNMDA(AUC=%0.4f)' % mean_auc)
#
# GPUDMDA = pd.read_excel('../GPUDMDA.xlsx', sheet_name=sheetName, header=None)
# mean_auc = sklearn.metrics.auc(GPUDMDA[0], GPUDMDA[1])
# plt.plot(GPUDMDA[0], GPUDMDA[1], color='#955AA6', label='GPUDMDA(AUC=%0.4f)' % mean_auc)
#
# plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))
# plt.gca().xaxis.set_major_locator(MultipleLocator(0.2))
# plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
# plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
# plt.tick_params(axis='both', which='major', direction='in', length=6)
# plt.tick_params(axis='both', which='minor', direction='in', length=3)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend(loc='center right', framealpha=0)
# plt.savefig('./HMDAD_CV1_AUPR.png')
# plt.show()
