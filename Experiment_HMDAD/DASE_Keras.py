import os

import keras
import xlsxwriter
from keras import layers, regularizers
import tensorflow as tf
from matplotlib.ticker import MultipleLocator
from sklearn import metrics
from sklearn.cluster import KMeans
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


# 使用keras搭建的深度稀疏自编码器
def deep_sparse_auto_encoder(x_train):
    # feature_size=331
    feature_size = x_train.shape[1]
    input_img = layers.Input(shape=(feature_size,))
    # layer = [256, 128, 96, 64]
    layer = [2048, 1024, 512, 256, 128]
    params = [0.1, 0.0005, 0.05]
    # 编码层
    encoded = layers.Dense(layer[0], activation='relu')(input_img)
    encoded = layers.Dense(layer[1], activation='relu')(encoded)
    encoded = layers.Dense(layer[2], activation='relu')(encoded)
    encoded = layers.Dense(layer[3], activation='relu')(encoded)
    encoder_output = layers.Dense(layer[4], activation='relu')(encoded)
    # 解码层
    decoded = layers.Dense(layer[3], activation='relu')(encoder_output)
    decoded = layers.Dense(layer[2], activation='relu')(decoded)
    decoded = layers.Dense(layer[1], activation='relu')(decoded)
    decoded = layers.Dense(layer[0], activation='relu')(decoded)
    decoded = layers.Dense(feature_size, activation='tanh')(decoded)

    # KL divergence regularization
    def kl_divergence(rho, activations):
        rho_hat = tf.reduce_mean(activations, axis=0)
        kl_div = rho * tf.math.log(rho / rho_hat) + (1 - rho) * tf.math.log((1 - rho) / (1 - rho_hat))
        return kl_div

    # Sparse loss function
    # params = [0.1, 0.0005, 0.05]
    def sparse_loss(penalty=params[1], sparsity=params[2]):
        # y_true 和 y_pred 并不需要手动传递，而是在模型训练时由 Keras 自动传递给损失函数
        def loss(y_true, y_pred):
            # 使用 TensorFlow 计算预测值 y_pred 和真实值 y_true 之间均方误差（MSE）的平均值
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            # 计算这些平均值的目的是为了实现稀疏性约束。稀疏性约束通常会鼓励模型学习到一组对于输入数据表示来说具有更少活跃值的特征。
            # 通过计算平均值，你可以评估模型在每个特征维度上的活跃程度，从而调整稀疏性约束的力度。
            activations = tf.reduce_mean(encoder_output, axis=0)
            kl_div = tf.reduce_sum(kl_divergence(sparsity, activations))
            return params[0] * mse_loss + penalty * kl_div

        return loss

        # Autoencoder model

    autoencoder = keras.Model(inputs=input_img, outputs=decoded)
    # Encoder model
    encoder = keras.Model(inputs=input_img, outputs=encoder_output)
    autoencoder.compile(optimizer='adam', loss=sparse_loss())
    # Train model

    # earlyStop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=80, mode='max', verbose=0,
    #                           restore_best_weights=True)
    autoencoder.fit(x_train, x_train, epochs=20, batch_size=32, shuffle=True, verbose=0)
    encoded_imgs = encoder.predict(x_train)
    return encoded_imgs


# -----------------------------------------------------Keras工具包--------------------------------------------------------
# CV1、CV2、CV3的交叉验证
def kfold_by_CV(data, k, row=0, col=0, cv=3):
    dlen = len(data)
    if cv == 2:
        lens = row
        split = col
    elif cv == 1:
        lens = col
        split = row
    else:
        lens = dlen
    d = list(range(lens))
    np.random.shuffle(d)
    test_n = lens // k
    n = lens % k
    test_res = []
    train_res = []
    for i in range(n):
        test = d[i * (test_n + 1):(i + 1) * (test_n + 1)]
        train = list(set(d) - set(test))
        test_res.append(test)
        train_res.append(train)
    # 下面这个for循环是为了将没有补全的test_res、 train_res补成5份，凑足五折交叉验证
    for i in range(n, k):
        test = d[i * test_n + n:(i + 1) * test_n + n]
        train = list(set(d) - set(test))
        test_res.append(test)
        train_res.append(train)
    if cv == 3:
        return train_res, test_res

    train_s = []
    test_s = []
    d = range(dlen)
    for k in range(len(test_res)):
        test = []
        for i in range(len(test_res[k])):
            if cv == 2:
                tmp = np.full(split, test_res[k][i] * split) + range(split)
            elif cv == 1:
                tmp = np.full(split, test_res[k][i]) + [i * lens for i in range(split)]
            test = np.concatenate((test, tmp), axis=0)
        test = np.array(test, dtype=int).tolist()
        test_s.append(test)
        train = np.array(list(set(d) - set(test)), dtype=int)
        train_s.append(train)
    return train_s, test_s


# xlsxwriter库储存数据到excel：以下示例是将AUC曲线的fpr，tpr分别作为x，y轴数据
def data_toExcel(x, y, fileName, sheet):
    # 创建一个新的Excel工作簿
    workbook = xlsxwriter.Workbook(fileName)
    # 根据提供的索引添加一个新的工作表，并激活该工作表
    worksheet1 = workbook.add_worksheet(sheet)
    worksheet1.activate()
    # 遍历x和y的数据，并将它们写入工作表中
    for i in range(len(x)):
        insertData = [x[i], y[i]]  # 准备要写入的数据行
        row = 'A' + str(i + 1)  # 计算要写入数据的行号
        worksheet1.write_row(row, insertData)  # 写入数据行
    # 关闭工作簿，将数据保存到文件
    workbook.close()


# 使用随机采样的方法获取负样本
def get_negative_sample_by_randomSample(NEGATIVE_SAMPLE_CHA_ALL, positive_sample_number):
    NEGATIVE_SAMPLE_CHA = random.sample(NEGATIVE_SAMPLE_CHA_ALL.tolist(), positive_sample_number)
    NEGATIVE_SAMPLE_CHA = np.array(NEGATIVE_SAMPLE_CHA)
    NEGATIVE_SAMPLE_CHA_LABEL = np.zeros((NEGATIVE_SAMPLE_CHA.shape[0], 1))
    return NEGATIVE_SAMPLE_CHA, NEGATIVE_SAMPLE_CHA_LABEL


# KMeans聚类方法选择负样本
def get_negative_sample_by_KMeans(NEGATIVE_SAMPLE_CHA_ALL, positive_sample_number):
    kmeans = KMeans(n_clusters=23, random_state=36).fit(NEGATIVE_SAMPLE_CHA_ALL)
    kmeans_labels = kmeans.labels_
    type = [[] for _ in range(23)]
    for i in range(len(kmeans_labels)):
        type[kmeans_labels[i]].append(NEGATIVE_SAMPLE_CHA_ALL[i])
    mytype = [[] for _ in range(23)]
    for j in range(23):
        mytype[j] = random.sample(type[j], positive_sample_number // 23)
    mytype_np = np.array(mytype)
    NEGATIVE_SAMPLE_CHA = mytype_np.reshape(-1, NEGATIVE_SAMPLE_CHA_ALL.shape[1])
    NEGATIVE_SAMPLE_CHA_LABEL = np.zeros((NEGATIVE_SAMPLE_CHA.shape[0], 1))
    return NEGATIVE_SAMPLE_CHA, NEGATIVE_SAMPLE_CHA_LABEL


# 在KMeans聚类的基础上使用cosine_distances计算样本与中心的距离,从而选择高质量的负样本
def get_negative_sample_by_KMeans_and_cosine_distances(NEGATIVE_SAMPLE_CHA_ALL, positive_sample_number):
    kmeans = KMeans(n_clusters=23, random_state=36).fit(NEGATIVE_SAMPLE_CHA_ALL)
    kmeans_labels = kmeans.labels_
    kmeans_cluster_centers = kmeans.cluster_centers_
    type = [[] for _ in range(23)]
    for i in range(len(kmeans_labels)):
        type[kmeans_labels[i]].append(NEGATIVE_SAMPLE_CHA_ALL[i])
    mytype = [[] for _ in range(23)]
    for j in range(23):
        type_numpy = np.array(type[j])
        kmeans_cluster_centers_numpy = np.array(kmeans_cluster_centers[j])
        cosine = cosine_distances(type_numpy, kmeans_cluster_centers_numpy.reshape(1, -1))
        sorted_index = np.argsort(cosine.ravel())
        mytype[j] = type_numpy[sorted_index[:positive_sample_number // 23]]
    mytype_np = np.array(mytype)
    NEGATIVE_SAMPLE_CHA = mytype_np.reshape(-1, NEGATIVE_SAMPLE_CHA_ALL.shape[1])
    NEGATIVE_SAMPLE_CHA_LABEL = np.zeros((NEGATIVE_SAMPLE_CHA.shape[0], 1))
    return NEGATIVE_SAMPLE_CHA, NEGATIVE_SAMPLE_CHA_LABEL


def draw_ROC_curve(FPR, TPR, cv):
    mean_fpr = np.linspace(0, 1, 1000)
    tprs = []
    for i in range(len(FPR)):
        tprs.append(np.interp(mean_fpr, FPR[i], TPR[i]))
        tprs[-1][0] = 0.0
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)

    filepath = "./Result/cv%d/" % (cv)
    os.makedirs(filepath, exist_ok=True)
    data_toExcel(mean_fpr, mean_tpr, filepath + "AUC_CV%d_%.4f_mean.xlsx" % (cv, mean_auc), "HMDAD_CV%d_AUC" % cv)
    '''
    修改刻度长度，并且显示双数，隐藏单数，建议根据需求进行更改
    '''
    plt.figure(figsize=(10, 8))
    plt.plot(mean_fpr, mean_tpr, color='red', label='Mean ROC (AUC = %0.4f)' % mean_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.2))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.tick_params(axis='both', which='major', direction='in', length=6)
    plt.tick_params(axis='both', which='minor', direction='in', length=3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve")
    # 将图例固定在右下
    plt.legend(loc=4)
    plt.show()


def draw_ROC_curve_by_five_fold(FPR, TPR):
    plt.figure(figsize=(10, 8))
    tprs = []
    mean_fpr = np.linspace(0, 1, 1000)
    for i in range(len(FPR)):
        # tpr是一个数量为len(mean_fpr)的一个数组
        tpr = np.interp(mean_fpr, FPR[i], TPR[i])
        tpr[0] = 0.0
        tprs.append(tpr)
        auc = metrics.auc(FPR[i], TPR[i])
        plt.plot(FPR[i], TPR[i], lw=0.5, label='ROC fold %d (auc=%0.4f)' % (i, auc))
        plt.legend()
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color="#D81C38", lw=2, label='Mean ROC (mean auc=%0.4f)' % (mean_auc))
    # 设置图像的刻度
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.2))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.tick_params(axis='both', which='major', direction='in', length=6)
    plt.tick_params(axis='both', which='minor', direction='in', length=3)
    # 设置图像的轴名称和标题
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve")
    # 将图例固定在右下
    plt.legend(loc=4)
    plt.show()


def draw_PR_curve(test_label_all, test_predict_prob_all, cv):
    y_real = np.concatenate(test_label_all)
    y_proba = np.concatenate(test_predict_prob_all)
    precisions, recalls, _ = precision_recall_curve(y_real, y_proba, pos_label=1)
    mean_aupr = metrics.auc(recalls, precisions)

    filepath = "./Result/cv%d/" % (cv)
    os.makedirs(filepath, exist_ok=True)
    data_toExcel(recalls, precisions, filepath + "AUPR_CV%d_%.4f_mean.xlsx" % (cv, mean_aupr), "HMDAD_CV%d_AUPR" % cv)

    plt.figure(figsize=(10, 8))
    plt.plot(recalls, precisions, color='red', label='Mean PR (AUPR = %0.4f)' % mean_aupr)
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.2))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.tick_params(axis='both', which='major', direction='in', length=6)
    plt.tick_params(axis='both', which='minor', direction='in', length=3)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    # 将图例固定在右下
    plt.legend(loc=4)
    plt.show()


def draw_PR_curve_new(RECALL, PRECISION):
    auprs = []
    for i in range(len(RECALL)):
        auprs.append(metrics.auc(RECALL[i], PRECISION[i]))
    mean_aupr = np.mean(auprs)
    plt.figure(figsize=(10, 8))
    plt.plot(RECALL[0], PRECISION[0], label='Mean PR (AUPR = %0.4f)' % auprs[0])
    plt.plot(RECALL[1], PRECISION[1], label='Mean PR (AUPR = %0.4f)' % auprs[1])
    plt.plot(RECALL[2], PRECISION[2], label='Mean PR (AUPR = %0.4f)' % auprs[2])
    plt.plot(RECALL[3], PRECISION[3], label='Mean PR (AUPR = %0.4f)' % auprs[3])
    plt.plot(RECALL[4], PRECISION[4], label='Mean PR (AUPR = %0.4f)' % auprs[4])
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.2))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.tick_params(axis='both', which='major', direction='in', length=6)
    plt.tick_params(axis='both', which='minor', direction='in', length=3)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve new')
    # 将图例固定在右下
    plt.legend(loc=4)
    plt.show()
