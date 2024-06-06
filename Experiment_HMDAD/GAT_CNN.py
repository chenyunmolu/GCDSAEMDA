import os
import random
import timeit

import dgl
import pandas as pd
import numpy as np
import torch
from dgl.nn.pytorch import GATv2Conv
from sklearn.metrics import roc_auc_score
from torch import nn
import torch.nn.functional as F

'''
    使用GAT_CNN模块生成特征矩阵：cnn_outputs
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# MLP（多层感知机）：常用于分类任务
# 自定义参数：embedding_size, drop_rate
class MLP(nn.Module):
    def __init__(self, embedding_size, drop_rate):
        super(MLP, self).__init__()
        self.embedding_size = embedding_size  # 指定嵌入大小和丢弃率
        self.drop_rate = drop_rate

        def init_weights(m):  # 初始化模型的权重
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Conv2d:
                nn.init.uniform_(m.weight)

        self.mlp_prediction = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size // 2),  # //表示整数除法操作符，表示对两个数进行除法操作，然后取结果的整数部分，以确保结果是整数
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 2, self.embedding_size // 4),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 4, self.embedding_size // 6),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 6, 1, bias=False),
            nn.Sigmoid()
        ).to(device)
        self.mlp_prediction.apply(init_weights)

    def forward(self, rd_features_embedding):
        predict_result = self.mlp_prediction(rd_features_embedding)
        return predict_result


# 改进后的模型
class GAT_CNN_DASE(nn.Module):
    def __init__(self, in_micfeat_size, in_disfeat_size, outfeature_size, heads, drop_rate, negative_slope,
                 features_embedding_size, negative_times):
        super(GAT_CNN_DASE, self).__init__()
        self.in_micfeat_size = in_micfeat_size
        self.in_disfeat_size = in_disfeat_size
        self.outfeature_size = outfeature_size
        self.heads = heads
        self.drop_rate = drop_rate
        self.negative_slope = negative_slope
        self.features_embedding_size = features_embedding_size
        self.negative_times = negative_times
        # 图注意层（多头）
        self.att_layer = GATv2Conv(self.outfeature_size, self.outfeature_size, self.heads, self.drop_rate,
                                   self.drop_rate, self.negative_slope)

        # 定义投影算子
        self.W_mic = nn.Parameter(torch.zeros(size=(self.in_micfeat_size, self.outfeature_size)))
        self.W_dis = nn.Parameter(torch.zeros(size=(self.in_disfeat_size, self.outfeature_size)))

        # 初始化投影算子，尾部的_表示"in-place"（原地操作）即：修改原值
        nn.init.xavier_uniform_(self.W_mic.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_dis.data, gain=1.414)

        # 定义卷积层的权重初始化函数
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)  # 使用Xavier均匀初始化对权重进行初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 将偏置初始化为0
            elif type(m) == nn.Conv2d:  # 如果m是二维卷积层，使用均匀分布初始化方法初始化方法
                nn.init.uniform_(m.weight)

        # 二维卷积层搭建
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(self.heads, 1), padding=0),  # 二维卷积层，对数据进行卷积操作
            nn.ReLU(),  # 激活函数层，对卷积层的输出进行非线性变换
            nn.Flatten()  # 将多维输入展平成一维的层，通常用于将卷积层的输出展平成一维向量，以便输入到全连接层中
        ).to(device)
        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(self.heads, 4), padding=0),
            # 同上，只是改变了卷积核的宽度（4）
            nn.ReLU(),
            nn.Flatten()
        ).to(device)

        self.cnn_layer16 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(self.heads, 16), padding=0),
            # 同上，只是改变了卷积核的宽度（16）
            nn.ReLU(),
            nn.Flatten()
        ).to(device)

        self.cnn_layer32 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(self.heads, 32), padding=0),
            # 同上，只是改变了卷积核的宽度（32）
            nn.ReLU(),
            nn.Flatten()
        ).to(device)
        # 初始化
        self.cnn_layer1.apply(init_weights)
        self.cnn_layer4.apply(init_weights)
        self.cnn_layer16.apply(init_weights)
        self.cnn_layer32.apply(init_weights)

        # MLP
        self.mlp_prediction = MLP(self.features_embedding_size, self.drop_rate)

    def forward(self, graph, mic_feature_tensor, dis_feature_tensor, association_matrix, train_model):
        print("----------------------------------将microbe和disease映射到同一维度----------------------------------")
        mic_mic_f = mic_feature_tensor.mm(self.W_mic)  # circ_feature_tensor和W_rna矩阵相乘
        dis_dis_f = dis_feature_tensor.mm(self.W_dis)  # dis_feature_tensor和W_dis矩阵相乘
        N = mic_mic_f.size()[0] + dis_dis_f.size()[0]  # 异构网络的节点个数,num_circ+num_dis

        h_m_d_feature = torch.cat((mic_mic_f, dis_dis_f), dim=0)  # 将两个tensor按行拼接成一个新的矩阵

        # 特征聚合
        print("----------------------------------使用GATv2Conv进行特征聚合----------------------------------")
        res = self.att_layer(graph, h_m_d_feature)  # size:[nodes,heads,outfeature_size]
        x = res.view(N, 1, self.heads, -1)  # 将res转换为一个四维张量
        print("----------------------------------使用具有不同卷积核的cnn进行卷积操作----------------------------------")
        cnn_embedding1 = self.cnn_layer1(x).view(N, -1)  # 进行相应的卷积处理，并调整为指定维度，‘-1’表示通过自动计算来确定剩余的维度，以保证数据的总维度不变
        cnn_embedding4 = self.cnn_layer4(x).view(N, -1)
        cnn_embedding16 = self.cnn_layer16(x).view(N, -1)
        cnn_embedding32 = self.cnn_layer32(x).view(N, -1)
        print("----------------------------------将四层CNN进行水平拼贴----------------------------------")
        cnn_outputs = torch.cat([cnn_embedding1, cnn_embedding4, cnn_embedding16, cnn_embedding32], dim=1)  # 按列拼接

        '''
        这段代码根据 train_model 的值，选择性地进行训练或测试，并返回相应的预测结果和标签。
        '''
        if train_model:
            print("----------------------------------根据cnn_outputs,生成用于训练的正负样本----------------------------------")
            mic_nums = association_matrix.size()[0]
            features_embedding_mic = cnn_outputs[0:mic_nums, :]  # 对特征矩阵进行切片操作，将其分为两大部分
            features_embedding_dis = cnn_outputs[mic_nums:cnn_outputs.size()[0], :]
            train_features_input, train_lable = [], []
            # positive position index
            positive_index_tuple = torch.where(association_matrix == 1)
            positive_index_list = list(zip(positive_index_tuple[0], positive_index_tuple[1]))

            for (r, d) in positive_index_list:
                # positive samples
                # 将正样本的特征乘积结果作为输入，添加到train_features_input列表中。
                train_features_input.append((features_embedding_mic[r, :] * features_embedding_dis[d, :]).unsqueeze(0))
                # 将标签值1添加到train_lable列表中
                train_lable.append(1)
            # negative samples
            # 接下来的代码块处理负样本
            negative_index_tuple = torch.where(association_matrix == 0)
            negative_index_list_temp = list(zip(negative_index_tuple[0], negative_index_tuple[1]))

            negative_index_list = random.sample(negative_index_list_temp, len(positive_index_list))
            for (r, d) in negative_index_list:
                train_features_input.append((features_embedding_mic[r, :] * features_embedding_dis[d, :]).unsqueeze(0))
                # 将标签值1添加到train_lable列表中
                train_lable.append(0)
            # 将训练数据列表和标签列表转换为tensor，以便后续操作
            train_features_input = torch.cat(train_features_input, dim=0).to(device)
            train_lable = torch.FloatTensor(np.array(train_lable)).unsqueeze(1).to(device)
            train_mlp_result = self.mlp_prediction(train_features_input)
            return train_mlp_result, train_lable, cnn_outputs
        else:
            mic_nums, dis_nums = association_matrix.size()[0], association_matrix.size()[1]
            features_embedding_mic = cnn_outputs[0:mic_nums, :]  # 对特征矩阵进行切片操作，将其分为两大部分
            features_embedding_dis = cnn_outputs[mic_nums:cnn_outputs.size()[0], :]
            test_features_input, test_lable = [], []
            for i in range(mic_nums):
                for j in range(dis_nums):
                    test_features_input.append(
                        (features_embedding_mic[i, :] * features_embedding_dis[j, :]).unsqueeze(0))
                    test_lable.append(association_matrix[i, j].item())  # 将tensor类型转为float，因为np.array函数无法接受tensor

            test_features_input = torch.cat(test_features_input, dim=0).to(device)
            test_lable = torch.FloatTensor(np.array(test_lable)).unsqueeze(1).to(device)
            test_mlp_result = self.mlp_prediction(test_features_input)
            return test_mlp_result, test_lable, cnn_outputs


# 构建miccobe-disease异构图网络
def build_heterograph(microbe_disease_matrix, microbeSimi, disSimi):
    # for microbe->adj
    matAdj_circ = np.where(microbeSimi > 0.5, 1, 0)

    # for disease->adj
    matAdj_dis = np.where(disSimi > 0.5, 1, 0)

    # Heterogeneous adjacency matrix
    h_adjmat_1 = np.hstack((matAdj_circ, microbe_disease_matrix))
    h_adjmat_2 = np.hstack((microbe_disease_matrix.transpose(), matAdj_dis))
    Heterogeneous = np.vstack((h_adjmat_1, h_adjmat_2))

    # heterograph
    g = dgl.heterograph(
        data_dict={
            ('microbe_disease', 'interaction', 'microbe_disease'): Heterogeneous.nonzero()},
        num_nodes_dict={
            'microbe_disease': microbe_disease_matrix.shape[0] + microbe_disease_matrix.shape[1]
        })
    return g


if __name__ == '__main__':
    start_time = timeit.default_timer()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 设置随机数，保证结果的可复现性
    seed = 36
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

    # 读取微生物和疾病的相似性融合矩阵，以及微生物和疾病关联矩阵
    MD_association_matrix = pd.read_csv('../Dataset/HMDAD/mircobe_disease_association_matrix.csv', index_col=0)
    microbe_similarity_fusion_matrix = pd.read_csv('../Dataset/HMDAD/microbe_similarity_fusion_matrix.csv', index_col=0)
    disease_similarity_fusion_matrix = pd.read_csv('../Dataset/HMDAD/disease_similarity_fusion_matrix.csv', index_col=0)

    mic_nums = MD_association_matrix.shape[0]
    dis_nums = MD_association_matrix.shape[1]

    MD = np.array(MD_association_matrix)
    MM = np.array(microbe_similarity_fusion_matrix)
    DD = np.array(disease_similarity_fusion_matrix)

    g = build_heterograph(MD, MM, DD).to(device)

    MM_tensor = torch.from_numpy(MM).to(torch.float32).to(device)
    DD_tensor = torch.from_numpy(DD).to(torch.float32).to(device)
    MD_tensor = torch.from_numpy(MD).to(torch.float32).to(device)

    model = GAT_CNN_DASE(MD.shape[0], MD.shape[1], 128, 8, 0.1, 0.3, 2778, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-7)
    epochs = 100
    # 模型训练
    model.train()
    for epoch in range(epochs):
        print("----------------------------------start {} training----------------------------------".format(epoch + 1))
        train_predict_result, train_lable, _ = model(g, MM_tensor, DD_tensor, MD_tensor, train_model=True)
        # binary_cross_entropy：二元交叉熵损失函数，通常用于二分类问题中的神经网络训练，该函数可以根据train_predict_result自行计算类别索引，然后计算loss
        loss = F.binary_cross_entropy(train_predict_result, train_lable)
        # steps.append(epoch)
        # loss_value.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch %d | train Loss: %.4f' % (epoch + 1, loss.item()))  # 格式化字符串，两个占位符‘%d’、'%.4f',分别表示整数和保留4位小数的浮点数

    _, _, cnn_outputs = model(g, MM_tensor, DD_tensor, MD_tensor, train_model=True)
    cnn_outputs = cnn_outputs.cpu().detach().numpy()
    cnn_outputs_dataframe = pd.DataFrame(cnn_outputs)
    cnn_outputs_dataframe.to_csv("cnn_outputs_dataframe.csv")

    end_time = timeit.default_timer()
    print("Execution Time: ", end_time - start_time)
