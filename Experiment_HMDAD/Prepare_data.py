import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
基于HMDAD数据库整合微生物和疾病的各种相似度
'''


# 计算微生物的功能相似性
def microbe_functional_similarity():
    # 修改目录即可
    # 微生物、疾病关联矩阵
    MD = pd.read_csv("../Dataset/HMDAD/mircobe_disease_association_matrix.csv", index_col=0)
    # 疾病语义相似性矩阵
    DS = pd.read_csv("../Dataset/HMDAD/disease_do_similarity.csv", index_col=0)
    # 将DataFrame类型转为numpy类型，等价于DS = np.array(DS.values)，舍弃了索引名和列名
    DS = np.array(DS)
    M_num = MD.shape[0]
    T = []
    for i in range(M_num):
        T.append(np.where(MD.iloc[i] == 1))
    FS = []
    for i in range(M_num):
        for j in range(M_num):
            T0_T1 = []
            if len(T[i][0]) != 0 and len(T[j][0]) != 0:
                for ti in T[i][0]:
                    max_ = []
                    for tj in T[j][0]:
                        max_.append(DS[ti][tj])
                    T0_T1.append(max(max_))
            if len(T[i][0]) == 0 or len(T[j][0]) == 0:
                T0_T1.append(0)
            T1_T0 = []
            if len(T[i][0]) != 0 and len(T[j][0]) != 0:
                for tj in T[j][0]:
                    max_ = []
                    for ti in T[i][0]:
                        max_.append(DS[tj][ti])
                    T1_T0.append(max(max_))
            if len(T[i][0]) == 0 or len(T[j][0]) == 0:
                T1_T0.append(0)

            a = len(T[i][0])
            b = len(T[j][0])
            S1 = sum(T0_T1)
            S2 = sum(T1_T0)
            fs = []
            if a != 0 and b != 0:
                fsim = (S1 + S2) / (a + b)
                fs.append(fsim)
            if a == 0 or b == 0:
                fs.append(0)
            FS.append(fs)
    FS = np.array(FS).reshape(M_num, M_num)
    FS = pd.DataFrame(FS)
    # 下列代码有点冗余，因为原本矩阵对角线上的值就全是1
    for index, rows in FS.iterrows():
        for col, rows in FS.iterrows():
            if index == col:
                FS.loc[index, col] = 1
    # 修改数据框的索引名和列名
    FS.index = MD.index
    FS.columns = MD.index
    return FS


# 根据关联矩阵计算微生物和疾病的GIP相似性
def GIP_similarity():
    MD = pd.read_csv("../Dataset/HMDAD/mircobe_disease_association_matrix.csv", index_col=0)
    # 首先求出microbe和disease的带宽参数rm，rd
    microbe_num = MD.shape[0]
    disease_num = MD.shape[1]
    EUC_M = np.linalg.norm(MD, ord=2, axis=1, keepdims=False)
    EUC_D = np.linalg.norm(MD.T, ord=2, axis=1, keepdims=False)
    SUM_EUC_M = np.sum(EUC_M ** 2)
    SUM_EUC_D = np.sum(EUC_D ** 2)
    rm = 1 / ((1 / microbe_num) * SUM_EUC_M)
    rd = 1 / ((1 / disease_num) * SUM_EUC_D)
    # 计算microbe和disease的GIP
    microbe_GIP = pd.DataFrame(0, index=MD.index, columns=MD.index)
    disease_GIP = pd.DataFrame(0, index=MD.columns, columns=MD.columns)
    MD = np.mat(MD)
    for i in range(microbe_num):
        for j in range(microbe_num):
            m_norm = np.linalg.norm(MD[i] - MD[j], ord=2, axis=1, keepdims=False)
            m_norm = m_norm ** 2
            m_norm_result = np.exp(-rm * m_norm)
            microbe_GIP.iloc[i, j] = m_norm_result
    for i in range(disease_num):
        for j in range(disease_num):
            d_norm = np.linalg.norm(MD.T[i] - MD.T[j], ord=2, axis=1, keepdims=False)
            d_norm = d_norm ** 2
            d_norm_result = np.exp(-rd * d_norm)
            disease_GIP.iloc[i, j] = d_norm_result
    return microbe_GIP, disease_GIP


if __name__ == '__main__':
    microbe_functional_similarity = microbe_functional_similarity()
    microbe_GIP_similarity, disease_GIP_similarity = GIP_similarity()
    microbe_functional_similarity.to_csv("../Dataset/HMDAD/microbe_functional_similarity.csv")
    microbe_GIP_similarity.to_csv("../Dataset/HMDAD/microbe_GIP_similarity.csv")
    disease_GIP_similarity.to_csv("../Dataset/HMDAD/disease_GIP_similarity.csv")

    microbe_similarity_fusion_matrix = pd.DataFrame(0, index=microbe_functional_similarity.index,
                                                    columns=microbe_functional_similarity.index)
    microbe_functional_similarity = np.array(microbe_functional_similarity)
    microbe_GIP_similarity = np.array(microbe_GIP_similarity)
    for i in range(microbe_functional_similarity.shape[0]):
        for j in range(microbe_functional_similarity.shape[1]):
            if microbe_functional_similarity[i, j] != 0:
                microbe_similarity_fusion_matrix.iloc[i, j] = (microbe_functional_similarity[i, j] +
                                                               microbe_GIP_similarity[i, j]) / 2
            else:
                microbe_similarity_fusion_matrix.iloc[i, j] = microbe_GIP_similarity[i, j]

    microbe_similarity_fusion_matrix.to_csv("../Dataset/HMDAD/microbe_similarity_fusion_matrix.csv")

    disease_do_similarity = pd.read_csv("../Dataset/HMDAD/disease_do_similarity.csv", index_col=0)
    disease_similarity_fusion_matrix = pd.DataFrame(0, index=disease_do_similarity.index,
                                                    columns=disease_do_similarity.index)
    disease_do_similarity = np.array(disease_do_similarity)
    disease_GIP_similarity = np.array(disease_GIP_similarity)
    for i in range(disease_do_similarity.shape[0]):
        for j in range(disease_do_similarity.shape[1]):
            if disease_do_similarity[i, j] != 0:
                disease_similarity_fusion_matrix.iloc[i, j] = (disease_do_similarity[i, j] +
                                                               disease_GIP_similarity[i, j]) / 2
            else:
                disease_similarity_fusion_matrix.iloc[i, j] = disease_GIP_similarity[i, j]
    disease_similarity_fusion_matrix.to_csv("../Dataset/HMDAD/disease_similarity_fusion_matrix.csv")
