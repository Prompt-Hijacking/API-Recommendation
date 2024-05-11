import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 假设你有一个名为 'test_3_lines_s_1000.csv' 的CSV文件
df = pd.read_csv('validate_3_lines_new_bimodal.csv')

# 获取英文句子
sentences = df['bimodal'].tolist()

# 将句子转换成 TF-IDF 矩阵
vectorizer = TfidfVectorizer(stop_words='english')  # 'english' 在这里是停用词表，可以根据需要调整
tfidf_matrix = vectorizer.fit_transform(sentences)

# 使用 KMeans 聚类找到代表性句子
num_clusters = min(3000, len(sentences))  # 根据你的需求调整聚类簇的数量
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=1)  # 显式地设置 n_init 来避免 FutureWarning
kmeans.fit(tfidf_matrix)

# 获取每个簇的代表性句子的索引
representative_indices = []
for i in range(num_clusters):
    cluster_center = kmeans.cluster_centers_[i]
    cluster_indices = (tfidf_matrix @ cluster_center).argsort()[-1:-2:-1]
    representative_indices.extend(cluster_indices)

# 选择前10条数据
selected_data = df.iloc[representative_indices[:3000]]

# 将选取的数据写入新的 CSV 文件
selected_data.to_csv('validate_3_lines_tf_bimodal_3000.csv', index=False)

# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
#
# # 加载数据
# df = pd.read_csv('train_3_lines_new_bimodal.csv')
# sentences = df['bimodal'].tolist()
#
# # 将句子转换成 TF-IDF 矩阵
# vectorizer = TfidfVectorizer(stop_words='english')
# tfidf_matrix = vectorizer.fit_transform(sentences)
#
# # 尝试不同的k值，找到最佳的k
# sse = []
# k_range = range(1, min(300, len(sentences)), 20)  # 以20为步长尝试不同的k值，这里的范围可以根据需要调整
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=1)
#     kmeans.fit(tfidf_matrix)
#     sse.append(kmeans.inertia_)  # inertia_属性是模型的SSE
#
# # 绘制SSE图
# plt.figure(figsize=(10, 6))
# plt.plot(k_range, sse, marker='o')
# plt.title('Elbow Method For Optimal k')
# plt.xlabel('Number of clusters k')
# plt.ylabel('SSE')
# plt.xticks(k_range)
# plt.show()

# 根据图形选择最佳的k值
# 假设根据肘部法则你选择了一个最佳k值
# optimal_k = # 这里需要你根据上图选择一个具体的值
#
# # 使用选定的最佳k值继续之前的聚类和数据处理步骤
# kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=1)
# kmeans.fit(tfidf_matrix)

# ...接下来是选择代表性句子和保存新CSV的代码，与之前相同


