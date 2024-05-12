import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 读取 JSON 文件
with open('./cluster/annotation_IA3.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# 创建一个空的数组来存储每个小数组最相似的结果
most_similar_results = []

# 对每个小数组计算结果之间的余弦相似度
for small_array in data:
    # 使用 TF-IDF 表示每个结果
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(small_array)

    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 找到每个结果与其他结果最相似的一个
    # most_similar_index = sorted(enumerate(similarity_matrix.mean(axis=1)), key=lambda x: x[1], reverse=True)[1][0]
    most_similar_index = sorted(enumerate(similarity_matrix.mean(axis=1)), key=lambda x: x[1], reverse=True)[0][0]

    # 选择与其他结果最相似的一个结果
    most_similar_result = small_array[most_similar_index]

    most_similar_results.append(most_similar_result)

# 将最相似的结果写入 TXT 文件
with open('annotation_yuxian1.txt', 'w', encoding='utf-8') as output_file:
    for result in most_similar_results:
        output_file.write(f'{result}\n')
