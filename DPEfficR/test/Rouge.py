from rouge import Rouge
import argparse

def calculate_rouge(reference_file, candidate_file):
    # 读取参考摘要和候选摘要的文本文件
    def read_text_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = [line.strip() for line in file]
        return data

    # 读取文本文件中的数据
    reference_data = read_text_file(reference_file)
    candidate_data = read_text_file(candidate_file)

    # 初始化 Rouge 评价器
    rouge_evaluator = Rouge()

    # 循环遍历每一行，计算 ROUGE 指标
    rouge_scores = {'rouge-1': {'r': 0, 'p': 0, 'f': 0}, 'rouge-2': {'r': 0, 'p': 0, 'f': 0}, 'rouge-l': {'r': 0, 'p': 0, 'f': 0}}

    for ref, cand in zip(reference_data, candidate_data):
        # 计算 ROUGE 指标
        scores = rouge_evaluator.get_scores(cand, ref)[0]

        # 累加 ROUGE 指标
        for metric in rouge_scores.keys():
            for measure in ['r', 'p', 'f']:
                rouge_scores[metric][measure] += scores[metric][measure]

    # 计算平均 ROUGE 指标
    num_samples = len(reference_data)
    for metric in rouge_scores.keys():
        for measure in ['r', 'p', 'f']:
            rouge_scores[metric][measure] /= num_samples

    return rouge_scores

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--reference', type=str, default='reference.txt', help='Reference File')
    argparser.add_argument('--candidate', type=str, default='candidate.txt', help='Candidate file')
    args = argparser.parse_args()

    # 计算 ROUGE 分数
    rouge_scores = calculate_rouge(args.reference, args.candidate)

    # 打印 ROUGE 分数
    print("ROUGE Scores:")
    for metric in rouge_scores.keys():
        for measure in ['r', 'p', 'f']:
            print(f"{metric.upper()}-{measure.upper()}: {rouge_scores[metric][measure]}")
