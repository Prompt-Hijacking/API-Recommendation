# calculate_cider_score.py

from pycocoevalcap.cider.cider import Cider
import argparse

def calculate_cider():
    # 读取参考句子和候选句子的文本文件
    def read_text_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = [line.strip() for line in file]
        return data

    # 解析命令行参数
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--reference', type=str, default='reference.txt', help='Reference File')
    argparser.add_argument('--candidate', type=str, default='candidate.txt', help='Candidate file')
    args = argparser.parse_args()

    # 读取文本文件中的数据
    reference_data = read_text_file(args.reference)
    candidate_data = read_text_file(args.candidate)

    # 确保两个文件的行数一致
    assert len(reference_data) == len(candidate_data), "Number of lines in reference and candidate files should be the same."

    # 将数据组织成字典格式
    references = {i + 1: [reference_data[i]] for i in range(len(reference_data))}
    candidates = {i + 1: [candidate_data[i]] for i in range(len(candidate_data))}

    # 初始化 CIDEr 评价器
    cider_eval = Cider()

    # 计算 CIDEr 分数
    cider_score, _ = cider_eval.compute_score(references, candidates)

    print(f"CIDEr Score: {cider_score}")

if __name__ == "__main__":
    calculate_cider()
