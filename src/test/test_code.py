import os
import torch
import random
import json
import pandas as pd
import numpy as np
import torch.nn as nn
from datasets import Dataset
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from peft import get_peft_model, PrefixTuningConfig, TaskType, PeftModel, PeftConfig
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AdamW, default_data_collator, get_linear_schedule_with_warmup, AutoModelForCausalLM, DataCollatorWithPadding

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = "cuda"
# model_name_or_path = "t5-large"
max_length = 256
# lr = 1e-2
batch_size = 32

peft_model_id = "../code_1500/model_best_bleu"
config = PeftConfig.from_pretrained(peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
# 加载 PeftModel
model = PeftModel.from_pretrained(model, peft_model_id)
model.to(device)  # 将模型移到 GPU 上

# 加载训练集、测试集和验证集
# test_data = pd.read_csv("../data/test_3_lines_dedup.csv")
test_data = pd.read_csv("/data/CM/Project/LLM4ApiRec/MulaRec/data/test_3_lines_new_bimodal.csv")
columns_to_keep = ["source_code", "target_api"]
test_data = test_data.filter(items=columns_to_keep)
test_dataset = test_data[["source_code", "target_api"]]


# 创建预测数据集
test_dataset = Dataset.from_pandas(test_data)

# 定义预处理函数
def preprocess_function(examples):
    inputs = examples["source_code"]
    targets = examples["target_api"]

    # 使用 tokenizer 处理标签
    labels = tokenizer(targets, padding="max_length", truncation=True, return_tensors="pt", add_special_tokens=False)

    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

    # 设置填充标记
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 对数据集进行预处理
test_dataset = test_dataset.map(preprocess_function, batched=True, num_proc=1)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=default_data_collator)


# # 模型评估部分
# model.eval()

# csv_true_list = []

# all_generated_apis = [[] for _ in range(20)]  # 创建包含10个空列表的列表，用于存储每次生成的结果

# for step, batch in enumerate(tqdm(test_dataloader)):
#     batch = {k: v.to(device) for k, v in batch.items()}
    
#     for i in range(20):  # 遍历每次生成
#         with torch.no_grad():
#             outputs = model.generate(input_ids=batch["input_ids"], max_length=256, temperature=0.85, do_sample=True, top_p=0.95, top_k=None)

#         # 将生成的文本解码为 API
#         generated_apis = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)

#         # 将生成的 API 添加到对应生成次数的列表中
#         all_generated_apis[i].extend(generated_apis)

#     for gold in batch["labels"]:
#         csv_true_list.append(tokenizer.decode(gold, skip_special_tokens=True))


# # 转置操作，将每个子数组的相同索引的元素合并成一个新的子数组
# transposed_generated_apis = list(map(list, zip(*all_generated_apis)))

# # 将所有生成的 API 保存到 JSON 文件
# output_json_path = "code_IA3_20.json"
# with open(output_json_path, 'w') as json_file:
#     json.dump(transposed_generated_apis, json_file, ensure_ascii=False, indent=2)

# df = pd.DataFrame(csv_true_list)
# df.to_csv("target_api.csv", index=False, header=None)


model.eval()

csv_pred_list = []
csv_true_list = []

for step, batch in enumerate(tqdm(test_dataloader)):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():

        outputs = model.generate(input_ids = batch["input_ids"], max_length=256)
    generated_text = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)

    for ref, gold in zip(generated_text, batch["labels"]):
        csv_pred_list.append(ref)
        csv_true_list.append(tokenizer.decode(gold, skip_special_tokens=True))


# 将预测和参考保存到文件
df = pd.DataFrame(csv_true_list)
df.to_csv("./1500/test_code_ref.csv", index=False, header=None)
df = pd.DataFrame(csv_pred_list)
df.to_csv("./1500/test_code_hyp.csv", index=False, header=None)