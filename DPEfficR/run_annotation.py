import os
import re
import torch
import random
import logging
import pandas as pd
import numpy as np
import torch.nn as nn
from datasets import Dataset
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from peft import get_peft_model, PrefixTuningConfig, IA3Config, TaskType, PromptTuningConfig, PromptTuningInit, PromptEncoderConfig, LoraConfig
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AdamW, default_data_collator, get_linear_schedule_with_warmup, AutoModelForCausalLM, DataCollatorWithPadding
import argparse


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda"
model_name_or_path = "Salesforce/codet5-large"
max_length = 128
lr = 1e-2
num_epochs = 100
train_batch_size = 4
validation_batch_size = 16


train_data_path = "../data/1500/train_3_lines_tf_bimodal_1500.csv"
test_data_path = "../data/test_3_lines_new_bimodal.csv"
validation_data_path = "../data/1500/validate_3_lines_random_150.csv"

def read_and_filter_data(data_path):
    data = pd.read_csv(data_path)
    return data.filter(items=["annotation", "target_api"])

train_data = read_and_filter_data(train_data_path)
test_data = read_and_filter_data(test_data_path)
validation_data = read_and_filter_data(validation_data_path)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

def preprocess_function(examples):
    inputs = examples["annotation"]
    targets = examples["target_api"]
    labels = tokenizer(targets, padding="max_length", truncation=True, return_tensors="pt", add_special_tokens=False)
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)
validation_dataset = Dataset.from_pandas(validation_data)

train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=1)
test_dataset = test_dataset.map(preprocess_function, batched=True, num_proc=1)
validation_dataset = validation_dataset.map(preprocess_function, batched=True, num_proc=1)

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=default_data_collator, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=validation_batch_size, collate_fn=default_data_collator)
test_dataloader = DataLoader(test_dataset, batch_size=validation_batch_size, collate_fn=default_data_collator)

parser = argparse.ArgumentParser(description="Script for training the model with different peft_config types.")
parser.add_argument("--peft_config_type", type=str, choices=["PrefixTuning", "PromptTuning", "P-tuning", "Lora", "IA3"],
                    default="IA3", help="Type of peft_config to use.")
# 添加 --output_dir 参数
parser.add_argument("--output_dir", type=str, default="./annotation", help="Output directory for saving logs and models.")
args = parser.parse_args()

# 使用 args.output_dir 构建 log_file 路径
log_file = os.path.join(args.output_dir, f"annotation_training_log.txt")

# 创建日志记录器
logger = logging.getLogger("MyLogger")
log_handler = logging.FileHandler(log_file)
log_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %A %H:%M:%S')
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

def train_model(peft_config):
    # 创建模型
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    model = model.to(device)

    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # 创建一个空列表来存储训练损失
    best_bleu_score = 0.0
    best_model_weights = None
    train_losses = []

    for epoch in range(num_epochs):
        logger.info(f"\n***** Starting Epoch {epoch} *****")
        model.train()
        total_loss = 0

        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # 记录每个训练步骤的损失
            train_losses.append(loss.item())

        # 计算并记录平均训练损失
        avg_train_loss = total_loss / (step + 1)
        train_ppl = torch.exp(avg_train_loss)
        logger.info(f"Epoch {epoch}: Average Training Loss: {avg_train_loss}")
        logger.info(f"Epoch {epoch}: Train_ppl: {train_ppl}")
        logger.info(f"******************************************")

        model_save_dir = os.path.join(args.output_dir, f"./model_last_epoch")
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        # 保存模型权重和配置文件到指定目录
        model.save_pretrained(model_save_dir)
        model.config.save_pretrained(model_save_dir)

        model.eval()
        eval_loss = 0
        generated_text = []

        csv_pred_list = []
        csv_true_list = []
        output_dir = args.output_dir 

        for step, batch in enumerate(tqdm(validation_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
            # 追加生成的文本到列表中
            generated_text = tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)

            eval_loss += loss.detach().float()
            avg_eval_loss = eval_loss / (step + 1)
            eval_ppl = torch.exp(avg_eval_loss)

            for ref, gold in zip(generated_text, batch["labels"]):
                csv_pred_list.append(ref)
                csv_true_list.append(tokenizer.decode(gold, skip_special_tokens=True))

        df = pd.DataFrame(csv_true_list)
        df.to_csv(os.path.join(output_dir, f"valid_ref.csv"), index=False, header=None)

        df = pd.DataFrame(csv_pred_list)
        df.to_csv(os.path.join(output_dir, f"valid_hyp.csv"), index=False, header=None)

        bleu_score = 0.0

        for ref, gold in zip(csv_pred_list, csv_true_list):
            bleu_score += sentence_bleu([gold.strip().split()], ref.strip().split())

        dev_bleu = bleu_score / len(csv_pred_list)

        # 记录验证损失
        logger.info(f"Epoch {epoch}: Validation Loss: {avg_eval_loss}")
        logger.info(f"Epoch {epoch}: Eval_ppl: {eval_ppl}")
        logger.info(f"******************************************")
        # 输出BLEU分数
        logger.info(f"Validation BLEU Score: {dev_bleu}")

        # 保存最佳模型
        if dev_bleu > best_bleu_score:
            best_bleu_score = dev_bleu
            best_model_weights = model.state_dict()

    # 最后，在训练循环之后，保存最佳模型权重
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)
            model_save_dir = os.path.join(args.output_dir, f"./model_best_bleu")  # 保存到指定目录
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)

            # 保存模型权重和配置文件到指定目录
            model.save_pretrained(model_save_dir)
            model.config.save_pretrained(model_save_dir)

            # 输出保存路径
            logger.info(f"Best BLEU Score: {best_bleu_score}")
            logger.info(f"Best model saved to {model_save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for training the model with different peft_config types.")
    parser.add_argument("--peft_config_type", type=str, choices=["PrefixTuning", "PromptTuning", "P-tuning", "Lora", "IA3"],
                        default="IA3", help="Type of peft_config to use.")
    # 添加 --output_dir 参数
    parser.add_argument("--output_dir", type=str, default="./annotation", help="Output directory for saving logs and models.")
    args = parser.parse_args()

    if args.peft_config_type == "PrefixTuning":
        peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20)
    elif args.peft_config_type == "PromptTuning":
        peft_config = PromptTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, prompt_tuning_init=PromptTuningInit.TEXT, num_virtual_tokens=20,
                                         prompt_tuning_init_text="Recommend a Java API sequence based on the following annotation",
                                         inference_mode=False, tokenizer_name_or_path=model_name_or_path)
    elif args.peft_config_type == "P-tuning":
        peft_config = PromptEncoderConfig(task_type=TaskType.SEQ_2_SEQ_LM, num_virtual_tokens=20, encoder_hidden_size=128)
    elif args.peft_config_type == "Lora":
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    elif args.peft_config_type == "IA3":
        peft_config = IA3Config(task_type=TaskType.SEQ_2_SEQ_LM, target_modules=["k", "v", "wo"], inference_mode=False, feedforward_modules=["wo"])
    else:
        raise ValueError(f"Unsupported peft_config_type: {args.peft_config_type}")

    train_model(peft_config)

