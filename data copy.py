# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset, load_from_disk 

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    # traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    # testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    traindata = load_from_disk('./data/wikitext2_train')
    testdata = load_from_disk('./data/wikitext2_test')
    # print(testdata['text'][:10])

    print(f"train data size: {len(traindata)}, test data size: {len(testdata)}")
    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ultrachat(nsamples, seed, seqlen, tokenizer):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    # ds = ds.select(range(10000))  # 提前选一个较大的范围以备采样
    ds = ds.shuffle(seed=42).select(range(10000))

    def preprocess(example):
        # processed = {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
        processed_opt = {"text": example["messages"][0]['content']}
        
        return processed_opt
    ds = ds.map(preprocess)
    random.seed(seed)
    samples = []
    idx = 0
    for _ in range(nsamples):
        idx+=1
        while True:
            i = random.randint(0, len(ds) - 1)
            tokenized = tokenizer(ds[i]["text"], return_tensors="pt", add_special_tokens=False)
            # print(ds[i]["text"], tokenized.input_ids.shape)

            # print(tokenized.input_ids.shape)
            if tokenized.input_ids.shape[1] > seqlen:
                break

        input_ids = tokenized.input_ids[0]
        start_idx = random.randint(0, input_ids.shape[0] - seqlen - 1)
        end_idx = start_idx + seqlen
        inp = input_ids[start_idx:end_idx]
        tar = inp.clone()
        tar[:-1] = -100  # 只计算最后一个 token 的 loss
        samples.append((inp.unsqueeze(0), tar.unsqueeze(0)))
    return samples, None



def get_ultrachat_(nsamples, seed, seqlen, tokenizer):
    # 加载数据（可以根据需要选取部分）
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=42).select(range(512))

    # 构造对话文本
    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
    ds = ds.map(preprocess)

    # tokenize（不添加特殊符号，不做 padding，也不截断）
    def tokenize(sample):
        return tokenizer(sample["text"], padding=True, max_length=4096, truncation=True, add_special_tokens=False)
    ds = ds.map(tokenize, remove_columns=ds.column_names)

    # 构造 input 和 target（target 的 label 只预测最后一个 token）
    samples = []
    for example in ds:
        ids = torch.tensor(example["input_ids"])
        target = ids.clone()
        target[:-1] = -100  # 只训练最后一个 token
        samples.append((ids.unsqueeze(0), target.unsqueeze(0)))

    return samples, None




# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    # traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    # traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    
    # traindata = load_from_disk('/home/zjnyly/repos/Pruner-Zero/c4_train')
    # valdata = load_from_disk('~/workspace/pruner-zero-private/data/c4_valid')
    
    traindata = load_dataset(
    'json',
    data_files={'train': '/home/zjnyly/repos/Pruner-Zero/c4_train/c4-train.00000-of-01024.json.gz'},
    split='train'
)   

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # # Prepare validation dataset
    # valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    # valenc = valenc.input_ids[:, :(256 * seqlen)]
    # valenc = TokenizerWrapper(valenc)
    return trainloader, None #valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if "ultra" in name:
        return get_ultrachat(nsamples, seed, seqlen, tokenizer)