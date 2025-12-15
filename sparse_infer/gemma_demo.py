# LLaMA model with KIVI
import warnings
warnings.filterwarnings("ignore")
import torch
import random
from gemma_rheo import Gemma2ForCausalLM_RHEO
from transformers import Gemma2Config, AutoTokenizer
from datasets import load_dataset

# For reproducibility
random.seed(0)
torch.manual_seed(0)

GEMMA_PATH = "/home/dac/zjn_data/gemma-2"

config = Gemma2Config.from_pretrained(GEMMA_PATH)
print(config)


model = Gemma2ForCausalLM_RHEO.from_pretrained(
    # pretrained_model_name_or_path='meta-llama/Llama-2-7b-hf',
    pretrained_model_name_or_path=GEMMA_PATH,
    config=config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).cpu()

state_dict = torch.load("sparse_weights.pth", map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.cuda()
# exit()

enc = AutoTokenizer.from_pretrained(
    GEMMA_PATH, 
    use_fast=False, 
    trust_remote_code=True)

dataset = load_dataset('gsm8k', 'main')

print("here")
prompt = ''
for i in range(5):
    prompt += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
prompt += "Question: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?"
inputs = enc(prompt, return_tensors="pt").input_ids.cuda()

output = model.generate(inputs, max_new_tokens=96)
config_str = f"# prompt tokens: {inputs.shape[1]}"

print(prompt + "\n" + "=" * 10 + f'\n{config_str}\n' + "=" * 10 + "\nKiVi Output:")
print(enc.decode(output[0].tolist()[inputs.shape[1]:], skip_special_tokens=True))

# https://chatgpt.com/c/6853b2e9-1b1c-8010-a911-0e18900393d9