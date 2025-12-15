import argparse
import os 
import time 
import numpy as np
import json 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
from data import get_loaders 
import torch.nn as nn




# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids
    

    
    # print(model.seqlen)
    model.seqlen = 2048

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen
    

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)


        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    # torch.cuda.empty_cache()

    return ppl.item()

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "wikitext2"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl_test 


# print('torch', version('torch'))
# print('transformers', version('transformers'))
# print('accelerate', version('accelerate'))
# print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default="4:8",  choices=["unstructured", "4:8", "2:4", "32:64", "33:64", "31:64", "5:8", "3:8", "16:64", "24:64", "40:64", "48:64"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", 
                        "ablate_wanda_iter", "search", "pruner-zero", "ablate_prunerzero_seq", "ablate_prunerzero_iter"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    # sparse_list 
    parser.add_argument('--sparse_list', type=str, default=None, help='Path to save the sparse_list.')
    parser.add_argument('--depth', type=int, default=0, help='current search depth')
    parser.add_argument('--direction', type=int, default=0, help='0 for left, 1 for right')
    
    # gradient_path 
    parser.add_argument("--gradient_path", type=str, default=None, help="Path to save the gradient.")
    parser.add_argument("--json_tree", type=str, default="data/best_tree.json", help="Path to load the json tree.")
    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)


    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    print(model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    

    device = torch.device("cuda:0")

    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")


if __name__ == '__main__':
    main()

# python genetic_search_llama_ppl_wide_range.py --full_model_path /home/dac/zjn_data/llama-1b --sparse_model_path llama-1b --base_sparsity 32 --layer_num 16